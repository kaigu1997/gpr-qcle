r"""expectation
===========
This module evaluates the expectation values (population, <x> and <p>, energy, etc)
"""
import abc
import copy
import math
import typing

import torch

import constant
import evolve
import gp
import pes
import point


class Averager(abc.ABC):
	r"""To calculate averages

	Parameters
	----------
	config : pes.ModelConfig
		Configuration of the model

	Attributes
	----------
	config : pes.ModelConfig
		Configuration of the model
	PURITY_FACTOR: float
		Pre-factor when calculating purity

	Methods
	----------
	population()
		To calculate the population on each potential energy surfaces
	coordinates()
		To calculate the average positions and momenta
	potential()
		To calculate the average potential energy
	kinetic()
		To calculate the average kinetic energy
	purity()
		To calculate contribution to purity of each element
	"""
	__slots__: typing.ClassVar[tuple] = ("config", "PURITY_FACTOR")
	config: typing.Final[pes.ModelConfig]
	PURITY_FACTOR: typing.Final[float]

	def __init__(self, config: pes.ModelConfig):
		self.config = config
		self.PURITY_FACTOR = (2.0 * math.pi * constant.HBAR) ** config.DIM

	@abc.abstractmethod
	def population(self) -> torch.Tensor:
		r"""To calculate the population on each potential energy surfaces

		Returns
		-------
		torch.Tensor
			Population on each surfaces
		"""

	@abc.abstractmethod
	def coordinates(self) -> torch.Tensor:
		r"""To calculate the average of phase space coordinates

		Returns
		-------
		torch.Tensor
			Average positions and momenta
		"""

	@abc.abstractmethod
	def square_coordinates(self) -> torch.Tensor:
		r"""To calculate averages of product of phase space coordinates

		Returns
		-------
		torch.Tensor
			A PHASEDIM-by-PHASEDIM matrix, whose ij term is <x_i*x_j>
		"""

	def covariance(self) -> torch.Tensor:
		r"""To calculate the covariance between phase space coordinates

		Returns
		-------
		torch.Tensor
			PHASEDIM-by-PHASEDIM covariance matrix
		"""
		coord_ave: typing.Final[torch.Tensor] = self.coordinates()
		return self.square_coordinates() + (self.population().sum() - 2.0) * coord_ave[:, torch.newaxis] * coord_ave[torch.newaxis, :]

	def standard_deviation(self) -> torch.Tensor:
		r"""To calculate the standard deviation of each phase space dimension

		Returns
		-------
		torch.Tensor
			Standard deviation of each dimension
		"""
		return torch.sqrt(torch.diagonal(self.covariance()))

	@abc.abstractmethod
	def potential(self, potential_model: pes.Potential) -> float:
		r"""To calculate the average potential energy

		Parameters
		----------
		potential : pes.Potential
			Quantities derived from potential

		Returns
		-------
		float
			Average potential energy
		"""

	def kinetic(self, mass: torch.Tensor) -> float:
		r"""To calculate the average kinetic energy

		Parameters
		----------
		mass : torch.Tensor
			Mass of classical degree of freedom

		Returns
		-------
		float
			Average kinetic energy
		"""
		return (self.square_coordinates()[self.config.P_DIM_RANGE, self.config.P_DIM_RANGE] / mass).sum().item() / 2.0

	@abc.abstractmethod
	def purity(self) -> torch.Tensor:
		r"""To calculate contribution to purity of each element

		Returns
		-------
		torch.Tensor
			Purity of each element
		"""

class MonteCarloAverage(Averager):
	r"""Using Monte Carlo to estimate averages

	Parameters
	----------
	config : pes.ModelConfig
		Configuration of the model
	num_pts : int
		The number of points

	Attributes
	----------
	DIAGONAL_TRIL_INDEX : torch.Tensor
		Lower-triangular index of diagonal elements
	point_set : torch.Tensor, double, shape of (NUM_TRIG, NUM_PTS, PHASEDIM)
		Sample points of each density matrix element for Monte Carlo
	weight : torch.Tensor, double, shape of (NUM_TRIG, NUM_PTS)
		Sample weight of each point
	density : torch.Tensor, complex double, shape of (NUM_TRIG, NUM_PTS)
		Density Matrix element of each point

	Methods
	-------
	update_pts(ref_pts, predictor): To predict density at next time step
	"""
	__slots__: tuple = ("DIAGONAL_TRIL_INDEX", "__num_pts", "point_set", "weight", "density")
	DIAGONAL_TRIL_INDEX: typing.Final[tuple[int, ...]]
	__num_pts: typing.Final[int]
	point_set: torch.Tensor
	weight: torch.Tensor
	density: torch.Tensor

	def __init__(self, config: pes.ModelConfig, num_pts: int):
		super().__init__(config)
		self.DIAGONAL_TRIL_INDEX = tuple(self.config.FLATTEN_TRIL_INDEX[i * (self.config.NUM_PES + 1)] for i in self.config.PES_RANGE)
		self.__num_pts = num_pts
		self.point_set = torch.empty((self.config.NUM_TRIG, self.__num_pts, self.config.PHASEDIM))
		self.weight = torch.empty((self.config.NUM_TRIG, self.__num_pts))
		self.density = torch.empty((self.config.NUM_TRIG, self.__num_pts), dtype=torch.cdouble)

	def update_pts(
		self,
		ref_pts: list[torch.Tensor],
		predictor: constant.Predictor
	) -> None:
		r"""To update the point set used

		Parameters
		----------
		ref_pts : list[torch.Tensor], len of NUM_TRIG, each of shape (NUM_PTS, PHASEDIM)
			Current points, used to estimate average and variance
		predictor : self.config.Predictor
			Used to predict the density of the points
		"""
		for iPES, jPES, iTrig in zip(self.config.TRIL_ROW_INDICES, self.config.TRIL_COL_INDICES, self.config.TRIG_RANGE):
			center: torch.Tensor = torch.mean(ref_pts[iTrig], 0)
			stddev: torch.Tensor = 1.5 * torch.std(ref_pts[iTrig], 0)
			self.point_set[iTrig] = point.normal_sample(self.__num_pts, center, stddev)
			self.weight[iTrig] = torch.exp(-torch.sum(((self.point_set[iTrig] - center) / stddev) ** 2, -1) / 2.0) / ((2.0 * math.pi) ** self.config.DIM * stddev.prod()) # N
			self.density[iTrig] = predictor(self.point_set[iTrig], iPES * self.config.NUM_PES + jPES)

	def population(self) -> torch.Tensor:
		return torch.mean(self.density[self.DIAGONAL_TRIL_INDEX, ...].real / self.weight[self.DIAGONAL_TRIL_INDEX, ...], -1)

	def coordinates(self) -> torch.Tensor:
		return torch.sum(
			torch.mean(
				torch.moveaxis(self.point_set[self.DIAGONAL_TRIL_INDEX, ...], -1, 0)
				* self.density[self.DIAGONAL_TRIL_INDEX, ...].real
				/ self.weight[self.DIAGONAL_TRIL_INDEX, ...],
				-1
			),
			1
		)

	def square_coordinates(self) -> torch.Tensor:
		return torch.sum(
			torch.mean(
				self.density[self.DIAGONAL_TRIL_INDEX, :, torch.newaxis, torch.newaxis].real
				* self.point_set[self.DIAGONAL_TRIL_INDEX, :, :, torch.newaxis]
				* self.point_set[self.DIAGONAL_TRIL_INDEX, :, torch.newaxis, :]
				/ self.weight[self.DIAGONAL_TRIL_INDEX, :, torch.newaxis, torch.newaxis],
				1
			),
			0
		)

	def covariance(self) -> torch.Tensor:
		coord_ave: typing.Final[torch.Tensor] = self.coordinates()
		return torch.sum(
			torch.mean(
				self.density[self.DIAGONAL_TRIL_INDEX, :, torch.newaxis, torch.newaxis].real
				* (self.point_set[self.DIAGONAL_TRIL_INDEX, :, :, torch.newaxis] - coord_ave[:, torch.newaxis])
				* (self.point_set[self.DIAGONAL_TRIL_INDEX, :, torch.newaxis, :] - coord_ave[torch.newaxis, :])
				/ self.weight[self.DIAGONAL_TRIL_INDEX, :, torch.newaxis, torch.newaxis],
				1
			),
			0
		)

	def potential(self, potential_model: pes.Potential) -> float:
		return torch.sum(torch.mean(
			potential_model(
				self.point_set[self.DIAGONAL_TRIL_INDEX, :, :self.config.DIM],
				adiabatic_potential=True
			)["adiabatic_potential"][self.config.PES_RANGE, :, self.config.PES_RANGE]
			* self.density[self.DIAGONAL_TRIL_INDEX, ...].real
			/ self.weight[self.DIAGONAL_TRIL_INDEX, ...],
			-1)
		).item()

	def purity(self) -> torch.Tensor:
		flatten_lower_trig: torch.Tensor = torch.zeros(self.config.NUM_PES, self.config.NUM_PES, dtype=torch.double)
		flatten_lower_trig[self.config.TRIL_ROW_INDICES, self.config.TRIL_COL_INDICES] = torch.mean((self.density.real ** 2 + self.density.imag ** 2) / self.weight, -1)
		return self.PURITY_FACTOR * (flatten_lower_trig + torch.tril(flatten_lower_trig, -1).T)


@typing.final
class EvolvingPointsMCAverage(MonteCarloAverage):
	r"""To calculate average by Monte Carlo estimate too,
	but using points evolving forward with same weights

	Parameters
	----------
	config : pes.ModelConfig
		Configuration of the model
	num_pts : int
		The number of points for monte carlo
	init_dist : self.config.InitialDistribution
		Initial distribution to generate points, density, and weights
	evolve_coordinates_only : bool, optional
		Whether to evolve phase space coordinates only or with its density as well, by default False

	Methods
	-------
	evolve(mass, dt, predictor)
		To evolve the coordinates, and density if applicable
	update_density(predictor)
		To update the density using the predictor if the density is not evolved
	"""
	__slots__: tuple = ("__evolve_coordinates_only",)

	def __init__(
		self,
		config: pes.ModelConfig,
		num_pts: int,
		init_dist: pes.InitialDistribution,
		evolve_coordinates_only: bool = False
	):
		super().__init__(config, num_pts)
		self.__evolve_coordinates_only: typing.Final[bool] = evolve_coordinates_only
		pts: typing.Final[torch.Tensor] = point.normal_sample(num_pts, init_dist.r0, init_dist.sigma_r0)
		den: typing.Final[torch.Tensor] = init_dist(pts)
		self.point_set[:] = pts
		self.density = den[:, self.config.TRIL_ROW_INDICES, self.config.TRIL_COL_INDICES].T
		self.weight = torch.abs(self.density) / init_dist.weight[self.config.TRIL_ROW_INDICES, self.config.TRIL_COL_INDICES, torch.newaxis]

	def evolve(
		self,
		potential_model: pes.Potential,
		mass: torch.Tensor,
		dt: float,
		predictor: constant.Predictor
	) -> None:
		r"""To evolve the coordinates, and density if applicable

		Parameters
		----------
		mass : torch.Tensor, shape of (DIM,)
			Mass of classical degree of freedom
		dt : float
			Time interval
		predictor : self.config.Predictor
			It predicts the density matrix element based on given coordinates and element index
		"""
		if self.__evolve_coordinates_only:
			for pts, row_idx, col_idx in zip(self.point_set, self.config.TRIL_ROW_INDICES, self.config.TRIL_COL_INDICES):
				pts[:, :self.config.DIM], pts[:, self.config.DIM:] = evolve.evolve_coordinates_adiabatically(
					potential_model,
					pts[:, :self.config.DIM],
					pts[:, self.config.DIM:],
					mass,
					dt,
					evolve.Direction.FORWARD,
					row_idx,
					col_idx
				)
		else:
			evolve.evolve(potential_model, [ps for ps in self.point_set], [den for den in self.density], mass, dt, predictor)

	def update_density(self, predictor: constant.Predictor) -> None:
		r"""To update the density using the predictor if the density is not evolved

		Parameters
		----------
		predictor : self.config.Predictor
			It predicts the density matrix element based on given coordinates and element index
		"""
		if self.__evolve_coordinates_only:
			for i, ElementIndex in enumerate(self.config.TRIL_ELEMENT_INDICES):
				self.density[i] = predictor(self.point_set[i], ElementIndex)


@typing.final
class AnalyticalAverager(Averager):
	r"""Using analytical integral of GPR to estimate averages

	Parameters
	----------
	config : pes.ModelConfig
		Configuration of the model
	pred : gp.GPRPredictors
		GPR predictors
	"""
	__slots__: tuple = ("__AVERAGE_CONSTANT", "__predictors",)
	__AVERAGE_CONSTANT: typing.Final[float]
	__predictors: typing.Final[gp.GPRPredictors]

	def __init__(self, config: pes.ModelConfig, pred: gp.GPRPredictors):
		super().__init__(config)
		self.__AVERAGE_CONSTANT = (2.0 * math.pi) ** self.config.DIM
		self.__predictors = pred

	def population(self) -> torch.Tensor:
		result: torch.Tensor = torch.empty(self.config.NUM_PES)
		for iPES in range(self.config.NUM_PES):
			ElementIndex: int = iPES * self.config.NUM_PES + iPES
			pred: gp.SinglePredictor = self.__predictors[ElementIndex]
			result[iPES] = pred.model.cov.lengthscale.prod().item() * pred.k_inv_y.sum().item()
		return result * self.__AVERAGE_CONSTANT

	def coordinates(self) -> torch.Tensor:
		result: torch.Tensor = torch.zeros(self.config.PHASEDIM)
		for iPES in range(self.config.NUM_PES):
			ElementIndex: int = iPES * self.config.NUM_PES + iPES
			pred: gp.SinglePredictor = self.__predictors[ElementIndex]
			result += pred.model.cov.lengthscale.prod().item() * (pred.k_inv_y[:, None] * pred.get_training_features()).sum(0)
		return result * self.__AVERAGE_CONSTANT

	def square_coordinates(self) -> torch.Tensor:
		result: torch.Tensor = torch.zeros((self.config.PHASEDIM, self.config.PHASEDIM))
		for iPES in range(self.config.NUM_PES):
			ElementIndex: int = iPES * self.config.NUM_PES + iPES
			pred: gp.SinglePredictor = self.__predictors[ElementIndex]
			result += pred.model.cov.lengthscale.prod().item() * (
				(pred.k_inv_y[:, None, None] * pred.get_training_features()[:, :, None] * pred.get_training_features()[:, None, :]).sum(0)
				+ pred.k_inv_y.sum() * torch.diagflat(pred.model.cov.lengthscale ** 2))
		return result * self.__AVERAGE_CONSTANT

	def covariance(self) -> torch.Tensor:
		return super().covariance()

	def potential(self, potential_model: pes.Potential) -> float:
		return math.nan

	def purity(self) -> torch.Tensor:
		result: torch.Tensor = torch.empty(self.config.NUM_PES, self.config.NUM_PES)
		for iPES in range(self.config.NUM_PES):
			for jPES in range(self.config.NUM_PES):
				pred: gp.SinglePredictor = self.__predictors[iPES * self.config.NUM_PES + jPES]
				model: gp.GP = copy.deepcopy(pred.model)
				with torch.no_grad():
					model.cov.lengthscale = model.cov.lengthscale * math.sqrt(2.0)
				result[iPES, jPES] = (math.pi ** self.config.DIM) * pred.model.cov.lengthscale.prod().item() * (pred.k_inv_y @ model.cov(pred.get_training_features()).to_dense() @ pred.k_inv_y).item()
		return self.PURITY_FACTOR * (result + result.T - torch.diag(torch.diag(result)))
