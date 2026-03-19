r"""expectation
===========
This module evaluates the expectation values (population, <x> and <p>, energy, etc)
"""
import abc
import math
import typing

import torch

import constant
import evolve
import gp
import pes

torch.set_default_dtype(constant.DTYPE)
torch.set_default_device(constant.DEVICE)
torch.manual_seed(constant.SEED)


def normal_sample(
	num_points: int,
	mean: torch.Tensor,
	stddev: torch.Tensor
) -> torch.Tensor:
	r"""To create normally distributed point set based on given mean and variance

	Parameters
	----------
	num_points : int
		The number of points needed
	mean : torch.Tensor, shape of (PHASEDIM,)
		The center of the points
	stddev : torch.Tensor, shape of (PHASEDIM,)
		The standard deviation of the points

	Returns
	-------
	torch.Tensor, shape of (NUM_PTS, PHASEDIM)
		Normally distributed point test
	"""
	return torch.randn((num_points, mean.numel()), device=stddev.device) * stddev + mean



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
	def potential(self, model: pes.Potential) -> float:
		r"""To calculate the average potential energy

		Parameters
		----------
		model : pes.Potential
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

	def _gaussian_weight(self, coord: torch.Tensor, center: torch.Tensor, stddev: torch.Tensor) -> torch.Tensor:
		r"""The multi-dimensional gaussian function

		Parameters
		----------
		coord : torch.Tensor, shape of (..., N)
			The coordinates whose weights are calculated
		center : torch.Tensor, shape of (N,)
			Centers of the normal distributions
		stddev : torch.Tensor, shape of (N,)
			Standard deviation of the normal distributions.
			This function assumes no correlation

		Returns
		-------
		torch.Tensor, shape of (..., N)
			Density at the given coordinates
		"""
		return torch.exp(-torch.sum(((coord - center) / stddev) ** 2, -1) / 2.0) / ((2.0 * math.pi) ** self.config.DIM * stddev.prod())

	@property
	def num_pts(self) -> int:
		r"""_summary_

		Returns
		-------
		int
			_description_
		"""
		return self.__num_pts

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
			self.point_set[iTrig] = normal_sample(self.__num_pts, center, stddev)
			self.weight[iTrig] = self._gaussian_weight(self.point_set[iTrig], center, stddev) # N
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

	def potential(self, model: pes.Potential) -> float:
		return torch.sum(torch.mean(
			model.adiabatic_potential(self.point_set[self.DIAGONAL_TRIL_INDEX, :, :self.config.DIM])[self.config.PES_RANGE, :, self.config.PES_RANGE]
			* self.density[self.DIAGONAL_TRIL_INDEX, ...].real
			/ self.weight[self.DIAGONAL_TRIL_INDEX, ...],
			-1)
		).item()

	def purity(self) -> torch.Tensor:
		flatten_lower_trig: torch.Tensor = torch.zeros(self.config.NUM_PES, self.config.NUM_PES, dtype=torch.double)
		flatten_lower_trig[self.config.TRIL_ROW_INDICES, self.config.TRIL_COL_INDICES] = torch.mean((self.density.real ** 2 + self.density.imag ** 2) / self.weight, -1)
		return self.PURITY_FACTOR * (flatten_lower_trig + torch.tril(flatten_lower_trig, -1).T)


class EvolvingPointsMCAverage(MonteCarloAverage):
	r"""To calculate average by Monte Carlo estimate too,
	but using points evolving forward with same weights

	Parameters
	----------
	config : pes.ModelConfig
		Configuration of the model
	num_pts : int
		The number of points for monte carlo
	init_dist : pes.InitialDistribution
		Initial distribution to generate points, density, and weights

	Methods
	-------
	evolve(mass, dt, predictor)
		To evolve the coordinates, and density if applicable
	update_density(predictor)
		To update the density using the predictor if the density is not evolved
	"""
	__slots__: tuple = ()

	def __init__(
		self,
		config: pes.ModelConfig,
		num_pts: int,
		init_dist: pes.InitialDistribution,
		stddev: torch.Tensor | None = None
	):
		super().__init__(config, num_pts)
		if stddev is None:
			stddev = init_dist.sigma_r0
		pts: typing.Final[torch.Tensor] = normal_sample(num_pts, init_dist.r0, stddev)
		den: typing.Final[torch.Tensor] = init_dist(pts)
		self.point_set[:] = pts
		self.density = den[:, self.config.TRIL_ROW_INDICES, self.config.TRIL_COL_INDICES].T
		self.weight[:] = self._gaussian_weight(pts, init_dist.r0, stddev)

	def evolve(
		self,
		model: pes.Potential,
		mass: torch.Tensor,
		dt: float,
		predictor: constant.Predictor
	) -> None:
		r"""To evolve the coordinates and density

		Parameters
		----------
		model : pes.Potential
			Quantities derived from potential
		mass : torch.Tensor, shape of (DIM,)
			Mass of classical degree of freedom
		dt : float
			Time interval
		predictor : constant.Predictor
			It predicts the density matrix element based on given coordinates and element index
		"""
		evolve.evolve(model, [ps for ps in self.point_set], [den for den in self.density], mass, dt, predictor)


@typing.final
class AnalyticalAverager(Averager):
	r"""Using analytical integral of GPR to estimate averages

	Parameters
	----------
	config : pes.ModelConfig
		Configuration of the model
	pred : GPRPredictors
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
			result[iPES] = pred.lengthscale.prod().item() * pred.k_inv_y.sum().item()
		return result * self.__AVERAGE_CONSTANT

	def coordinates(self) -> torch.Tensor:
		result: torch.Tensor = torch.zeros(self.config.PHASEDIM)
		for iPES in range(self.config.NUM_PES):
			ElementIndex: int = iPES * self.config.NUM_PES + iPES
			pred: gp.SinglePredictor = self.__predictors[ElementIndex]
			result += pred.lengthscale.prod().item() * (pred.k_inv_y[:, None] * pred.x_ind).sum(0)
		return result * self.__AVERAGE_CONSTANT

	def square_coordinates(self) -> torch.Tensor:
		result: torch.Tensor = torch.zeros((self.config.PHASEDIM, self.config.PHASEDIM))
		for iPES in range(self.config.NUM_PES):
			ElementIndex: int = iPES * self.config.NUM_PES + iPES
			pred: gp.SinglePredictor = self.__predictors[ElementIndex]
			result += pred.lengthscale.prod().item() * (
				(pred.k_inv_y[:, None, None] * pred.x_ind[:, :, None] * pred.x_ind[:, None, :]).sum(0)
				+ pred.k_inv_y.sum() * torch.diagflat(pred.lengthscale ** 2))
		return result * self.__AVERAGE_CONSTANT

	def covariance(self) -> torch.Tensor:
		return super().covariance()

	def potential(self, model: pes.Potential) -> float:
		return math.nan

	def purity(self) -> torch.Tensor:
		result: torch.Tensor = torch.empty(self.config.NUM_PES, self.config.NUM_PES)
		for iPES in range(self.config.NUM_PES):
			for jPES in range(self.config.NUM_PES):
				pred: gp.SinglePredictor = self.__predictors[iPES * self.config.NUM_PES + jPES]
				result[iPES, jPES] = (math.pi ** self.config.DIM) * pred.lengthscale.prod().item() * (pred.k_inv_y @ gp.rbf(pred.lengthscale * math.sqrt(2.0), pred.x_ind, pred.x_ind) @ pred.k_inv_y).item()
		return self.PURITY_FACTOR * (result + result.T - torch.diag(torch.diag(result)))


@typing.final
class Points(EvolvingPointsMCAverage):
	r"""The class to tract trajectories and generate inducing points

	Parameters
	----------
	config : pes.ModelConfig
		Configuration of the model
	num_pts : int
		The number of inducing points
	init_dist : pes.InitialDistribution
		Initial distribution to generate points, density, and weights
	init_stddev : torch.Tensor
		The standard deviation of the initial points, used to generate the initial point set
	model : pes.Potential
		Quantities derived from potential
	mass : torch.Tensor, shape of (DIM,)
		Mass of classical degree of freedom

	Attributes
	----------
	ind_pts : torch.Tensor, dtype of double, shape of (NUM_TRIG, NUM_PT)
		The inducing point set

	Methods
	-------
	evolve(model, mass, dt, predictor)
		To evolve the coordinates and density
	adjust_weight(model, mass)
		To adjust the trajectory density by conservation constraints
	"""
	__slots__: typing.Final[tuple] = ("__last_purity_lambda", "__init_E")
	__last_purity_lambda: float
	__init_E: typing.Final[float]

	def __init__(
		self,
		config: pes.ModelConfig,
		num_pts: int,
		init_dist: pes.InitialDistribution,
		init_stddev: torch.Tensor,
		model: pes.Potential,
		mass: torch.Tensor,
	) -> None:
		super().__init__(config, num_pts, init_dist, init_stddev)
		self.__last_purity_lambda = 1.0
		self.__init_E = self.potential(model) + self.kinetic(mass)

	@property
	def scale(self) -> torch.Tensor:
		r"""To get the rescaling factor for each element, which is the inverse of the maximum density of the points

		Returns
		-------
		torch.Tensor, shape of (NUM_ELM,)
			The rescaling factor
		"""
		result: torch.Tensor = torch.empty(self.config.NUM_ELM)
		for iTrig, iPES, jPES, iElement in zip(self.config.TRIG_RANGE, self.config.TRIL_ROW_INDICES, self.config.TRIL_COL_INDICES, self.config.TRIL_ELEMENT_INDICES):
			if iPES == jPES:
				result[iElement] = 1.0 / self.density[iTrig].real.abs().max()
			else:
				result[iElement] = 1.0 / self.density[iTrig].imag.abs().max()
				result[jPES * self.config.NUM_PES + iPES] = 1.0 / self.density[iTrig].real.abs().max()
		return result

	def evolve(
		self,
		model: pes.Potential,
		mass: torch.Tensor,
		dt: float,
		predictor: constant.Predictor
	) -> None:
		r"""To evolve the coordinates and density

		Parameters
		----------
		model : pes.Potential
			Quantities derived from potential
		mass : torch.Tensor, shape of (DIM,)
			Mass of classical degree of freedom
		dt : float
			Time interval
		predictor : constant.Predictor
			It predicts the density matrix element based on given coordinates and element index
		"""
		super().evolve(model, mass, dt, predictor)

	def adjust_weight(self, model: pes.Potential, mass: torch.Tensor) -> None:
		r"""To adjust the trajectory density by conservation constraints

		Parameters
		----------
		model : pes.Potential
			Quantities derived from potential
		mass : torch.Tensor, shape of (DIM,)
			Mass of classical degree of freedom
		"""
		vec_ppl: typing.Final[torch.Tensor] = 1.0 / self.num_pts / self.weight[self.DIAGONAL_TRIL_INDEX, ...] # shape of (NUM_PES, NUM_PT)
		vec_eng: typing.Final[torch.Tensor] = (
			model.adiabatic_potential(self.point_set[self.DIAGONAL_TRIL_INDEX, :, :self.config.DIM])[self.config.PES_RANGE, :, self.config.PES_RANGE] # potential
			+ (torch.square(self.point_set[self.DIAGONAL_TRIL_INDEX, :, self.config.DIM:]) / mass).sum(-1) # kinetic energy
		) / self.weight[self.DIAGONAL_TRIL_INDEX, ...] / self.num_pts # shape of (NUM_PES, NUM_PT)
		tril_diag: typing.Final[torch.Tensor] = torch.tensor(self.config.TRIL_ROW_INDICES) == torch.tensor(self.config.TRIL_COL_INDICES)
		mat_prt: typing.Final[torch.Tensor] = self.PURITY_FACTOR / self.num_pts * torch.diag_embed(torch.where(tril_diag, 1.0, 2.0)[:, torch.newaxis] / self.weight) # off-diagonal has a factor of 2, shape of (NUM_TRIL, NUM_PT, NUM_PT)

		# solve for lambda_S
		def equation(lambda_S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
			r"""The purity equation, solve for lambda_S from this equation

			Parameters
			----------
			lambda_S : torch.Tensor
				Lagrange multiplier for purity conservation

			Returns
			-------
			tuple[torch.Tensor, torch.Tensor]
				Error of the equation, and the updated density
			"""
			# solve for lambda_1, and lambda_E:
			mat_inv: typing.Final[torch.Tensor] = 1.0 / (torch.eye(self.num_pts) - lambda_S * mat_prt) # reverse diagonal, shape of (NUM_TRIL, NUM_PT, NUM_PT)
			lambda_1E: typing.Final[torch.Tensor] = torch.linalg.solve(
				torch.stack((
					torch.stack(((vec_ppl[:, torch.newaxis, :] @ mat_inv[tril_diag] @ vec_ppl[:, :, torch.newaxis]).sum(), (vec_ppl[:, torch.newaxis, :] @ mat_inv[tril_diag] @ vec_eng[:, :, torch.newaxis]).sum())),
					torch.stack(((vec_eng[:, torch.newaxis, :] @ mat_inv[tril_diag] @ vec_ppl[:, :, torch.newaxis]).sum(), (vec_eng[:, torch.newaxis, :] @ mat_inv[tril_diag] @ vec_eng[:, :, torch.newaxis]).sum())),
				)),
				torch.stack((1.0 - (vec_ppl[:, torch.newaxis, :] @ mat_inv[tril_diag] @ self.density[:, :, torch.newaxis].real).sum(), self.__init_E - (vec_eng[:, torch.newaxis, :] @ mat_inv[tril_diag] @ self.density[:, :, torch.newaxis].real).sum()))
			)
			# solve for update z:
			updated_den: typing.Final[torch.Tensor] = (mat_inv @ self.density.index_add(0, torch.arange(self.config.NUM_PES) * (torch.arange(self.config.NUM_PES) + 3) // 2, lambda_1E[0] * vec_ppl + lambda_1E[1] * vec_eng)[..., torch.newaxis])[..., 0] # shape of (NUM_TRIL, NUM_PT)
			return (updated_den.conj()[:, torch.newaxis, :] @ mat_prt @ updated_den[:, :, torch.newaxis]).real.sum() - 1.0, updated_den

		# use Newton downhill method
		lambda_S: torch.Tensor = torch.full((1,), self.__last_purity_lambda, requires_grad=True)
		loss, updated_den = equation(lambda_S)
		grad: torch.Tensor = torch.autograd.grad(loss, lambda_S)[0]
		while loss.abs().item() > gp.Optimizer.FTOL:
			change: float = loss.item() / grad.item()
			lr = 1.0
			new_loss, _ = equation((lambda_S - lr * change).detach())
			while new_loss.abs().item() > loss.abs().item():
				lr /= 2.0
				new_loss, _ = equation((lambda_S - lr * change).detach())
			lambda_S = (lambda_S - lr * change).detach().requires_grad_()
			loss, updated_den = equation(lambda_S)
			grad = torch.autograd.grad(loss, lambda_S)[0]

		self.__last_purity_lambda = lambda_S.item()
		self.density = updated_den.detach()
