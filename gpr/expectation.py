"""
expectation
===========
This module evaluates the expectation values (population, <x> and <p>, energy, etc)
"""
import abc
import collections.abc
import copy
import os
import math
import sys
import typing

import numpy as np
import numpy.typing as npt
import torch

sys.path.append(os.path.dirname(__file__))

import evolve
import gp
import pes
import point

PURITY_FACTOR: typing.Final = (2.0 * math.pi * pes.HBAR) ** pes.DIM


class Averager(abc.ABC):
	"""
	To calculate averages

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
	@abc.abstractmethod
	def population(self) -> npt.NDArray[np.double]:
		"""
		To calculate the population on each potential energy surfaces

		Returns
		-------
		npt.NDArray[np.double]
			Population on each surfaces
		"""

	@abc.abstractmethod
	def coordinates(self) -> npt.NDArray[np.double]:
		"""
		To calculate the average of phase space coordinates

		Returns
		-------
		npt.NDArray[np.double]
			Average positions and momenta
		"""

	@abc.abstractmethod
	def square_coordinates(self) -> npt.NDArray[np.double]:
		"""
		To calculate averages of product of phase space coordinates

		Returns
		-------
		npt.NDArray[np.double]
			A PHASEDIM-by-PHASEDIM matrix, whose ij term is <x_i*x_j>
		"""

	def covariance(self) -> npt.NDArray[np.double]:
		"""
		To calculate the covariance between phase space coordinates

		Returns
		-------
		npt.NDArray[np.double]
			PHASEDIM-by-PHASEDIM covariance matrix
		"""
		coord_ave: typing.Final[npt.NDArray[np.double]] = self.coordinates()
		return self.square_coordinates() + (self.population().sum() - 2.0) * coord_ave[:, np.newaxis] * coord_ave[np.newaxis, :]

	def standard_deviation(self) -> npt.NDArray[np.double]:
		"""
		To calculate the standard deviation of each phase space dimension

		Returns
		-------
		npt.NDArray[np.double]
			Standard deviation of each dimension
		"""
		return np.sqrt(np.diagonal(self.covariance()))

	@abc.abstractmethod
	def potential(self) -> float:
		"""
		To calculate the average potential energy

		Returns
		-------
		float
			Average potential energy
		"""

	def kinetic(self, mass: npt.NDArray[np.double]) -> float:
		"""
		To calculate the average kinetic energy

		Parameters
		----------
		mass : npt.NDArray[np.double]
			Mass of classical degree of freedom

		Returns
		-------
		float
			Average kinetic energy
		"""
		return (self.square_coordinates()[np.arange(pes.DIM, pes.PHASEDIM), np.arange(pes.DIM, pes.PHASEDIM)] / mass).sum() / 2.0

	@abc.abstractmethod
	def purity(self) -> npt.NDArray[np.double]:
		"""
		To calculate contribution to purity of each element

		Returns
		-------
		npt.NDArray[np.double]
			Purity of each element
		"""

class MonteCarloAverage(Averager):
	"""
	Using Monte Carlo to estimate averages

	Parameters
	----------
	num_pts : int
		The number of points
	center : npt.NDArray[np.double]
		The center of the points
	stddev : npt.NDArray[np.double]
		The standard deviation of the points
	predictor : collections.abc.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]
		The function that gives the density matrix element at corresponding phase points

	Attributes
	----------
	DIAGONAL_TRIL_INDEX : npt.NDArray[np.int_]
		Lower-triangular index of diagonal elements

	Methods
	-------
	update_pts(ref_pts, predictor): To predict density at next time step
	"""
	DIAGONAL_TRIL_INDEX: typing.Final[npt.NDArray[np.int_]] = pes.flatten_tril_index[np.arange(pes.NUM_PES), np.arange(pes.NUM_PES)]
	__slots__: tuple = ("__num_pts", "point_set", "weight", "density")

	def __init__(self, num_pts: int):
		self.__num_pts: typing.Final[int] = num_pts
		self.point_set: npt.NDArray[np.double] = np.empty((pes.NUM_TRIG, self.__num_pts, pes.PHASEDIM))
		self.weight: npt.NDArray[np.double] = np.empty((pes.NUM_TRIG, self.__num_pts))
		self.density: npt.NDArray[np.cdouble] = np.empty((pes.NUM_TRIG, self.__num_pts), np.cdouble)

	def update_pts(
		self,
		ref_pts: list[npt.NDArray[np.double]],
		predictor: collections.abc.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]
	):
		"""
		To update the point set used

		Parameters
		----------
		ref_pts : list[npt.NDArray[np.double]], len of NUM_TRIG, each of shape (NUM_PTS, PHASEDIM)
			Current points, used to estimate average and variance
		predictor : collections.abc.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]
			Used to predict the density of the points
		"""
		for iPES, jPES, iTrig in zip(pes.tril_row_indices, pes.tril_col_indices, np.arange(pes.NUM_TRIG)):
			center: npt.NDArray[np.double] = np.mean(ref_pts[iTrig], 0)
			stddev: npt.NDArray[np.double] = 1.5 * np.std(ref_pts[iTrig], 0)
			self.point_set[iTrig] = point.normal_sample(self.__num_pts, center, stddev)
			self.weight[iTrig] = np.exp(-np.sum(((self.point_set[iTrig] - center) / stddev) ** 2, -1) / 2.0) / ((2.0 * math.pi) ** pes.DIM * stddev.prod()) # N
			self.density[iTrig] = predictor(self.point_set[iTrig], iPES * pes.NUM_PES + jPES)

	def population(self) -> npt.NDArray[np.double]:
		return np.average(self.density[MonteCarloAverage.DIAGONAL_TRIL_INDEX].real / self.weight[MonteCarloAverage.DIAGONAL_TRIL_INDEX], -1)

	def coordinates(self) -> npt.NDArray[np.double]:
		return np.sum(np.average(np.moveaxis(self.point_set[MonteCarloAverage.DIAGONAL_TRIL_INDEX], -1, 0) * self.density[MonteCarloAverage.DIAGONAL_TRIL_INDEX].real / self.weight[MonteCarloAverage.DIAGONAL_TRIL_INDEX], axis=-1), 1)

	def square_coordinates(self) -> npt.NDArray[np.double]:
		return np.sum(
			np.average(
				self.density[MonteCarloAverage.DIAGONAL_TRIL_INDEX, :, np.newaxis, np.newaxis].real
				* self.point_set[MonteCarloAverage.DIAGONAL_TRIL_INDEX, :, :, np.newaxis]
				* self.point_set[MonteCarloAverage.DIAGONAL_TRIL_INDEX, :, np.newaxis, :]
				/ self.weight[MonteCarloAverage.DIAGONAL_TRIL_INDEX, :, np.newaxis, np.newaxis],
				1
			),
			0
		)

	def covariance(self) -> npt.NDArray[np.double]:
		coord_ave: typing.Final[npt.NDArray[np.double]] = self.coordinates()
		return np.sum(
			np.average(
				self.density[MonteCarloAverage.DIAGONAL_TRIL_INDEX, :, np.newaxis, np.newaxis].real
				* (self.point_set[MonteCarloAverage.DIAGONAL_TRIL_INDEX, :, :, np.newaxis] - coord_ave[:, np.newaxis])
				* (self.point_set[MonteCarloAverage.DIAGONAL_TRIL_INDEX, :, np.newaxis, :] - coord_ave[np.newaxis, :])
				/ self.weight[MonteCarloAverage.DIAGONAL_TRIL_INDEX, :, np.newaxis, np.newaxis],
				1
			),
			0
		)

	def potential(self) -> float:
		return np.sum(np.average(pes.adiabatic_potential(self.point_set[MonteCarloAverage.DIAGONAL_TRIL_INDEX, :, :pes.DIM])[np.arange(pes.NUM_PES), :, np.arange(pes.NUM_PES)] * self.density[MonteCarloAverage.DIAGONAL_TRIL_INDEX].real / self.weight[MonteCarloAverage.DIAGONAL_TRIL_INDEX], -1))

	def purity(self) -> npt.NDArray[np.double]:
		return PURITY_FACTOR * pes.lower_triangular_to_full(np.average((self.density.real ** 2 + self.density.imag ** 2) / self.weight, -1))


class EvolvingPointsMCAverage(MonteCarloAverage):
	"""
	To calculate average by Monte Carlo estimate too,
	but using points evolving forward with same weights

	Parameters
	----------
	num_pts : int
		The number of points for monte carlo
	init_dist : pes.InitialDistribution
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

	def __init__(self, num_pts: int, init_dist: pes.InitialDistribution, evolve_coordinates_only: bool = False):
		super().__init__(num_pts)
		self.__evolve_coordinates_only: typing.Final[bool] = evolve_coordinates_only
		self.point_set[:] = point.normal_sample(num_pts, init_dist.r0, init_dist.sigma_r0)
		for i, ElementIndex in enumerate(pes.tril_element_indices):
			self.density[i] = init_dist(self.point_set[i], ElementIndex)
			self.weight[i] = np.abs(self.density[i]) / init_dist.weight[ElementIndex // pes.NUM_PES, ElementIndex % pes.NUM_PES]

	def evolve(
		self,
		mass: npt.NDArray[np.double],
		dt: float,
		predictor: collections.abc.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]
	) -> None:
		"""
		To evolve the coordinates, and density if applicable

		Parameters
		----------
		mass : npt.NDArray[np.double], shape of (DIM,)
			Mass of classical degree of freedom
		dt : float
			Time interval
		predictor : collections.abc.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]
			It predicts the density matrix element based on given coordinates and element index
		"""
		if self.__evolve_coordinates_only:
			for pts, row_idx, col_idx in zip(self.point_set, pes.tril_row_indices, pes.tril_col_indices):
				pts[:, :pes.DIM], pts[:, pes.DIM:] = evolve.evolve_coordinates_adiabatically(
					pts[:, :pes.DIM],
					pts[:, pes.DIM:],
					mass,
					dt,
					evolve.Direction.FORWARD,
					row_idx,
					col_idx
				)
		else:
			evolve.evolve([ps for ps in self.point_set], [den for den in self.density], mass, dt, predictor)

	def update_density(self, predictor: collections.abc.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]) -> None:
		"""
		To update the density using the predictor if the density is not evolved

		Parameters
		----------
		predictor : collections.abc.Callable[[npt.NDArray[np.double], int, bool], npt.NDArray[np.cdouble]]
			It predicts the density matrix element based on given coordinates and element index
		"""
		if self.__evolve_coordinates_only:
			for i, ElementIndex in enumerate(pes.tril_element_indices):
				self.density[i] = predictor(self.point_set[i], ElementIndex)


class AnalyticalAverager(Averager):
	"""
	Using analytical integral of GPR to estimate averages

	Attributes
	----------
	AVERAGE_CONSTANT : float
		The prefactor used in calculation of averages

	Parameters
	----------
	pred : gp.GPRPredictors
		GPR predictors
	"""
	__AVERAGE_CONSTANT: typing.Final = (2.0 * math.pi) ** pes.DIM
	__slots__: tuple = ("__predictors",)

	def __init__(self, pred: gp.GPRPredictors):
		self.__predictors: gp.GPRPredictors = pred

	def population(self) -> npt.NDArray[np.double]:
		result: npt.NDArray[np.double] = np.empty(pes.NUM_PES, np.double)
		for iPES in range(pes.NUM_PES):
			ElementIndex: int = iPES * pes.NUM_PES + iPES
			pred: gp.SinglePredictor = self.__predictors[ElementIndex]
			result[iPES] = pred.model.cov.lengthscale.prod().item() * pred.k_inv_y.sum().item()
		return result * __class__.__AVERAGE_CONSTANT

	def coordinates(self) -> npt.NDArray[np.double]:
		result: npt.NDArray[np.double] = np.zeros(pes.PHASEDIM, np.double)
		for iPES in range(pes.NUM_PES):
			ElementIndex: int = iPES * pes.NUM_PES + iPES
			pred: gp.SinglePredictor = self.__predictors[ElementIndex]
			result += pred.model.cov.lengthscale.prod().item() * (pred.k_inv_y[:, None] * pred.get_training_features()).sum(0).detach().numpy()
		return result * __class__.__AVERAGE_CONSTANT

	def square_coordinates(self) -> npt.NDArray[np.double]:
		result: npt.NDArray[np.double] = np.zeros((pes.PHASEDIM, pes.PHASEDIM))
		for iPES in range(pes.NUM_PES):
			ElementIndex: int = iPES * pes.NUM_PES + iPES
			pred: gp.SinglePredictor = self.__predictors[ElementIndex]
			result += pred.model.cov.lengthscale.prod().item() * (
				(pred.k_inv_y[:, None, None] * pred.get_training_features()[:, :, None] * pred.get_training_features()[:, None, :]).sum(0)
				+ pred.k_inv_y.sum() * torch.diagflat(pred.model.cov.lengthscale ** 2)).detach().numpy()
		return result * __class__.__AVERAGE_CONSTANT

	def covariance(self) -> npt.NDArray[np.double]:
		return super().covariance()

	def potential(self) -> float:
		return math.nan

	def purity(self) -> npt.NDArray[np.double]:
		result: npt.NDArray[np.double] = np.empty((pes.NUM_PES, pes.NUM_PES), np.double)
		for iPES in range(pes.NUM_PES):
			for jPES in range(pes.NUM_PES):
				pred: gp.SinglePredictor = self.__predictors[iPES * pes.NUM_PES + jPES]
				model: gp.GP = copy.deepcopy(pred.model)
				with torch.no_grad():
					model.cov.lengthscale = model.cov.lengthscale * math.sqrt(2.0)
				result[iPES, jPES] = (math.pi ** pes.DIM) * pred.model.cov.lengthscale.prod().item() * (pred.k_inv_y @ model.cov(pred.get_training_features()).to_dense() @ pred.k_inv_y).item()
		return PURITY_FACTOR * (result + result.T - np.diag(np.diag(result)))
