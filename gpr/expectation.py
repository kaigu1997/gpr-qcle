"""
expectation
===========

This module evaluates the expectation values (population, <x> and <p>, energy, etc)
"""
import abc
import copy
import typing

import numpy as np
import numpy.typing as npt
import torch

import gp
import pes
import sample

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
		To calculate the average potential energy

		Returns
		-------
		npt.NDArray[np.double]
			Average positions and momenta
		"""

	@abc.abstractmethod
	def potential(self) -> float:
		"""
		To calculate the average potential energy

		Returns
		-------
		float
			Average potential energy
		"""

	@abc.abstractmethod
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
	predictor : typing.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]
		The function that gives the density matrix element at corresponding phase points

	Methods
	-------
	update_pts(ref_pts, predictor): To predict density at next time step
	"""
	def __init__(self, num_pts: int):
		self.num_pts: int = num_pts
		self.point_set: npt.NDArray[np.double] = np.empty((pes.NUM_ELM, num_pts, pes.PHASEDIM))
		self.weight: npt.NDArray[np.double] = np.empty((pes.NUM_ELM, num_pts))
		self.density: npt.NDArray[np.cdouble] = np.empty((pes.NUM_ELM, num_pts), np.cdouble)

	def update_pts(self, ref_pts: npt.NDArray[np.double], predictor: typing.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]):
		"""
		To update the point set used

		Parameters
		----------
		ref_pts : npt.NDArray[np.double]
			_description_
		predictor : typing.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]
			_description_
		"""
		for iPES in range(pes.NUM_PES):
			for jPES in range(iPES + 1):
				ElementIndex: int = iPES * pes.NUM_PES + jPES
				center: npt.NDArray[np.double] = np.mean(ref_pts[ElementIndex], 0)
				stddev: npt.NDArray[np.double] = 1.5 * np.std(ref_pts[ElementIndex], 0)
				self.point_set[ElementIndex] = sample.normal_sample(self.num_pts, center, stddev)
				self.weight[ElementIndex] = np.exp(-np.sum(((self.point_set[ElementIndex] - center) / stddev) ** 2, -1) / 2.0) / ((2.0 * np.pi) ** pes.DIM * stddev.prod()) # N
				self.density[ElementIndex] = predictor(self.point_set[ElementIndex], ElementIndex)
				if iPES != jPES:
					self.point_set[jPES * pes.NUM_PES + iPES] = self.point_set[ElementIndex]
					self.weight[jPES * pes.NUM_PES + iPES] = self.weight[ElementIndex]
					self.density[jPES * pes.NUM_PES + iPES] = np.conj(self.density[ElementIndex])

	def population(self) -> npt.NDArray[np.double]:
		"""
		To calculate the population on each potential energy surfaces

		Returns
		-------
		npt.NDArray[np.double]
			Population on each surfaces
		"""
		result: npt.NDArray[np.double] = np.empty(pes.NUM_PES, np.double)
		for iPES in range(pes.NUM_PES):
			ElementIndex: int = iPES * pes.NUM_PES + iPES
			result[iPES] = np.average(self.density.real[ElementIndex] / self.weight[ElementIndex])
		return result

	def coordinates(self) -> npt.NDArray[np.double]:
		"""
		To calculate the average potential energy

		Returns
		-------
		npt.NDArray[np.double]
			Average positions and momenta
		"""
		result: npt.NDArray[np.double] = np.zeros(pes.PHASEDIM, np.double)
		for iPES in range(pes.NUM_PES):
			ElementIndex: int = iPES * pes.NUM_PES + iPES
			result += np.average(np.swapaxes(self.point_set[ElementIndex], -1, -2) * self.density.real[ElementIndex] / self.weight[ElementIndex], axis=-1)
		return result

	def potential(self) -> float:
		"""
		To calculate the average potential energy

		Returns
		-------
		float
			Average potential energy
		"""
		result: float = 0.0
		for iPES in range(pes.NUM_PES):
			ElementIndex: int = iPES * pes.NUM_PES + iPES
			result += np.average(pes.adiabatic_potential(self.point_set[ElementIndex, :, :pes.DIM])[..., iPES] * self.density.real[ElementIndex] / self.weight[ElementIndex])
		return result

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
		result: float = 0.0
		for iPES in range(pes.NUM_PES):
			ElementIndex: int = iPES * pes.NUM_PES + iPES
			result += np.average(np.sum(self.point_set[ElementIndex, :, pes.DIM] ** 2 / mass, -1) * self.density.real[ElementIndex] / self.weight[ElementIndex])
		return result

	def purity(self) -> npt.NDArray[np.double]:
		"""
		To calculate contribution to purity of each element

		Returns
		-------
		npt.NDArray[np.double]
			Purity of each element
		"""
		return pes.PURITY_FACTOR * np.average((self.density.real ** 2 + self.density.imag ** 2) / self.weight, -1)


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
	AVERAGE_CONSTANT: float = (2.0 * np.pi) ** pes.DIM

	def __init__(self, pred: gp.GPRPredictors):
		self._predictors: gp.GPRPredictors = pred

	# @property
	# def predictors(self) -> gp.GPRPredictors:
	# 	return self._predictors

	# @predictors.setter
	# def predcitors(self, pred: gp.GPRPredictors) -> None:
	# 	self._predictors = pred

	def population(self) -> npt.NDArray[np.double]:
		"""
		To calculate the population on each potential energy surfaces

		Returns
		-------
		npt.NDArray[np.double]
			Population on each surfaces
		"""
		result: npt.NDArray[np.double] = np.empty(pes.NUM_PES, np.double)
		for iPES in range(pes.NUM_PES):
			ElementIndex: int = iPES * pes.NUM_PES + iPES
			pred: gp.SinglePredictor = self._predictors[ElementIndex]
			result[iPES] = pred.model.cov.lengthscale.prod().item() * pred.get_weights().sum().item() / self._predictors.scale[ElementIndex]
		return result * AnalyticalAverager.AVERAGE_CONSTANT

	def coordinates(self) -> npt.NDArray[np.double]:
		"""
		To calculate the average potential energy

		Returns
		-------
		npt.NDArray[np.double]
			Average positions and momenta
		"""
		result: npt.NDArray[np.double] = np.zeros(pes.PHASEDIM, np.double)
		for iPES in range(pes.NUM_PES):
			ElementIndex: int = iPES * pes.NUM_PES + iPES
			pred: gp.SinglePredictor = self._predictors[ElementIndex]
			result += pred.model.cov.lengthscale.prod().item() * (pred.get_weights().unsqueeze(-1) * pred.get_training_features()).sum(0).detach().numpy() / self._predictors.scale[ElementIndex]
		return result * AnalyticalAverager.AVERAGE_CONSTANT

	def potential(self) -> float:
		"""
		To calculate the average potential energy

		Returns
		-------
		float
			Average potential energy
		"""
		return np.nan

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
		result: float = 0.0
		for iPES in range(pes.NUM_PES):
			ElementIndex: int = iPES * pes.NUM_PES + iPES
			pred: gp.SinglePredictor = self._predictors[ElementIndex]
			result += pred.model.cov.lengthscale.prod().item() * ((pred.get_training_features()[:, pes.DIM] ** 2 / torch.from_numpy(mass)).sum(-1) * pred.get_weights().unsqueeze(-1)).sum().item() / self._predictors.scale[ElementIndex]
		return result * AnalyticalAverager.AVERAGE_CONSTANT

	def purity(self) -> npt.NDArray[np.double]:
		"""
		To calculate contribution to purity of each element

		Returns
		-------
		npt.NDArray[np.double]
			Purity of each element
		"""
		result: npt.NDArray[np.double] = np.empty((pes.NUM_PES, pes.NUM_PES), np.double)
		for iPES in range(pes.NUM_PES):
			for jPES in range(pes.NUM_PES):
				pred: gp.SinglePredictor = self._predictors[iPES * pes.NUM_PES + jPES]
				model: gp.GP = copy.deepcopy(pred.model)
				model.cov.lengthscale *= np.sqrt(2.0)
				result[iPES, jPES] = (np.pi ** pes.DIM) * pred.model.cov.lengthscale.prod().item() * (pred.get_weights() @ model.cov(pred.get_training_features(), pred.get_training_features()) @ pred.get_weights()).item() / (self._predictors.scale[iPES * pes.NUM_PES + jPES] ** 2)
		return pes.PURITY_FACTOR * (result + result.T - np.diag(np.diag(result)))
