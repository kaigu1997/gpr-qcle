"""
points
======
This module provides sample points evolution.
"""
import os
import sys
import typing

import numpy as np
import numpy.typing as npt
import sklearn.cluster

sys.path.append(os.path.dirname(__file__))

import evolve
import pes
import sample
import utility

class AllPoints:
	"""
	_summary_

	Parameters
	----------
	num_pts : int
		_description_
	num_extra_ratio : int
		_description_
	init_dist : pes.InitialDistribution
		_description_
	"""
	__slots__: tuple = ("__all_points", "__central_points", "__central_belonging", "__extra_ratio", "__extra_points", "__all_density", "__central_density", "__extra_density")

	def __init__(
		self,
		num_pts: int,
		num_extra_ratio: int,
		init_dist: pes.InitialDistribution
	):
		# allocate space for central and extra points
		self.__all_points: npt.NDArray[np.double] = np.empty((pes.NUM_TRIG, num_pts * (1 + num_extra_ratio), pes.PHASEDIM), np.double)
		# sample central points with kmeans
		central_pts: npt.NDArray[np.double] = sample.normal_sample(num_pts, init_dist.r0, init_dist.sigma_r0)
		kmeans: sklearn.cluster.KMeans = sklearn.cluster.KMeans(num_pts, init="k-means++", n_init="auto", random_state=np.random.RandomState(sample.np_rng.bit_generator), algorithm="lloyd").fit(np.concatenate([central_pts, sample.sample_extra_points(central_pts, num_extra_ratio)]))
		central_pts = kmeans.cluster_centers_
		print(
			"\tSample Central Points, {}, {}".format(
				utility.format_array("<r>", np.average(central_pts, 0)),
				utility.format_array("stddev", np.std(central_pts, 0))
			)
		)
		self.__central_points: npt.NDArray[np.double] = self.__all_points[:, :num_pts, :]
		self.__central_points[...] = central_pts
		# set belonging
		weights: npt.NDArray[np.double] = init_dist.weight_phase.real ** 2 + init_dist.weight_phase.imag ** 2
		self.__central_belonging: npt.NDArray[np.int_] = sample.np_rng.choice(pes.NUM_ELM, num_pts, True, weights.reshape(-1) / weights.sum()) # shape of (NUM_PTS,), normalize by purity
		row: npt.NDArray[np.int_] = self.__central_belonging // pes.NUM_PES
		col: npt.NDArray[np.int_] = self.__central_belonging % pes.NUM_PES
		self.__central_belonging = np.maximum(row, col) * pes.NUM_PES + np.minimum(row, col) # then flip
		# sample extra points and calculate density
		self.__extra_ratio: int = num_extra_ratio
		self.__extra_points: npt.NDArray[np.double] = self.__all_points[:, num_pts:, :]
		self.__all_density: npt.NDArray[np.cdouble] = np.empty((pes.NUM_TRIG, num_pts * (1 + num_extra_ratio)), np.cdouble)
		self.__central_density: npt.NDArray[np.cdouble] = self.__all_density[:, :num_pts]
		self.__extra_density: npt.NDArray[np.cdouble] = self.__all_density[:, num_pts:]
		for idx, ctr_pts, ctr_den, xtr_pts, xtr_den in zip(pes.tril_element_indices, self.__central_points, self.__central_density, self.__extra_points, self.__extra_density):
			xtr_pts[...] = sample.sample_extra_points(ctr_pts, self.__extra_ratio)
			ctr_den[...] = init_dist(ctr_pts, idx)
			xtr_den[...] = init_dist(xtr_pts, idx)

	@property
	def central_points(self) -> npt.NDArray[np.double]:
		return self.__central_points

	@property
	def central_density(self) -> npt.NDArray[np.cdouble]:
		return self.__central_density

	@property
	def all_points(self) -> npt.NDArray[np.double]:
		return self.__all_points

	@property
	def all_density(self) -> npt.NDArray[np.cdouble]:
		return self.__all_density

	def get_rescale_factor(self) -> npt.NDArray[np.double]:
		"""
		To calculate the scale factor from the given data.

		The scale would be 1 / max() in general, or 0 if all are 0

		Returns
		-------
		npt.NDArray[np.double], of shape (NUM_ELM,)
			Scale of each element
		"""
		def get_element_scale(density: npt.NDArray[np.double]) -> float:
			"""
			To calculate the scale of an element

			Parameters
			----------
			density : npt.NDArray[np.double], shape of (NUM_PTS * (1 + NUM_XTR_RATIO),)
				The density of central and extra points

			Returns
			-------
			float
				Scale of the element. 0 if density of all points is 0,
				otherwise the reciprocal of max density
			"""
			if np.all(density == 0.0):
				return 0.0
			else:
				return 1.0 / np.max(np.abs(density))

		result: npt.NDArray[np.double] = np.empty((pes.NUM_PES, pes.NUM_PES), np.double)
		for iPES in range(pes.NUM_PES):
			for jPES in range(iPES + 1):
				TrilIndex: int = pes.flatten_tril_index[iPES, jPES]
				if iPES == jPES:
					result[iPES, jPES] = get_element_scale(self.__all_density[TrilIndex].real)
				else:
					result[iPES, jPES] = get_element_scale(self.__all_density[TrilIndex].imag)
					result[jPES, iPES] = get_element_scale(self.__all_density[TrilIndex].real)
		return result.reshape(-1)
	
	def evolve(
		self,
		mass: npt.NDArray[np.double],
		dt: float,
		predictor: typing.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]
	) -> None:
		"""
		To evolve the central and extra points for a single step

		Parameters
		----------
		mass : npt.NDArray[np.double], shape of (DIM,)
			Mass of classical degree of freedom
		dt : float
			Time interval
		predictor : typing.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]
			It predicts the density matrix element based on given coordinates and element index
		"""
		# evolve central points with surface hopping
		evolve.sh_evolve(self.__central_points, self.__central_density, self.__central_belonging, mass, dt, predictor)
		# lower triangular loop, evolve extra points coordinates and density
		evolve.evolve(self.__extra_points, self.__extra_density, mass, dt, predictor)

	def resample_extra(self, predictor: typing.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]) -> None:
		"""
		To resample extra points

		Parameters
		----------
		predictor : typing.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]
			To predict the density of the newly sampled extra points
		"""
		for ctr_pts, xtr_pts, xtr_den, idx in zip(self.__central_points, self.__extra_points, self.__extra_density, pes.tril_element_indices):
			xtr_pts[...] = sample.sample_extra_points(ctr_pts, self.__extra_ratio)
			xtr_den[...] = predictor(xtr_pts, idx)
