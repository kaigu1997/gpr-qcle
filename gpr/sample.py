"""
sample
======

This module provides methods for sampling.
"""

import numpy as np
import numpy.typing as npt

import gp
import pes

VARIANCE_RATIO: float = 5.0
np_rng: np.random.Generator = np.random.Generator(np.random.MT19937(0))


def init_sample(num_points: int, mean: npt.NDArray[np.double], stddev: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
	"""
	To create the initial point set

	The points will be normally distributed based on given mean and variance

	Parameters
	----------
	num_points : int
		The number of points needed
	mean : npt.NDArray[np.double], shape of (PHASEDIM,)
		The center of the points
	stddev : npt.NDArray[np.double], shape of (PHASEDIM,)
		The standard deviation of the points

	Returns
	-------
	npt.NDArray[np.double], shape of (NUM_ELM, NUM_PTS, PHASEDIM)
		Initial normally distributed point test
	"""
	center_pts: npt.NDArray[np.double] = np_rng.multivariate_normal(mean, np.diag(stddev ** 2), num_points, 'raise', method='cholesky')
	return np.repeat(center_pts[np.newaxis, ...], pes.NUM_ELM, axis=0)


def sample_central_points(
	all_points: npt.NDArray[np.double],
	all_density: npt.NDArray[np.cdouble],
	num_points: int,
	predictors: gp.GPRPredictors
) -> None:
	"""
	To resample the central points

	Parameters
	----------
	all_points : npt.NDArray[np.double], shape of (NUM_ELM, NUM_PTS * (1 + NUM_XTR_RATIO), PHASEDIM)
		All the points of all elements and dimensions
	all_density : npt.NDArray[np.cdouble], shape of (NUM_ELM, NUM_PTS * (1 + NUM_XTR_RATIO))
		The corresponding density of the points
	num_points : int
		The number of central points. They will be placed at first
	predictors : gp.GPRPredictors
		Predictor, which provides the variance
	"""
	def kernel_matrix_to_linearity(kernel_matrix: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
		"""
		Based on the given kernel matrix (symmetric, positive definite), calculate the linearity between points

		Parameters
		----------
		kernel_matrix : npt.NDArray[np.double]
			The kernel matrix, which is symmetric and positive definite

		Returns
		-------
		npt.NDArray[np.double]
			Linearity (or dependency) between points. Ranging [0, 1],
			from totally linear (=0) to totally dependent (=1)
		"""
		# kernel_matrix_to_linearity.power_constant = 0.5
		kernel_matrix /= np.linalg.norm(kernel_matrix, None, 1, True) # normalize each row
		# return np.abs(1.0 - np.abs(kernel_matrix @ kernel_matrix.T)) ** kernel_matrix_to_linearity.power_constant # absolute value of inner products between indices
		return np.cos(np.pi / 2.0 * kernel_matrix @ kernel_matrix.T)

	def density_to_probability(absolute_density: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
		"""
		To turn density (>=0) to probability (normalized, sum to 1)

		Parameters
		----------
		absolute_density : npt.NDArray[np.double]
			Density, all elements should be non-negative

		Returns
		-------
		npt.NDArray[np.double]
			Probability, which sums up to 1.0
		"""
		assert np.all(absolute_density >= 0.0)
		result: npt.NDArray[np.double] = absolute_density
		return result / np.sum(result)

	for iPES in range(pes.NUM_PES):
		for jPES in range(iPES + 1):
			ElementIndex: int = iPES * pes.NUM_PES + jPES
			den_abs: npt.NDArray[np.double] = np.abs(all_density[ElementIndex])
			if np.any(den_abs != 0.0):
				print(np.average(all_points[ElementIndex, :num_points], 0), np.std(all_points[ElementIndex, :num_points], 0))
				print(predictors[ElementIndex].variance(), predictors[jPES * pes.NUM_PES + iPES].variance())
				max_indices: npt.NDArray[np.int_]
				if np.all(VARIANCE_RATIO * predictors[ElementIndex].variance() < np.std(all_points[ElementIndex, :num_points], 0)) and (iPES == jPES or np.all(VARIANCE_RATIO * predictors[jPES * pes.NUM_PES + iPES].variance() < np.std(all_points[ElementIndex, :num_points], 0))):
					print("maximum")
					max_indices = np_rng.choice(all_points.shape[-2], num_points, False, density_to_probability(den_abs))
				else:
					print("linearity")
					linearity: npt.NDArray[np.double]
					if iPES == jPES:
						linearity = kernel_matrix_to_linearity(gp.construct_kernel_matrix(predictors[ElementIndex], all_points[ElementIndex]))
					else:
						linearity = np.minimum(kernel_matrix_to_linearity(gp.construct_kernel_matrix(predictors[ElementIndex], all_points[ElementIndex])), kernel_matrix_to_linearity(gp.construct_kernel_matrix(predictors[jPES * pes.NUM_PES + iPES], all_points[jPES * pes.NUM_PES + iPES])))
					# first, find the maximum
					max_indices: npt.NDArray[np.int_] = np.empty(num_points, np.int_)
					max_indices[0] = np.argmax(den_abs)
					for i in range(1, num_points):
						max_indices[i] = np.argmax(np.min(linearity[:, max_indices[:i]], axis=-1) * den_abs)
				source_set: set[int] = set(max_indices) # the selected points
				print(len(source_set - set(np.argpartition(den_abs, -num_points)[-num_points:])))
				places_set: set[int] = set(range(num_points)) # the first N indices
				source_set, places_set = source_set - places_set, places_set - source_set # remove duplicate
				assert len(source_set) == len(places_set)
				# then replace the selected points with the initial points
				source_list: list[int] = list(source_set)
				places_list: list[int] = list(places_set)
				source_list, places_list = source_list + places_list, places_list + source_list
				all_points[ElementIndex, places_list] = all_points[ElementIndex, source_list]
				all_density[ElementIndex, places_list] = all_density[ElementIndex, source_list]
				print(np.average(all_points[ElementIndex, :num_points], 0), np.std(all_points[ElementIndex, :num_points], 0), '\n')
				if iPES != jPES:
					all_points[jPES * pes.NUM_PES + iPES] = all_points[ElementIndex]
					all_density[jPES * pes.NUM_PES + iPES] = np.conj(all_density[ElementIndex])


def sample_extra_points(all_points: npt.NDArray[np.double], num_points: int, predictors: gp.GPRPredictors | None = None) -> None:
	"""
	To create the extra point set

	First num_points points remain the same, and the rest of the points are resampled based on the mean of the first num_points points and variance from predictors

	Parameters
	----------
	all_points : npt.NDArray[np.double], shape of (NUM_ELM, NUM_PTS * (1 + NUM_XTR_RATIO), PHASEDIM)
		Points of all elements and dimensions
	num_points : int
		The number of central points. The rest of the points are resampled
	predictors : gp.GPRPredictors | None, optional
		Predictor, which provides the variance, by default None
	"""
	assert all_points.shape[-2] % num_points == 0
	extra_point_ratio: int = all_points.shape[-2] // num_points - 1
	for iPES in range(pes.NUM_PES):
		for jPES in range(iPES + 1):
			ElementIndex: int = iPES * pes.NUM_PES + jPES
			var: npt.NDArray[np.double]
			if predictors is not None:
				var = VARIANCE_RATIO * np.maximum(predictors[ElementIndex].variance(), VARIANCE_RATIO * predictors[jPES * pes.NUM_PES + iPES].variance())
			else:
				var = np.var(all_points[ElementIndex, :num_points], 0)
			var = np.diag(var)
			all_points[ElementIndex, num_points:] = np.concatenate([np_rng.multivariate_normal(pt, var, extra_point_ratio, 'raise', method='cholesky') for pt in all_points[ElementIndex, :num_points]])
			if iPES != jPES:
				all_points[jPES * pes.NUM_PES + iPES, num_points:] = all_points[ElementIndex, num_points:]
