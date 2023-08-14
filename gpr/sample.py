import gp
import pes

import numpy as np
import numpy.typing as npt

VARIANCE_RATIO: float = 5.0
np_rng: np.random.Generator = np.random.Generator(np.random.MT19937(0))


def init_sample(num_points: int, mean: npt.NDArray[np.double], stddev: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
	"""
	_summary_

	Parameters
	----------
	num_points : int
		_description_
	mean : npt.NDArray[np.double], shape of (PHASEDIM,)
		_description_
	stddev : npt.NDArray[np.double], shape of (PHASEDIM,)
		_description_

	Returns
	-------
	npt.NDArray[np.double], shape of (NUM_ELM, NUM_PTS, PHASEDIM)
		_description_
	"""
	center_pts: npt.NDArray[np.double] = np_rng.multivariate_normal(mean, np.diag(stddev ** 2), num_points, 'raise', method='cholesky')
	return np.repeat(center_pts[np.newaxis, ...], pes.NUM_ELM, axis=0)


def sample_extra_points(all_points: npt.NDArray[np.double], num_points: int, predictors: gp.GPRPredictors | None = None) -> None:
	"""
	_summary_

	Parameters
	----------
	all_points : npt.NDArray[np.double], shape of (NUM_ELM, NUM_PTS * (1 + NUM_XTR_RATIO), PHASEDIM)
		_description_
	num_points : int
		_description_
	predictors : gp.GPRPredictors
		_description_
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


def resample(all_points: npt.NDArray[np.double], all_density: npt.NDArray[np.cdouble], num_points: int, predictors: gp.GPRPredictors) -> None:
	"""
	_summary_

	Parameters
	----------
	all_points : npt.NDArray[np.double], shape of (NUM_ELM, NUM_PTS * (1 + NUM_XTR_RATIO), PHASEDIM)
		_description_
	all_density : npt.NDArray[np.cdouble], shape of (NUM_ELM, NUM_PTS * (1 + NUM_XTR_RATIO))
		_description_
	num_points : int
		_description_
	predictors : gp.GPRPredictors
		_description_
	"""
	def kernel_matrix_to_linearity(kernel_matrix: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
		# kernel_matrix_to_linearity.power_constant = 0.5
		kernel_matrix /= np.linalg.norm(kernel_matrix, None, 1, True) # normalize each row
		# return np.abs(1.0 - np.abs(kernel_matrix @ kernel_matrix.T)) ** kernel_matrix_to_linearity.power_constant # absolute value of inner products between indices
		return np.cos(np.pi / 2.0 * kernel_matrix @ kernel_matrix.T)

	def density_to_probability(absolute_density: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
		result: npt.NDArray[np.double] = absolute_density
		return result / np.sum(result)

	resample.STEPS = 1000
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
