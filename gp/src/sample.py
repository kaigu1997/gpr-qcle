import evolve

import numpy as np
import numpy.typing as npt

np_rng: np.random.Generator = np.random.Generator(np.random.MT19937(0))


def init_sample(num_points: int, mean: npt.NDArray[np.double], stddev: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
	center_pts: npt.NDArray[np.double] = np_rng.multivariate_normal(mean, np.diag(stddev), num_points, 'raise', method='cholesky')
	return np.repeat(center_pts[np.newaxis, ...], evolve.NUM_ELM, axis=0)


def sample_extra_points(sample_points: npt.NDArray[np.double], extra_point_ratio: int) -> npt.NDArray[np.double]:
	vars: npt.NDArray[np.double] = np.concatenate([np.diag(var)[np.newaxis, :, :] for var in np.var(sample_points, 1)])
	return np.array([np.concatenate([pts] + [np_rng.multivariate_normal(pt, var, extra_point_ratio, 'raise', method='cholesky') for pt in pts]) for pts, var in zip(sample_points, vars)])
