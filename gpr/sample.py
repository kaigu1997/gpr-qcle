"""
sample
======
This module provides methods for sampling.
"""
import os
import sys

import numpy as np
import numpy.typing as npt
import torch

sys.path.append(os.path.dirname(__file__))

import pes
import utility

np_rng: np.random.Generator = np.random.Generator(np.random.MT19937(0))
torch.manual_seed(0)


def normal_sample(
	num_points: int,
	mean: npt.NDArray[np.double],
	stddev: npt.NDArray[np.double]
) -> npt.NDArray[np.double]:
	"""
	To create normally distributed point set based on given mean and variance

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
	npt.NDArray[np.double], shape of (NUM_PTS, PHASEDIM)
		Normally distributed point test
	"""
	return torch.randn((num_points, pes.PHASEDIM), dtype=torch.float).detach().numpy() * stddev + mean


def sample_extra_points(
	central_points: npt.NDArray[np.double],
	extra_point_ratio: int
) -> npt.NDArray[np.double]:
	"""
	To create the extra point set

	Parameters
	----------
	central_points : npt.NDArray[np.double], shape of (NUM_PTS, PHASEDIM)
		The central points. They are the centers for the extra points
	extra_point_ratio : int
		The number of extra points around each central point

	Returns
	-------
	npt.NDArray[np.double], shape of (NUM_PTS * NUM_XTR_RATIO, PHASEDIM)
		The extra points. Same for all elements.
		If different points for each elements are needed, `np.tile` could be used
	"""
	var: npt.NDArray[np.double] = np.diag(np.var(central_points, 0))
	result: npt.NDArray[np.double] = np.concatenate([np_rng.multivariate_normal(pt, var, extra_point_ratio, "raise", method="cholesky") for pt in central_points])
	print(
		"\tSample Extra Points, {}, {}".format(
			utility.format_array("<r>", np.average(result, 0)),
			utility.format_array("stddev", np.std(result, 0))
		),
		flush=True
	)
	return result
