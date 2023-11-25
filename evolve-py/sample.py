"""
sample
======

This module provides methods for sampling.
"""
import typing

import numpy as np
import numpy.typing as npt
import sklearn.cluster
import torch

import pes

VARIANCE_RATIO: float = 5.0
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


def sample_central_points(num_points: int, all_points: npt.NDArray[np.double]) -> None:
	"""
	To resample the central points

	Parameters
	----------
	num_points : int
		The number of central points. They will be placed at first
	all_points : npt.NDArray[np.double], shape of (NUM_ELM, NUM_PTS * (1 + NUM_XTR_RATIO), PHASEDIM)
		All the points of all elements and dimensions
	"""
	for iPES in range(pes.NUM_PES):
		for jPES in range(iPES + 1):
			ElementIndex: int = iPES * pes.NUM_PES + jPES
			kmeans: sklearn.cluster.KMeans = sklearn.cluster.KMeans(num_points, init="k-means++", n_init="auto", random_state=np.random.RandomState(np_rng.bit_generator), algorithm="lloyd").fit(all_points[ElementIndex])
			all_points[ElementIndex, :num_points] = kmeans.cluster_centers_
			print("<r> = {}, stddev = {}\n".format(np.average(all_points[ElementIndex, :num_points], 0), np.std(all_points[ElementIndex, :num_points], 0)), flush=True)
			if iPES != jPES:
				SymmetricElementIndex: int = jPES * pes.NUM_PES + iPES
				all_points[SymmetricElementIndex] = all_points[ElementIndex]


def sample_extra_points(num_points: int, all_points: npt.NDArray[np.double]) -> None:
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
			var: npt.NDArray[np.double] = np.diag(np.var(all_points[ElementIndex, :num_points], 0))
			all_points[ElementIndex, num_points:] = np.concatenate([np_rng.multivariate_normal(pt, var, extra_point_ratio, "raise", method="cholesky") for pt in all_points[ElementIndex, :num_points]])
			if iPES != jPES:
				all_points[jPES * pes.NUM_PES + iPES, num_points:] = all_points[ElementIndex, num_points:]
