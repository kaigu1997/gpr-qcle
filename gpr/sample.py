"""
sample
======

This module provides methods for sampling.
"""
import os
import sys

import numpy as np
import numpy.typing as npt
import sklearn.cluster
import torch

sys.path.append(os.path.dirname(__file__))

import pes
import utility

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


def sample_central_points(num_points: int | npt.NDArray[np.int_], all_points: list[npt.NDArray[np.double]]) -> None:
	"""
	To resample the central points

	Parameters
	----------
	num_points : int | npt.NDArray[np.int_]
		The number of central points. They will be placed at first
	all_points : list[npt.NDArray[np.double]], len of NUM_TRIG, each of shape (num_point * (1 + NUM_XTR_RATIO), PHASEDIM)
		All the points of all elements and dimensions
	"""
	if isinstance(num_points, int):
		num_points = np.full(pes.NUM_TRIG, num_points, np.int_)
	for iPES in range(pes.NUM_PES):
		for jPES in range(iPES + 1):
			TrilIndex: int = pes.flatten_tril_index[iPES, jPES]
			kmeans: sklearn.cluster.KMeans = sklearn.cluster.KMeans(num_points[TrilIndex], init="k-means++", n_init="auto", random_state=np.random.RandomState(np_rng.bit_generator), algorithm="lloyd").fit(all_points[TrilIndex])
			all_points[TrilIndex][:num_points[TrilIndex]] = kmeans.cluster_centers_
			print(
				"rho({}, {}), {}, {}".format(
					iPES,
					jPES,
					utility.format_array("<r>", np.average(all_points[TrilIndex][:num_points[TrilIndex]], 0)),
					utility.format_array("stddev", np.std(all_points[TrilIndex][:num_points[TrilIndex]], 0))
				),
				flush=True
			)


def sample_extra_points(num_points: int | npt.NDArray[np.int_], all_points: list[npt.NDArray[np.double]]) -> None:
	"""
	To create the extra point set

	First num_points points remain the same, and the rest of the points are resampled based on the mean of the first num_points points and variance from predictors

	Parameters
	----------
	num_points : int | npt.NDArray[np.int_]
		The number of central points. The rest of the points are resampled
	all_points : list[npt.NDArray[np.double]], len of NUM_TRIG, each of shape (num_point * (1 + NUM_XTR_RATIO), PHASEDIM)
		Points of all elements and dimensions
	"""
	if isinstance(num_points, int):
		num_points = np.full(pes.NUM_TRIG, num_points, np.int_)
	for iPES in range(pes.NUM_PES):
		for jPES in range(iPES + 1):
			TrilIndex: int = pes.flatten_tril_index[iPES, jPES]
			assert all_points[TrilIndex].shape[-2] % num_points[TrilIndex] == 0
			extra_point_ratio: int = all_points[TrilIndex].shape[-2] // num_points[TrilIndex] - 1
			var: npt.NDArray[np.double] = np.var(all_points[TrilIndex][:num_points[TrilIndex]], 0)
			var = np.diag(var)
			all_points[TrilIndex][num_points[TrilIndex]:] = np.concatenate([np_rng.multivariate_normal(pt, var, extra_point_ratio, "raise", method="cholesky") for pt in all_points[TrilIndex][:num_points[TrilIndex]]])
