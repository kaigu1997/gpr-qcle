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

import gp
import pes

VARIANCE_RATIO: float = 5.0
np_rng: np.random.Generator = np.random.Generator(np.random.MT19937(0))
torch.manual_seed(0)


def pseudo_potential(
	X: torch.Tensor,
	dist_func: typing.Callable[[torch.Tensor], torch.Tensor] = lambda x: torch.exp(-(x ** 2).sum(-1) / 2)
) -> torch.Tensor:
	"""
	_summary_

	Parameters
	----------
	X : torch.Tensor
		_description_
	dist : typing.Callable[[torch.Tensor], torch.Tensor]
		_description_

	Returns
	-------
	torch.Tensor
		_description_
	"""
	pseudo_potential.m = pes.PHASEDIM
	sigma: torch.Tensor = torch.pow(dist_func(X), -pseudo_potential.m / pes.PHASEDIM) # shape of (N,)
	pdist_sq: torch.Tensor = ((X.unsqueeze(1) - X) ** 2).sum(-1).fill_diagonal_(1.0) # shape of (N, N)
	result: torch.Tensor = (sigma + sigma.unsqueeze(1) ** 2) / torch.pow(pdist_sq, pseudo_potential.m / 2)
	return torch.triu(result, 1).sum()


def simulate_tempering(
	x: torch.Tensor,
	potential: typing.Callable[[torch.Tensor], torch.Tensor] = pseudo_potential,
	high_temp: float = 300.0,
	low_temp: float = 10.0,
	alpha: float = 0.95,
) -> torch.Tensor:
	"""
	_summary_

	Parameters
	----------
	x : torch.Tensor
		_description_
	potential : _type_, optional
		_description_, by default pseudo_potential
	high_temp : float, optional
		_description_, by default 300.0
	low_temp : float, optional
		_description_, by default 10.0
	alpha : float, optional
		_description_, by default 0.99
	iter_per_temp : int, optional
		_description_, by default 100
	"""
	Vx: torch.Tensor = potential(x)
	current_temp: float = high_temp
	disp: torch.Tensor = torch.Tensor([torch.min(torch.pdist(x[:, i:i+1], torch.inf)) for i in range(x.shape[-1])])
	iter_per_temp: int = int(torch.max(1.0 / disp).item())
	while current_temp > low_temp:
		print("T = {}, V = {}".format(current_temp, Vx.item()))
		for _ in range(iter_per_temp):
			x_new: torch.Tensor = x + torch.randn(x.shape) * disp
			Vx_new: torch.Tensor = potential(x_new)
			acc: torch.Tensor = torch.exp((Vx - Vx_new) / current_temp) > torch.rand(1)
			if acc.item():
				x = x_new
				Vx = Vx_new
		current_temp *= alpha
	return x


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
	stddev : npt.NDArray[np.double], shape of (PHASEpes.PHASEDIM,)
		The standard deviation of the points

	Returns
	-------
	npt.NDArray[np.double], shape of (NUM_ELM, NUM_PTS, PHASEDIM)
		Normally distributed point test
	"""
	return torch.randn((num_points, pes.PHASEDIM), dtype=torch.float).detach().numpy() * stddev + mean


def sample_central_points(
	num_points: int,
	all_points: npt.NDArray[np.double],
	all_density: npt.NDArray[np.cdouble] | None = None,
	predictors: gp.GPRPredictors | None = None
) -> None:
	"""
	To resample the central points

	Parameters
	----------
	num_points : int
		The number of central points. They will be placed at first
	all_points : npt.NDArray[np.double], shape of (NUM_ELM, NUM_PTS * (1 + NUM_XTR_RATIO), PHASEDIM)
		All the points of all elements and dimensions
	all_density : npt.NDArray[np.cdouble], shape of (NUM_ELM, NUM_PTS * (1 + NUM_XTR_RATIO))
		The corresponding density of the points
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

	def dist_to_one(val: float) -> float:
		return val if abs(val) > 1.0 else 1.0 / val

	for iPES in range(pes.NUM_PES):
		for jPES in range(iPES + 1):
			ElementIndex: int = iPES * pes.NUM_PES + jPES
			if all_density is None:
				print("k-means")
				kmeans: sklearn.cluster.KMeans = sklearn.cluster.KMeans(num_points, init="k-means++", n_init="auto", random_state=np.random.RandomState(np_rng.bit_generator), algorithm="lloyd").fit(all_points[ElementIndex])
				all_points[ElementIndex, :num_points] = kmeans.cluster_centers_
			else:
				den_abs: npt.NDArray[np.double] = np.abs(all_density[ElementIndex])
				print("<r> = {}, stddev = {}".format(np.average(all_points[ElementIndex, :num_points], 0), np.std(all_points[ElementIndex, :num_points], 0)))
				max_indices: npt.NDArray[np.int_]
				if predictors is not None:
					print("Predictor variance = {} and {}".format(predictors[ElementIndex].variance(), predictors[jPES * pes.NUM_PES + iPES].variance()))
					# if np.all(VARIANCE_RATIO * predictors[ElementIndex].variance() < np.std(all_points[ElementIndex, :num_points], 0)) and (iPES == jPES or np.all(VARIANCE_RATIO * predictors[jPES * pes.NUM_PES + iPES].variance() < np.std(all_points[ElementIndex, :num_points], 0))):
					# 	print("maximum")
					# 	max_indices = np_rng.choice(all_points.shape[-2], num_points, False, density_to_probability(den_abs))
					# else:
					print("linearity")
					linearity: npt.NDArray[np.double]
					if iPES == jPES:
						linearity = kernel_matrix_to_linearity(gp.construct_kernel_matrix(predictors[ElementIndex], all_points[ElementIndex]))
					else:
						linearity = np.minimum(kernel_matrix_to_linearity(gp.construct_kernel_matrix(predictors[ElementIndex], all_points[ElementIndex])), kernel_matrix_to_linearity(gp.construct_kernel_matrix(predictors[jPES * pes.NUM_PES + iPES], all_points[jPES * pes.NUM_PES + iPES])))
					# first, find the maximum
					max_indices = np.empty(num_points, np.int_)
					max_indices[0] = np.argmax(den_abs)
					for i in range(1, num_points):
						max_indices[i] = np.argmax(np.min(linearity[:, max_indices[:i]], axis=-1) * den_abs)
				else:
					print("weighted k-means")
					old_power: float = 1.0
					power: float = 1.0
					stddev: npt.NDArray[np.double] = np.std(all_points[ElementIndex, :num_points], 0)
					kmeans: sklearn.cluster.KMeans = sklearn.cluster.KMeans(num_points, init="k-means++", n_init="auto", random_state=np.random.RandomState(np_rng.bit_generator), algorithm="lloyd").fit(all_points[ElementIndex], sample_weight=np.power(np.abs(all_density[ElementIndex]), power))
					old_close_to_one: float = np.inf
					close_to_one: float = (np.std(kmeans.cluster_centers_, 0) / stddev).prod()
					print('\t', power, close_to_one, np.std(kmeans.cluster_centers_, 0), stddev)
					while dist_to_one(close_to_one) < dist_to_one(old_close_to_one) and close_to_one < 1.0:
						old_close_to_one, old_power = close_to_one, power
						power *= close_to_one ** (1.0 / pes.PHASEDIM)
						kmeans = kmeans.fit(all_points[ElementIndex], sample_weight=np.power(np.abs(all_density[ElementIndex]), power))
						close_to_one = (np.std(kmeans.cluster_centers_, 0) / stddev).prod()
						print('\t', power, close_to_one, np.std(kmeans.cluster_centers_, 0), stddev)
					power = old_power
					kmeans = kmeans.fit(all_points[ElementIndex], sample_weight=np.power(np.abs(all_density[ElementIndex]), power))
					if np.all(np.std(kmeans.cluster_centers_, 0) < stddev):
						kmeans = kmeans.fit(all_points[ElementIndex], sample_weight=np.power(np.abs(all_density[ElementIndex]), power * np.sqrt(2.0))) # the power is too small, need adjusted. Otherwise
					# find the closest
					sq_dist: npt.NDArray[np.double] = ((kmeans.cluster_centers_[:, np.newaxis] - all_points[ElementIndex]) ** 2).sum(axis=-1)
					max_indices = np.argmin(sq_dist, -1)
					indices: npt.NDArray[np.int_]
					_, indices = np.unique(max_indices, True)
					while indices.size < num_points:
						print("{} selected, {} left".format(indices.size, num_points - indices.size))
						sq_dist[list(range(num_points)), max_indices] = np.inf
						diff: list[int] = list(set(range(num_points)) - set(indices))
						max_indices[diff] = np.argmin(sq_dist[diff], -1)
						_, indices = np.unique(max_indices, True)
				source_set: set[int] = set(max_indices) # the selected points
				print("{} points changed".format(len(source_set - set(np.argpartition(den_abs, -num_points)[-num_points:]))))
				places_set: set[int] = set(range(num_points)) # the first N indices
				source_set, places_set = source_set - places_set, places_set - source_set # remove duplicate
				assert len(source_set) == len(places_set)
				# then replace the selected points with the initial points
				source_list: list[int] = list(source_set)
				places_list: list[int] = list(places_set)
				source_list, places_list = source_list + places_list, places_list + source_list
				all_points[ElementIndex, places_list] = all_points[ElementIndex, source_list]
				all_density[ElementIndex, places_list] = all_density[ElementIndex, source_list]
			print("<r> = {}, stddev = {}\n".format(np.average(all_points[ElementIndex, :num_points], 0), np.std(all_points[ElementIndex, :num_points], 0)), flush=True)
			if iPES != jPES:
				SymmetricElementIndex: int = jPES * pes.NUM_PES + iPES
				all_points[SymmetricElementIndex] = all_points[ElementIndex]
				if all_density is not None:
					all_density[SymmetricElementIndex] = np.conj(all_density[ElementIndex])


def sample_extra_points(num_points: int, all_points: npt.NDArray[np.double], predictors: gp.GPRPredictors | None = None) -> None:
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
				var = VARIANCE_RATIO * np.maximum(predictors[ElementIndex].variance(), predictors[jPES * pes.NUM_PES + iPES].variance())
			else:
				var = np.var(all_points[ElementIndex, :num_points], 0)
			var = np.diag(var)
			all_points[ElementIndex, num_points:] = np.concatenate([np_rng.multivariate_normal(pt, var, extra_point_ratio, "raise", method="cholesky") for pt in all_points[ElementIndex, :num_points]])
			if iPES != jPES:
				all_points[jPES * pes.NUM_PES + iPES, num_points:] = all_points[ElementIndex, num_points:]
