import collections.abc
import typing

import numpy as np
import numpy.typing as npt
import scipy.spatial
import sklearn.cluster
import torch

torch.set_default_dtype(torch.float64)


def format_array(arr_name : str | None, arr: typing.Any, sep: str = " ") -> str:
	result: str = (arr_name + " = ") if arr_name else ""
	if isinstance(arr, torch.Tensor) or isinstance(arr, np.ndarray):
		result += sep.join(format_array(None, val.item()) for val in arr.ravel())
	elif isinstance(arr, collections.abc.Iterable):
		result += sep.join(format_array(None, item) for item in arr)
	elif isinstance(arr, complex):
		result += f"{arr.real} + {arr.imag}i"
	else:
		result += str(arr)
	return result


class distribution:
	N_GAUSS = 32
	DIM = 2
	MAX: float = 20.0
	MIN: float = -MAX
	N_GRIDS = 201
	x1: npt.NDArray[np.double]
	x2: npt.NDArray[np.double]
	x1, x2 = np.meshgrid(np.linspace(MIN, MAX, N_GRIDS, dtype=np.double), np.linspace(MIN, MAX, N_GRIDS, dtype=np.double))
	x_test: npt.NDArray[np.double] = np.concatenate((x1[..., np.newaxis], x2[..., np.newaxis]), axis=-1).reshape(-1, DIM)

	def __init__(self, seed: int | None) -> None:
		self.rng: np.random.Generator = np.random.Generator(np.random.MT19937(seed))
		self.is_trivial: bool = seed is None
		if self.is_trivial:
			self.WIDTH: npt.NDArray[np.double] = np.array([1.0, 1.0])
			print("Widths are\n", self.WIDTH, sep=None)
		else:
			self.CENTER: npt.NDArray[np.double] = self.rng.uniform(-__class__.MAX / 10, __class__.MAX / 10, (__class__.N_GAUSS, __class__.DIM))
			self.WIDTH: npt.NDArray[np.double] = self.rng.uniform(0.1, __class__.MAX / 10, (__class__.N_GAUSS, __class__.DIM))
			self.WEIGHT: npt.NDArray[np.double] = self.rng.uniform(-1.0, 1.0, __class__.N_GAUSS)
			print("Centers are at\n", self.CENTER, sep=None)
			print("Weights of Gaussian are\n", self.WEIGHT, sep=None, flush=True)
		self.y_test = self.dist_func(__class__.x_test).reshape(__class__.N_GRIDS, __class__.N_GRIDS)

	def dist_func(self, x: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
		if x.ndim == 1:
			x = x[np.newaxis]
		if self.is_trivial:
			return np.exp(-np.sum((x / self.WIDTH) ** 2, -1) / 2.0)
		else:
			return np.sum(self.WEIGHT * np.exp(-np.sum(np.square((x[:, np.newaxis, :] - self.CENTER) / self.WIDTH), axis=-1) / 2.0), axis=-1)


class data:
	@staticmethod
	def __random_walk(dist: distribution, draw_sample_pts: collections.abc.Callable[[npt.NDArray[np.double], npt.NDArray[np.double], int], None]) -> tuple[npt.NDArray[np.double], npt.NDArray[np.double]]:
		def print_coordinate_distribution(title: str, coordinates: npt.NDArray[np.double]) -> None:
			print(f"{(title + ": ") if title != "" else ""}{format_array("<r>", np.average(coordinates, 0))}, {format_array("stddev", np.std(coordinates, 0))}")

		central_pts: npt.NDArray[np.double]
		if dist.is_trivial:
			N_PTS = 512
			central_pts = dist.rng.multivariate_normal(np.zeros(distribution.DIM), np.diag(dist.WIDTH ** 2), N_PTS, "raise", method="cholesky")
		else:
			PTS_RATIO = 16
			STEP_SIZE: float = np.sqrt(np.max(scipy.spatial.distance.pdist(dist.CENTER, 'sqeuclidean')))
			NUM_STEP = 1000
			BOX_SIZE: float = distribution.MAX - distribution.MIN
			print(f"Step size for random walk is {STEP_SIZE}")
			print(f"Number of steps for random walk is {NUM_STEP}")
			central_pts = np.repeat(dist.CENTER, PTS_RATIO, 0)
			weights: npt.NDArray[np.double] = dist.dist_func(central_pts)
			acc_ratio: npt.NDArray[np.double] = np.zeros(weights.shape)
			for _ in range(NUM_STEP):
				new_pts: npt.NDArray[np.double] = central_pts + dist.rng.uniform(-STEP_SIZE, STEP_SIZE, central_pts.shape)
				new_pts = np.where(new_pts < distribution.MIN, new_pts + BOX_SIZE, new_pts)
				new_pts = np.where(new_pts > distribution.MAX, new_pts - BOX_SIZE, new_pts)
				new_weights: npt.NDArray[np.double] = dist.dist_func(new_pts)
				acc: npt.NDArray[np.bool_] = np.power(np.abs(new_weights / weights), 2.0) > dist.rng.uniform(0, 1, weights.shape)
				acc_ratio += acc
				central_pts = np.where(acc[:, np.newaxis], new_pts, central_pts)
				weights = np.where(acc, new_weights, weights)
			acc_ratio /= NUM_STEP
			print(f"Acceptance ratio for {central_pts.size // distribution.DIM} central points is within [{np.min(acc_ratio)}, {np.max(acc_ratio)}]", flush=True)
		print_coordinate_distribution("Sample Central Points", central_pts)
		EXTRA_RATIO = 25
		var: npt.NDArray[np.double] = np.diag(np.var(central_pts, 0))
		extra_pts: npt.NDArray[np.double] = np.concatenate([dist.rng.multivariate_normal(pt, var, EXTRA_RATIO, "raise", method="cholesky") for pt in central_pts])
		print_coordinate_distribution("Sample Extra Points", extra_pts)
		kmeans_center: sklearn.cluster.KMeans = sklearn.cluster.KMeans(central_pts.shape[0], init="k-means++", n_init="auto", random_state=np.random.RandomState(dist.rng.bit_generator), algorithm="lloyd")
		kmeans_extra: sklearn.cluster.KMeans = sklearn.cluster.KMeans(central_pts.shape[0] * EXTRA_RATIO, init="k-means++", n_init="auto", random_state=np.random.RandomState(dist.rng.bit_generator), algorithm="lloyd")
		num_plot: int = 0
		draw_sample_pts(central_pts, extra_pts, num_plot)
		num_plot += 1
		central_pts[...] = kmeans_center.fit(np.concatenate([central_pts, extra_pts])).cluster_centers_
		print_coordinate_distribution("KMeans Central Points", central_pts)
		var = np.diag(np.var(kmeans_center.cluster_centers_, 0))
		extra_pts[...] = kmeans_extra.fit(np.concatenate([dist.rng.multivariate_normal(pt, var, EXTRA_RATIO ** 2, "raise", method="cholesky") for pt in kmeans_center.cluster_centers_])).cluster_centers_
		print_coordinate_distribution("KMeans Extra Points", extra_pts)
		draw_sample_pts(central_pts, extra_pts, num_plot)
		num_plot += 1
		kmeans_center.fit(np.concatenate([central_pts, extra_pts]))
		print_coordinate_distribution("Unused KMeans Central Points", kmeans_center.cluster_centers_)
		# central_pts[...] = kmeans_center.cluster_centers_
		var = np.diag(np.var(kmeans_center.cluster_centers_, 0))
		extra_pts = kmeans_extra.fit(np.concatenate([extra_pts, kmeans_extra.fit(np.concatenate([dist.rng.multivariate_normal(pt, var, EXTRA_RATIO ** 2, "raise", method="cholesky") for pt in kmeans_center.cluster_centers_])).cluster_centers_])).cluster_centers_
		print_coordinate_distribution("KMeans Extra Points", extra_pts)
		draw_sample_pts(central_pts, extra_pts, num_plot)
		num_plot += 1
		return central_pts, extra_pts

	def __init__(
		self,
		dist: distribution,
		draw_sample_pts: collections.abc.Callable[[npt.NDArray[np.double], npt.NDArray[np.double], int], None]
	) -> None:
		self.x: npt.NDArray[np.double]
		self.extra_x: npt.NDArray[np.double]
		self.x, self.extra_x = __class__.__random_walk(dist, draw_sample_pts)
		self.y: npt.NDArray[np.double] = dist.dist_func(self.x)
		self.extra_y: npt.NDArray[np.double] = dist.dist_func(self.extra_x)
		self.x_all: npt.NDArray[np.double] = np.concatenate((self.x, self.extra_x))
		self.y_all: npt.NDArray[np.double] = np.concatenate((self.y, self.extra_y))
		self.x_test: npt.NDArray[np.double] = distribution.x_test
		self.y_test: npt.NDArray[np.double] = dist.y_test
		self.x_t: torch.Tensor = torch.from_numpy(self.x)
		self.y_t: torch.Tensor = torch.from_numpy(self.y)
		self.extra_x_t: torch.Tensor = torch.from_numpy(self.extra_x)
		self.extra_y_t: torch.Tensor = torch.from_numpy(self.extra_y)
		self.x_all_t: torch.Tensor = torch.from_numpy(self.x_all)
		self.y_all_t: torch.Tensor = torch.from_numpy(self.y_all)
		self.x_test_t: torch.Tensor = torch.from_numpy(self.x_test)
		self.y_test_t: torch.Tensor = torch.from_numpy(self.y_test)
