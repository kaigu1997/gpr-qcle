"""
sample
======

This module provides methods for sampling.
"""
import collections.abc
import os
import sys
import typing

import numpy as np
import numpy.typing as npt
import sklearn.cluster
import torch

sys.path.append(os.path.dirname(__file__))

import evolve
import pes

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
	# return np_rng.multivariate_normal(mean, np.diag(stddev ** 2), num_points, "raise", method="eigh")


class Points:
	"""
	The class to evolve and sample points used to construct GPR

	Parameters
	----------
	init_dist : pes.InitialDistribution
		Initial distribution
	num_pts : int, optional
		The number of central points to sample, by default __NUM_PTS
	extra_ratio : int, optional
		The ratio of extra points vs central points, by default __NUM_XTR_RATIO

	Attributes
	----------
	__dest_row_idx : npt.NDArray[np.int_]
		Possible indices of the row of the destinations from the starting lower-triangular element. It must be no less than corresponding `__dest_col_idx`
	__dest_col_idx : npt.NDArray[np.int_]
		Possible indices of the column of the destinations from the starting lower-triangular element. It must be no greater than corresponding `__dest_row_idx`
	__dest_idx : npt.NDArray[np.int_]
		Possible row-major element indices of the destinations from  the starting lower-triangular element.
	__coup_row_idx : npt.NDArray[np.int_]
		Row indices of the coupling element between the starting element and the destination. This is the different row/column index of the destination.
	__coup_col_idx : npt.NDArray[np.int_]
		Column indices of the coupling element between the starting element and the destination. This is the different row/column index of the source.
	__NUM_PTS : typing.Literal[256]
		The number of central points
	__NUM_XTR_RATIO : typing.Literal[50]
		The ratio of the number of extra points over the number of central points

	Methods
	-------
	num_center()
		To calculate the number of points in the center that corresponding to each element
	center()
		To give the phase space coordinates of each density matrix element
	density()
		To give the density matrix element
	rescale_factor()
		To give the rescale factor that makes `max|rho|==1`
	__print_coordinate_distribution(title, coordinates)
		To print the average and standard deviation of the points
	__sample_extra_points(all_points)
		To create the extra point set
	print_belonging(belong_file)
		To print the belonging index of the central points
	evolve(mass, dt, predictor)
		Non-adiabatic dynamics with surface hopping
	"""
	__dest_row_idx: npt.NDArray[np.int_]
	__dest_col_idx: npt.NDArray[np.int_]
	# k = np.arange(n).reshape(n, 1), l = np.arange(n).reshape(1, n)
	# mask_diff_col = (k == i[:, None, None] and l != j[:, None, None])
	# mask_diff_row (k != i[:, None, None] and l == j[:, None, None])
	# all_values = np.tile(np.stack(np.meshgrid(np.arange(n), np.arange(n)), 0).reshape(2, 1, n, n), (1, 10, 1, 1))
	# np.concat((all_values[:, mask_diff_row].reshape(2, len(i), n - 1), all_values[:, mask_diff_col].reshape(2, len(i), n - 1)), -1)
	__dest_row_idx, __dest_col_idx = np.concatenate(
		(
			np.tile(np.stack(np.meshgrid(np.arange(pes.NUM_PES), np.arange(pes.NUM_PES), indexing='ij'), axis=0)[:, np.newaxis], (1, pes.NUM_TRIG, 1, 1))[:, np.logical_and((np.arange(pes.NUM_PES) == pes.tril_row_indices[:, np.newaxis])[..., np.newaxis], np.arange(pes.NUM_PES) != pes.tril_col_indices[:, np.newaxis, np.newaxis])].reshape(2, pes.NUM_TRIG, pes.NUM_PES - 1),
			np.tile(np.stack(np.meshgrid(np.arange(pes.NUM_PES), np.arange(pes.NUM_PES), indexing='ij'), axis=0)[:, np.newaxis], (1, pes.NUM_TRIG, 1, 1))[:, np.logical_and((np.arange(pes.NUM_PES) != pes.tril_row_indices[:, np.newaxis])[..., np.newaxis], np.arange(pes.NUM_PES) == pes.tril_col_indices[:, np.newaxis, np.newaxis])].reshape(2, pes.NUM_TRIG, pes.NUM_PES - 1)
		),
		-1
	)
	__coup_row_idx: typing.Final[npt.NDArray[np.int_]] = np.concatenate((__dest_col_idx[:, :pes.NUM_PES - 1], __dest_row_idx[:, pes.NUM_PES - 1:]), -1) # also the diff index of destination
	__coup_col_idx: typing.Final[npt.NDArray[np.int_]] = np.repeat(np.concatenate((pes.tril_col_indices[:, np.newaxis], pes.tril_row_indices[:, np.newaxis]), -1), pes.NUM_PES - 1, -1) # also the diff index of source
	__dest_row_idx, __dest_col_idx = np.maximum(__dest_row_idx, __dest_col_idx), np.minimum(__dest_row_idx, __dest_col_idx)
	__dest_idx: typing.Final[npt.NDArray[np.int_]] = __dest_row_idx * pes.NUM_PES + __dest_col_idx
	__NUM_PTS: typing.Final = 256
	__NUM_XTR_RATIO: typing.Final = 50

	@staticmethod
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
				print("rho({}, {}), <r> = {}, stddev = {}".format(iPES, jPES, np.average(all_points[TrilIndex][:num_points[TrilIndex]], 0), np.std(all_points[TrilIndex][:num_points[TrilIndex]], 0)), flush=True)

	@staticmethod
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

	def __init__(self, init_dist: pes.InitialDistribution, num_pts: int = __NUM_PTS, extra_ratio: int = __NUM_XTR_RATIO) -> None:
		self.extra_ratio = extra_ratio
		self.gp_pts: list[npt.NDArray[np.double]] = [arr for arr in np.tile(normal_sample(num_pts, init_dist.r0, init_dist.sigma_r0)[np.newaxis], (pes.NUM_TRIG, 1 + extra_ratio, 1))]
		self.num_points: npt.NDArray[np.int_] = np.full(pes.NUM_TRIG, num_pts)
		__class__.sample_extra_points(self.num_points, self.gp_pts)
		__class__.sample_central_points(self.num_points, self.gp_pts)
		__class__.sample_extra_points(self.num_points, self.gp_pts)
		self.gp_density: list[npt.NDArray[np.cdouble]] = [init_dist(pts, idx) for pts, idx in zip(self.gp_pts, pes.tril_element_indices)]
		# prepare for surface-hopping part
		self.sh_pts: npt.NDArray[np.double] = np.concatenate([arr[:num_pts] for arr in self.gp_pts], 0)
		self.sh_belonging_idx: npt.NDArray[np.int_] = np.repeat(pes.tril_element_indices, num_pts, 0)
		self.sh_density: npt.NDArray[np.cdouble] = np.concatenate([den[:num_pts] for den in self.gp_density], 0)

	@property
	def num_center(self) -> npt.NDArray[np.int_]:
		"""
		To calculate the number of points in the center that corresponding to each element

		Returns
		-------
		npt.NDArray[np.int_]
			Central points cooresponding to each element
		"""
		return self.num_points

	@property
	def center(self) -> list[npt.NDArray[np.double]]:
		"""
		To give the phase space coordinates of each density matrix element

		Returns
		-------
		list[npt.NDArray[np.double]]
			Coordinates of each density matrix element
		"""
		return self.gp_pts

	@property
	def density(self) -> list[npt.NDArray[np.cdouble]]:
		"""
		To give the density matrix element

		Returns
		-------
		list[npt.NDArray[np.cdouble]]
			Density matrix element, cooresponding to `center()`
		"""
		return self.gp_density

	@property
	def rescale_factor(self) -> npt.NDArray[np.double]:
		"""
		To give the rescale factor that makes `max|rho|==1`

		Returns
		-------
		npt.NDArray[np.double]
			Rescale factor that scale up to 1.0; if all samples are 0, return 0
		"""
		result: npt.NDArray[np.double] = np.empty((pes.NUM_PES, pes.NUM_PES), np.double)
		for iPES, jPES, iTrig in zip(pes.tril_row_indices, pes.tril_col_indices, np.arange(pes.NUM_TRIG)):
			element_density: npt.NDArray[np.cdouble] = self.gp_density[iTrig]
			if iPES == jPES:
				result[iPES, jPES] = 0 if np.all(element_density == 0.0) else 1.0 / np.max(np.abs(element_density))
			else:
				result[iPES, jPES] = 0 if np.all(element_density.imag == 0.0) else 1.0 / np.max(np.abs(element_density.imag))
				result[jPES, iPES] = 0 if np.all(element_density.real == 0.0) else 1.0 / np.max(np.abs(element_density.real))
		return result.reshape(-1)

	def print_belonging(self, belong_file: typing.IO) -> None:
		"""
		To print the belonging index of the central points

		Notice the belonging index is in its original order (central -> extra, keep SH points at original place),
		while the points / density are sorted by their corresponding density matrix element.
		In other words, indices and points / density are not in correspondence with each other,
		while points and density corresponds to each other.

		Parameters
		----------
		belong_file : io.TextIOWrapper
			The file to save the belonging indices
		"""
		np.savetxt(belong_file, np.repeat(np.concat([[idx, idx + pes.NUM_ELM] for idx in pes.tril_element_indices]), np.concat([[n, pt.shape[0] - n] for n, pt in zip(self.num_points, self.gp_pts)]))[np.newaxis], footer='\n', comments="")

	@staticmethod
	def sh_evolve(
		points: npt.NDArray[np.double],
		densities: npt.NDArray[np.cdouble],
		belonging_idx: npt.NDArray[np.int_],
		mass: npt.NDArray[np.double],
		dt: float,
		predictor: collections.abc.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]
	) -> None:
		"""
		_summary_

		Parameters
		----------
		points : npt.NDArray[np.double], shape of (NUM_TRIG * NUM_PTS, PHASEDIM)
			Phase space coordinates of selected points for each density matrix element
		densities : npt.NDArray[np.cdouble], shape of (NUM_TRIG * NUM_PTS,)
			Density matrix element of the points
		belonging_idx : npt.NDArray[np.int_], shape of (NUM_TRIG * NUM_PTS,)

		mass : npt.NDArray[np.double], shape of (DIM,)
			Mass of classical degree of freedom
		dt : float
			Time interval
		predictor : collections.abc.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]
			It predicts the density matrix element based on given coordinates and element index
		"""
		# evolve coordinates and hopping
		x0: npt.NDArray[np.double] = points[:, :pes.DIM] # N * D
		p0: npt.NDArray[np.double] = points[:, pes.DIM:] # N * D
		x2: npt.NDArray[np.double] = np.empty_like(x0) # N * D
		p1: npt.NDArray[np.double] = np.empty_like(p0) # N * D
		x4: npt.NDArray[np.double] = np.empty_like(x0) # N * D
		p2: npt.NDArray[np.double] = np.empty_like(p0) # N * D
		current_indices: npt.NDArray[np.bool_] = belonging_idx == pes.tril_element_indices[:, np.newaxis]
		for iBelongPES in range(pes.NUM_PES):
			for jBelongPES in range(iBelongPES + 1):
				TrilIndex = pes.flatten_tril_index[iBelongPES, jBelongPES]
				indices: npt.NDArray[np.int_] = np.flatnonzero(current_indices[TrilIndex]) # n
				# get x and p, and 2 semi adiabatic steps
				x2[indices], p1[indices] = evolve.evolve_coordinates_adiabatically(x0[indices], p0[indices], mass, dt / 2.0, evolve.Direction.FORWARD, iBelongPES, jBelongPES)
				x4[indices], p2[indices] = evolve.evolve_coordinates_adiabatically(x2[indices], p1[indices], mass, dt / 2.0, evolve.Direction.FORWARD, iBelongPES, jBelongPES)
				# surface hopping
				# choose the one to jump to
				num_pts: int = indices.size
				idx_of_dest_idx: npt.NDArray[np.int_] = np_rng.integers(0, 2 * pes.NUM_PES - 2, size=num_pts, dtype=np.int_) # n
				velocity: npt.NDArray[np.double] = p2[indices] / mass # n * D
				coupling: npt.NDArray[np.double] = pes.adiabatic_coupling(x4[indices])[np.arange(num_pts), :, __class__.__coup_row_idx[TrilIndex, idx_of_dest_idx], __class__.__coup_col_idx[TrilIndex, idx_of_dest_idx]] # n * D
				transition_rate: npt.NDArray[np.double] = np.abs(np.sum(velocity * coupling, -1) * dt) # n
				transition_prob: npt.NDArray[np.double] = transition_rate / (1.0 + transition_rate) # n
				# energy conservation
				potential: npt.NDArray[np.double] = pes.adiabatic_potential(x4[indices]) # n * N
				momentum_rescale_factor_sq: npt.NDArray[np.double] = 1.0\
					+ (potential[np.arange(num_pts), __class__.__coup_col_idx[TrilIndex, idx_of_dest_idx]] - potential[np.arange(num_pts), __class__.__coup_row_idx[TrilIndex, idx_of_dest_idx]]) / np.sum(p2[indices] ** 2 / mass, -1) # n
				# judgment
				transition: npt.NDArray[np.bool_] = np.logical_and(np_rng.random(num_pts) < transition_prob, momentum_rescale_factor_sq >= 0.0) # n
				# change index, momentum, and the back propagation
				if np.any(transition):
					belonging_idx[indices[transition]] = __class__.__dest_idx[TrilIndex, idx_of_dest_idx[transition]] # indices[transition] equivalent to [current_indices[TrilIndex]][transition]
					p2[indices[transition]] *= np.sqrt(momentum_rescale_factor_sq[transition])[:, np.newaxis]
					# change density
					for iPredictPES in range(pes.NUM_PES):
						for jPredictPES in range(iPredictPES + 1):
							if iBelongPES == iPredictPES and jBelongPES == jPredictPES:
								pass
							PredictElementIndex: int = iPredictPES * pes.NUM_PES + jPredictPES
							need_predict: npt.NDArray[np.bool_] = np.logical_and(transition, __class__.__dest_idx[TrilIndex, idx_of_dest_idx] == PredictElementIndex) # in case the transition happens to this element
							if np.any(need_predict):
								x2[indices[need_predict]], p1[indices[need_predict]] = evolve.evolve_coordinates_adiabatically(x4[indices[need_predict]], p2[indices[need_predict]], mass, dt / 2.0, evolve.Direction.BACKWARD, iPredictPES, jPredictPES)
								x0[indices[need_predict]], p0[indices[need_predict]] = evolve.evolve_coordinates_adiabatically(x2[indices[need_predict]], p1[indices[need_predict]], mass, dt / 2.0, evolve.Direction.BACKWARD, iPredictPES, jPredictPES)
								densities[indices[need_predict]] = predictor(points[indices[need_predict]], PredictElementIndex)
		# evolve density
		current_indices: npt.NDArray[np.bool_] = belonging_idx == pes.tril_element_indices[:, np.newaxis]
		for iPES in range(pes.NUM_PES):
			for jPES in range(iPES + 1):
				TrilIndex = pes.flatten_tril_index[iPES, jPES]
				densities[current_indices[TrilIndex]] = evolve.evolve_density_non_adiabatically(
					densities[current_indices[TrilIndex]],
					x4[current_indices[TrilIndex]],
					p2[current_indices[TrilIndex]],
					x2[current_indices[TrilIndex]],
					p1[current_indices[TrilIndex]],
					mass,
					dt,
					predictor,
					iPES,
					jPES
				)
				# finally set up the point coordinates
				points[current_indices[TrilIndex], :pes.DIM] = x4[current_indices[TrilIndex]]
				points[current_indices[TrilIndex], pes.DIM:] = p2[current_indices[TrilIndex]]

	def evolve(self, mass: npt.NDArray[np.double], dt: float, predictor: collections.abc.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]) -> None:
		evolve.evolve(self.gp_pts, self.gp_density, mass, dt, predictor)
		__class__.sh_evolve(self.sh_pts, self.sh_density, self.sh_belonging_idx, mass, dt, predictor)

	def exchange_and_resample(self, predictor: collections.abc.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]) -> None:
		self.num_points = np.unique(self.sh_belonging_idx, return_counts=True)[1]
		self.gp_pts = [np.tile(self.sh_pts[self.sh_belonging_idx == iTril], (1 + self.extra_ratio, 1)) for iTril in pes.tril_element_indices]
		self.gp_density = [np.tile(self.sh_density[self.sh_belonging_idx == iTril], 1 + self.extra_ratio) for iTril in pes.tril_element_indices]
		# point extra points and predict them
		__class__.sample_extra_points(self.num_points, self.gp_pts)
		for iPES in range(pes.NUM_PES):
			for jPES in range(iPES + 1):
				TrilIndex: int = pes.flatten_tril_index[iPES, jPES]
				self.gp_density[TrilIndex][self.num_points[TrilIndex]:] = predictor(self.gp_pts[TrilIndex][self.num_points[TrilIndex]:], iPES * pes.NUM_PES + jPES)
