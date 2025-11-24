r"""point
=====
This module provides methods for sampling.
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
import utility

SEED: typing.Final = 0
np_rng: np.random.Generator = np.random.Generator(np.random.MT19937(SEED))


def normal_sample(
	num_points: int,
	mean: npt.NDArray[np.double],
	stddev: npt.NDArray[np.double]
) -> npt.NDArray[np.double]:
	r"""To create normally distributed point set based on given mean and variance

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
	return np_rng.multivariate_normal(mean, np.diag(stddev ** 2), num_points, "raise", method="eigh")


class Points:
	r"""The class to evolve and sample points used to construct GPR

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
	__print_coordinate_distribution(title, coordinates)
		To print the average and standard deviation of the points
	__sample_extra_points(all_points)
		To create the extra point set
	num_center()
		To calculate the number of points in the center that corresponding to each element
	center()
		To give the phase space coordinates of each density matrix element
	density()
		To give the density matrix element
	rescale_factor()
		To give the rescale factor that makes `max|rho|==1`
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
	__slots__: tuple = ("__init_purity", "__num_all_centers", "__coordinate", "__index", "__density")

	@staticmethod
	def __print_coordinate_distribution(title: str, coordinates: npt.NDArray[np.double]) -> None:
		r"""To print the average and standard deviation of the points

		Parameters
		----------
		title : str
			What is the meaning of the given coordinates
		coordinates : npt.NDArray[np.double], shape of (NUM_POINTS, PHASEDIM)
			The phase space coordinates
		"""
		print(((title + ": ") if title != "" else "") + f"{utility.format_array("<r>", np.average(coordinates, 0))}, {utility.format_array("stddev", np.std(coordinates, 0))}")

	@staticmethod
	def __sample_extra_points(central_points: npt.NDArray[np.double], extra_ratio: int) -> npt.NDArray[np.double]:
		r"""To create the extra point set

		First `num_points` points remain the same, and the rest of the points are resampled based on the first `num_points` points and their variance

		Parameters
		----------
		central_points : shape of (num_point, PHASEDIM)
			Points who will be used as center of sampling
		extra_ratio : int
			The number of points around each central point
		"""
		stddev: typing.Final[npt.NDArray[np.double]] = np.std(central_points, 0)
		result: typing.Final[npt.NDArray[np.double]] = np.concatenate([normal_sample(extra_ratio, pt, stddev) for pt in central_points])
		__class__.__print_coordinate_distribution("Sample Extra Points", result)
		return result

	def __init__(
		self,
		init_dist: pes.InitialDistribution,
		num_pts: int = __NUM_PTS,
		extra_ratio: int = __NUM_XTR_RATIO
	) -> None:
		self.__init_purity: typing.Final[npt.NDArray[np.double]] = init_dist.weight ** 2
		self.__num_all_centers: typing.Final[int] = pes.NUM_TRIG * num_pts
		kmeans_center: sklearn.cluster.KMeans = sklearn.cluster.KMeans(num_pts, init="k-means++", n_init="auto", random_state=np.random.RandomState(np_rng.bit_generator), algorithm="lloyd")
		kmeans_extra: sklearn.cluster.KMeans = sklearn.cluster.KMeans(num_pts * extra_ratio, init="k-means++", n_init="auto", random_state=np.random.RandomState(np_rng.bit_generator), algorithm="lloyd")
		self.__coordinate: npt.NDArray[np.double] = np.tile(normal_sample(num_pts, init_dist.r0, init_dist.sigma_r0), (1 + extra_ratio, 1))
		__class__.__print_coordinate_distribution("Sample Central Points", self.__coordinate[:num_pts])
		self.__coordinate[num_pts:] = __class__.__sample_extra_points(self.__coordinate[:num_pts], extra_ratio)
		kmeans_center.fit(self.__coordinate)
		self.__coordinate[:num_pts] = kmeans_center.cluster_centers_
		__class__.__print_coordinate_distribution("KMeans Central Points", kmeans_center.cluster_centers_)
		kmeans_extra.fit(__class__.__sample_extra_points(kmeans_center.cluster_centers_, extra_ratio ** 2))
		__class__.__print_coordinate_distribution("KMeans Extra Points", kmeans_extra.cluster_centers_)
		self.__coordinate[num_pts:] = kmeans_extra.cluster_centers_
		# kmeans_center.fit(self.__coordinate)
		# __class__.__print_coordinate_distribution("KMeans Central Points", kmeans_center.cluster_centers_)
		# self.__coordinate[:num_pts] = kmeans_center.cluster_centers_
		# kmeans_extra.fit(__class__.__sample_extra_points(kmeans_center.cluster_centers_, extra_ratio ** 2))
		# __class__.__print_coordinate_distribution("KMeans Extra Points", kmeans_extra.cluster_centers_)
		# self.__coordinate[num_pts:] = kmeans_extra.fit(np.concatenate([self.__coordinate[num_pts:], kmeans_extra.cluster_centers_])).cluster_centers_
		# __class__.__print_coordinate_distribution("KMeans Extra Points", kmeans_extra.cluster_centers_)
		__class__.__print_coordinate_distribution("Overall", self.__coordinate)
		self.__coordinate = np.concat([np.tile(self.__coordinate[:num_pts], (pes.NUM_TRIG, 1)), np.tile(self.__coordinate[num_pts:], (pes.NUM_TRIG, 1))]) # central pts of all elements, then extra pts of all elements
		self.__index: npt.NDArray[np.int_] = np.concat([np.repeat(pes.tril_element_indices, num_pts), np.repeat(pes.tril_element_indices, num_pts * extra_ratio)])
		self.__density: npt.NDArray[np.cdouble] = np.empty(self.__coordinate.shape[:-1], np.cdouble)
		for iTrig, iElement in enumerate(pes.tril_element_indices):
			central_start: int = iTrig * num_pts
			central_end: int = (iTrig + 1) * num_pts
			self.__density[central_start:central_end] = init_dist(self.__coordinate[central_start:central_end], iElement)
			extra_start: int = self.__num_all_centers + iTrig * num_pts * extra_ratio
			extra_end: int = self.__num_all_centers + (iTrig + 1) * num_pts * extra_ratio
			self.__density[extra_start:extra_end] = init_dist(self.__coordinate[extra_start:extra_end], iElement)
		print("", end="", flush=True)

	@property
	def num_center(self) -> npt.NDArray[np.int_]:
		r"""To calculate the number of points in the center that corresponding to each element

		Returns
		-------
		npt.NDArray[np.int_]
			Central points cooresponding to each element
		"""
		return (self.__index[:self.__num_all_centers] == pes.tril_element_indices[:, np.newaxis]).astype(np.int_).sum(-1)

	@property
	def center(self) -> list[npt.NDArray[np.double]]:
		r"""To give the phase space coordinates of each density matrix element

		Returns
		-------
		list[npt.NDArray[np.double]]
			Coordinates of each density matrix element
		"""
		return [self.__coordinate[self.__index == iElement] for iElement in pes.tril_element_indices]

	@property
	def density(self) -> list[npt.NDArray[np.cdouble]]:
		r"""To give the density matrix element

		Returns
		-------
		list[npt.NDArray[np.cdouble]]
			Density matrix element, cooresponding to `center()`
		"""
		return [self.__density[self.__index == iElement] for iElement in pes.tril_element_indices]

	@property
	def rescale_factor(self) -> npt.NDArray[np.double]:
		r"""To give the rescale factor that makes `max|rho|==1`

		Returns
		-------
		npt.NDArray[np.double]
			Rescale factor that scale up to 1.0; if all samples are 0, return 0
		"""
		result: npt.NDArray[np.double] = np.empty((pes.NUM_PES, pes.NUM_PES), np.double)
		for iPES, jPES, iElement in zip(pes.tril_row_indices, pes.tril_col_indices, pes.tril_element_indices):
			element_density: npt.NDArray[np.cdouble] = self.__density[self.__index == iElement]
			if iPES == jPES:
				result[iPES, jPES] = 0 if np.all(element_density == 0.0) else 1.0 / np.max(np.abs(element_density))
			else:
				result[iPES, jPES] = 0 if np.all(element_density.imag == 0.0) else 1.0 / np.max(np.abs(element_density.imag))
				result[jPES, iPES] = 0 if np.all(element_density.real == 0.0) else 1.0 / np.max(np.abs(element_density.real))
		return result.reshape(-1)

	def print_belonging(self, belong_file: typing.IO) -> None:
		r"""To print the belonging index of the central points

		Notice the belonging index is in its original order (central -> extra, keep SH points at original place),
		while the points / density are sorted by their corresponding density matrix element.
		In other words, indices and points / density are not in correspondence with each other,
		while points and density corresponds to each other.

		Parameters
		----------
		belong_file : io.TextIOWrapper
			The file to save the belonging indices
		"""
		np.savetxt(
			belong_file,
			np.repeat(np.concat([[idx, idx + pes.NUM_ELM] for idx in pes.tril_element_indices]), np.concat([[m, n - m] for m, n in zip(self.num_center, (self.__index == pes.tril_element_indices[:, np.newaxis]).astype(np.int_).sum(-1))]))[np.newaxis],
			footer="\n",
			comments=""
		)

	def evolve(
		self,
		mass: npt.NDArray[np.double],
		dt: float,
		predictor: pes.Predictor
	) -> None:
		r"""Non-adiabatic dynamics with surface hopping

		Parameters
		----------
		mass : npt.NDArray[np.double], shape of (DIM,)
			Mass of classical degree of freedom
		dt : float
			Time interval
		predictor : pes.Predictor
			It predicts the density matrix element based on given coordinates and element index
		purity : npt.NDArray[np.double], shape of (NUM_PES, NUM_PES)
			The purity of each element, indicating the transition allowance to other elements
		"""
		# evolve coordinates and hopping
		x0: npt.NDArray[np.double] = self.__coordinate[:, :pes.DIM] # N * D, slice of centers
		p0: npt.NDArray[np.double] = self.__coordinate[:, pes.DIM:] # N * D, slice of centers
		x2: npt.NDArray[np.double] = np.empty_like(x0) # N * D
		p1: npt.NDArray[np.double] = np.empty_like(p0) # N * D
		x4: npt.NDArray[np.double] = np.empty_like(x0) # N * D
		p2: npt.NDArray[np.double] = np.empty_like(p0) # N * D
		current_indices: npt.NDArray[np.bool_] = self.__index == pes.tril_element_indices[:, np.newaxis]
		for iTrig, (iBelongPES, jBelongPES) in enumerate(zip(pes.tril_row_indices, pes.tril_col_indices)):
			indices: npt.NDArray[np.int_] = np.flatnonzero(current_indices[iTrig]) # NUM_PTS
			num_pts: int = indices.size
			if num_pts > 0: # only have elements
				# get x and p, and 2 semi adiabatic steps for all points
				x2[indices], p1[indices] = evolve.evolve_coordinates_adiabatically(x0[indices], p0[indices], mass, dt / 2.0, evolve.Direction.FORWARD, iBelongPES, jBelongPES)
				x4[indices], p2[indices] = evolve.evolve_coordinates_adiabatically(x2[indices], p1[indices], mass, dt / 2.0, evolve.Direction.FORWARD, iBelongPES, jBelongPES)
				# surface hopping
				# choose the one to jump to
				idx_of_dest_idx: npt.NDArray[np.int_] = np_rng.integers(0, 2 * pes.NUM_PES - 2, size=num_pts, dtype=np.int_) # n
				velocity: npt.NDArray[np.double] = p2[indices] / mass # n * D
				coupling: npt.NDArray[np.double] = pes.adiabatic_coupling(x4[indices])[np.arange(num_pts), :, __class__.__coup_row_idx[iTrig, idx_of_dest_idx], __class__.__coup_col_idx[iTrig, idx_of_dest_idx]] # n * D
				transition_rate: npt.NDArray[np.double] = np.abs(np.sum(velocity * coupling, -1) * dt) # n
				transition_prob: npt.NDArray[np.double] = transition_rate / (1.0 + transition_rate) # n
				# energy conservation
				potential: npt.NDArray[np.double] = pes.adiabatic_potential(x4[indices]) # n * N
				momentum_rescale_factor_sq: npt.NDArray[np.double] = 1.0\
					+ (potential[np.arange(num_pts), __class__.__coup_col_idx[iTrig, idx_of_dest_idx]] - potential[np.arange(num_pts), __class__.__coup_row_idx[iTrig, idx_of_dest_idx]]) / np.sum(p2[indices] ** 2 / mass, -1) # n
				# judgment
				transition: npt.NDArray[np.bool_] = np.logical_and(np_rng.random(num_pts) < transition_prob, momentum_rescale_factor_sq >= 0.0) # n
				# change index, momentum, and the back propagation
				if np.any(transition):
					transit_indices: npt.NDArray[np.int_] = indices[transition] # indices[transition] equivalent to [current_indices[iTrig]][transition]
					transit_idx_of_dest_idx: npt.NDArray[np.int_] = idx_of_dest_idx[transition]
					self.__index[transit_indices] = __class__.__dest_idx[iTrig, transit_idx_of_dest_idx]
					p2[transit_indices] *= np.sqrt(momentum_rescale_factor_sq[transition]).reshape(-1, 1)
					# change density
					for iPredictPES, jPredictPES, iPredictElement in zip(pes.tril_row_indices, pes.tril_col_indices, pes.tril_element_indices):
						if iBelongPES == iPredictPES and jBelongPES == jPredictPES:
							pass
						need_predict_idx: npt.NDArray[np.int_] = transit_indices[__class__.__dest_idx[iTrig, transit_idx_of_dest_idx] == iPredictElement] # in case the transition happens to this element
						if need_predict_idx.size > 0:
							x2[need_predict_idx], p1[need_predict_idx] = evolve.evolve_coordinates_adiabatically(x4[need_predict_idx], p2[need_predict_idx], mass, dt / 2.0, evolve.Direction.BACKWARD, iPredictPES, jPredictPES)
							x0[need_predict_idx], p0[need_predict_idx] = evolve.evolve_coordinates_adiabatically(x2[need_predict_idx], p1[need_predict_idx], mass, dt / 2.0, evolve.Direction.BACKWARD, iPredictPES, jPredictPES)
							self.__density[need_predict_idx] = predictor(self.__coordinate[need_predict_idx], iPredictElement)
		# evolve density
		for iPES, jPES, iElement in zip(pes.tril_row_indices, pes.tril_col_indices, pes.tril_element_indices):
			filters: npt.NDArray[np.bool_] = self.__index == iElement
			if np.count_nonzero(filters) > 0:
				self.__density[filters] = evolve.evolve_density_non_adiabatically(
					self.__density[filters],
					x4[filters],
					p2[filters],
					x2[filters],
					p1[filters],
					mass,
					dt,
					predictor,
					iPES,
					jPES
				)
		# finally change coordinates
		self.__coordinate[:, :pes.DIM] = x4
		self.__coordinate[:, pes.DIM:] = p2
