"""
point
=====

"""
import io
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
import utility

SEED = 0
np_rng: np.random.Generator = np.random.Generator(np.random.MT19937(SEED))
torch.manual_seed(SEED)


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


class Points:
	"""
	The class to evolve and sample points used to construct GPR

	Parameters
	----------
	init_dist : pes.InitialDistribution
		Initial distribution

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
	__sample_extra_points(all_points)
		To create the extra point set
	print_belonging(belong_file)
		To print the belonging index of the central points
	evolve(mass, dt, predictor)
		Non-adiabatic dynamics with surface hopping
	sh_to_gp(predictor)
		To copy the surface hopping (SH) points to Gaussian Process (GP), and re-sample the extra points for GP
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
	__coup_row_idx: npt.NDArray[np.int_] = np.concatenate((__dest_col_idx[:, :pes.NUM_PES - 1], __dest_row_idx[:, pes.NUM_PES - 1:]), -1) # also the diff index of destination
	__coup_col_idx: npt.NDArray[np.int_] = np.repeat(np.concatenate((pes.tril_col_indices[:, np.newaxis], pes.tril_row_indices[:, np.newaxis]), -1), pes.NUM_PES - 1, -1) # also the diff index of source
	__dest_row_idx, __dest_col_idx = np.maximum(__dest_row_idx, __dest_col_idx), np.minimum(__dest_row_idx, __dest_col_idx)
	__dest_idx: npt.NDArray[np.int_] = __dest_row_idx * pes.NUM_PES + __dest_col_idx
	__NUM_PTS = 256
	__NUM_XTR_RATIO = 50
	__slots__: tuple = ("__extra_ratio", "__gp_num", "__gp_pts", "__gp_den", "__sh_pts", "__sh_idx", "__sh_den")

	@staticmethod
	def __sample_extra_points(central_points: npt.NDArray[np.double], extra_ratio: int) -> npt.NDArray[np.double]:
		"""
		To create the extra point set

		First `num_points` points remain the same, and the rest of the points are resampled based on the first `num_points` points and their variance

		Parameters
		----------
		central_points : shape of (num_point, PHASEDIM)
			Points who will be used as center of sampling
		extra_ratio : int
			The number of points around each central point
		"""
		stddev: npt.NDArray[np.double] = np.std(central_points, 0)
		result: npt.NDArray[np.double] = np.concatenate([normal_sample(extra_ratio, pt, stddev) for pt in central_points])
		print("Sample Extra Points: {}, {}".format(
			utility.format_array("<r>", np.average(result, 0)),
			utility.format_array("stddev", np.std(result, 0))
		))
		return result

	def __init__(self, init_dist: pes.InitialDistribution, num_pts: int = __NUM_PTS, extra_ratio: int = __NUM_XTR_RATIO) -> None:
		self.__extra_ratio: int = extra_ratio
		self.__gp_num: npt.NDArray[np.int_] = np.full(pes.NUM_TRIG, num_pts)
		self.__gp_pts: list[npt.NDArray[np.double]] = [arr for arr in np.tile(normal_sample(num_pts, init_dist.r0, init_dist.sigma_r0), (pes.NUM_TRIG, 1 + self.__extra_ratio, 1))]
		print("Sample Central Points: {}, {}".format(
			utility.format_array("<r>", np.average(self.__gp_pts[0][:num_pts], 0)),
			utility.format_array("stddev", np.std(self.__gp_pts[0][:num_pts], 0))
		))
		for idx, num, pts in zip(pes.tril_element_indices, self.__gp_num, self.__gp_pts):
			print("rho({}, {})".format(idx // pes.NUM_PES, idx % pes.NUM_PES))
			pts[num:] = __class__.__sample_extra_points(pts[:num], self.__extra_ratio)
			kmeans: sklearn.cluster.KMeans = sklearn.cluster.KMeans(num, init="k-means++", n_init="auto", random_state=np.random.RandomState(np_rng.bit_generator), algorithm="lloyd").fit(pts)
			pts[:num] = kmeans.cluster_centers_
			print("KMeans Central Points: {}, {}".format(
				utility.format_array("<r>", np.average(pts[:num], 0)),
				utility.format_array("stddev", np.std(pts[:num], 0))
			))
			pts[num:] = __class__.__sample_extra_points(pts[:num], self.__extra_ratio)
			pts[num:] = __class__.__sample_extra_points(kmeans.fit(pts).cluster_centers_, self.__extra_ratio)
		self.__gp_den: list[npt.NDArray[np.cdouble]] = [init_dist(pts, idx) for pts, idx in zip(self.__gp_pts, pes.tril_element_indices)] # N_PT
		self.__sh_pts: npt.NDArray[np.double] = np.concatenate([pts[:num] for num, pts in zip(self.__gp_num, self.__gp_pts)], 0)
		self.__sh_idx: npt.NDArray[np.int_] = np.repeat(pes.tril_element_indices, self.__gp_num, 0)
		self.__sh_den: npt.NDArray[np.cdouble] = np.concatenate([init_dist(pts[:num], idx) for num, pts, idx in zip(self.__gp_num, self.__gp_pts, pes.tril_element_indices)]) # tril only
		print("", end="", flush=True)

	@property
	def num_center(self) -> npt.NDArray[np.int_]:
		return self.__gp_num

	@property
	def center(self) -> list[npt.NDArray[np.double]]:
		return self.__gp_pts
	
	@property
	def density(self) -> list[npt.NDArray[np.cdouble]]:
		return self.__gp_den

	@property
	def rescale_factor(self) -> npt.NDArray[np.double]:
		result: npt.NDArray[np.double] = np.empty((pes.NUM_PES, pes.NUM_PES), np.double)
		for iPES in range(pes.NUM_PES):
			for jPES in range(iPES + 1):
				TrilIndex: int = pes.flatten_tril_index[iPES, jPES]
				if iPES == jPES:
					result[iPES, jPES] = 0 if np.all(self.__gp_den[TrilIndex] == 0.0) else 1.0 / np.max(np.abs(self.__gp_den[TrilIndex]))
				else:
					result[iPES, jPES] = 0 if np.all(self.__gp_den[TrilIndex].imag == 0.0) else 1.0 / np.max(np.abs(self.__gp_den[TrilIndex].imag))
					result[jPES, iPES] = 0 if np.all(self.__gp_den[TrilIndex].real == 0.0) else 1.0 / np.max(np.abs(self.__gp_den[TrilIndex].real))
		return result.reshape(-1)

	def print_belonging(self, belong_file: io.TextIOWrapper) -> None:
		"""
		To print the belonging index of the central points

		Parameters
		----------
		belong_file : io.TextIOWrapper
			The file to save the belonging indices
		"""
		np.savetxt(
			belong_file,
			np.concatenate([np.repeat(pes.tril_element_indices, [num * (1 + self.__extra_ratio) for num in self.__gp_num])])[np.newaxis],
			fmt="%d",
			footer="\n",
			comments=""
		)

	def evolve(
		self,
		mass: npt.NDArray[np.double],
		dt: float,
		predictor: typing.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]],
		purity: npt.NDArray[np.double]
	) -> None:
		"""
		Non-adiabatic dynamics with surface hopping

		Parameters
		----------
		mass : npt.NDArray[np.double], shape of (DIM,)
			Mass of classical degree of freedom
		dt : float
			Time interval
		predictor : typing.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]
			It predicts the density matrix element based on given coordinates and element index
		purity : npt.NDArray[np.double], shape of (NUM_PES, NUM_PES)
			The purity of each element, indicating the transition allowance to other elements
		"""
		x0: npt.NDArray[np.double] = self.__sh_pts[:, :pes.DIM] # N * D, slice of centers
		p0: npt.NDArray[np.double] = self.__sh_pts[:, pes.DIM:] # N * D, slice of centers
		x2: npt.NDArray[np.double] = np.empty_like(x0) # N * D
		p1: npt.NDArray[np.double] = np.empty_like(p0) # N * D
		x4: npt.NDArray[np.double] = np.empty_like(x0) # N * D
		p2: npt.NDArray[np.double] = np.empty_like(p0) # N * D
		current_indices: npt.NDArray[np.bool_] = self.__sh_idx == pes.tril_element_indices[:, np.newaxis]
		# evolve coordinates and hopping for SH points
		for iBelongPES in range(pes.NUM_PES):
			for jBelongPES in range(iBelongPES + 1):
				TrilIndex = pes.flatten_tril_index[iBelongPES, jBelongPES]
				indices: npt.NDArray[np.int_] = np.flatnonzero(current_indices[TrilIndex]) # NUM_PT
				num_pts: int = indices.size
				if num_pts > 0: # only have elements
					# get x and p, and 2 semi adiabatic steps for all points
					x2[indices], p1[indices] = evolve.evolve_coordinates_adiabatically(x0[indices], p0[indices], mass, dt / 2.0, evolve.Direction.Forward, iBelongPES, jBelongPES)
					x4[indices], p2[indices] = evolve.evolve_coordinates_adiabatically(x2[indices], p1[indices], mass, dt / 2.0, evolve.Direction.Forward, iBelongPES, jBelongPES)
					# surface hopping for central points only
					num_states_avail: int = (1 if iBelongPES == jBelongPES else 2) * (pes.NUM_PES - 1)
					state_prob: npt.NDArray[np.double] = np.zeros((num_pts, num_states_avail + 1)) # +1 for keep the same
					# purity factor
					current_purity: float = abs(purity[iBelongPES, jBelongPES])
					dest_purity: npt.NDArray[np.double] = np.abs(purity[__class__.__dest_row_idx[TrilIndex, :num_states_avail], __class__.__dest_col_idx[TrilIndex, :num_states_avail]]) # NUM_STATE
					purity_ratio: npt.NDArray[np.double] = current_purity / (current_purity + dest_purity)
					# weights
					velocity: npt.NDArray[np.double] = p2[indices] / mass # NUM_PT * DIM
					coupling: npt.NDArray[np.double] = pes.adiabatic_coupling(x4[indices])[..., __class__.__coup_row_idx[TrilIndex, :num_states_avail], __class__.__coup_col_idx[TrilIndex, :num_states_avail]] # NUM_PT * DIM * NUM_STATE
					state_prob[:, :-1] = np.abs(np.sum(velocity[..., np.newaxis] * coupling, -2) * dt) # NUM_PT * NUM_STATE, |v*d*dt|
					# energy conservation rule
					potential: npt.NDArray[np.double] = pes.adiabatic_potential(x4[indices]) # NUM_PT * NUM_PES
					momentum_rescale_factor_sq: npt.NDArray[np.double] = 1.0\
						+ (potential[:, __class__.__coup_col_idx[TrilIndex, :num_states_avail]] - potential[:, __class__.__coup_row_idx[TrilIndex, :num_states_avail]]) / np.sum(p2[indices] ** 2 / mass, -1, keepdims=True) # NUM_PT * NUM_STATE
					state_prob[:, :-1] *= np.where(momentum_rescale_factor_sq >= 0.0, purity_ratio, 0.0)
					state_prob[:, -1] = 1.0 # stay at original state
					state_prob /= np.sum(state_prob, -1, keepdims=True) # normalize
					# choose the one to jump to
					idx_of_dest_idx: npt.NDArray[np.int_] = np.array([np_rng.choice(num_states_avail + 1, p=p) for p in state_prob]) # NUM_PT
					# change index, momentum, and the back propagation
					transition: npt.NDArray[np.bool_] = idx_of_dest_idx != num_states_avail
					if np.any(transition):
						transit_indices: npt.NDArray[np.int_] = indices[transition] # indices[transition] equivalent to [current_indices[TrilIndex]][transition]
						transit_idx_of_dest_idx: npt.NDArray[np.int_] = idx_of_dest_idx[transition]
						self.__sh_idx[transit_indices] = __class__.__dest_idx[TrilIndex, transit_idx_of_dest_idx]
						p2[transit_indices] *= np.sqrt(momentum_rescale_factor_sq[transition, transit_idx_of_dest_idx])[:, np.newaxis]
						# change density
						for iPredictPES in range(pes.NUM_PES):
							for jPredictPES in range(iPredictPES + 1):
								if iBelongPES == iPredictPES and jBelongPES == jPredictPES:
									pass
								PredictElementIndex: int = iPredictPES * pes.NUM_PES + jPredictPES
								need_predict_idx: npt.NDArray[np.int_] = transit_indices[__class__.__dest_idx[TrilIndex, transit_idx_of_dest_idx] == PredictElementIndex] # in case the transition happens to this element
								if need_predict_idx.size > 0:
									x2[need_predict_idx], p1[need_predict_idx] = evolve.evolve_coordinates_adiabatically(x4[need_predict_idx], p2[need_predict_idx], mass, dt / 2.0, evolve.Direction.Backward, iPredictPES, jPredictPES)
									x0[need_predict_idx], p0[need_predict_idx] = evolve.evolve_coordinates_adiabatically(x2[need_predict_idx], p1[need_predict_idx], mass, dt / 2.0, evolve.Direction.Backward, iPredictPES, jPredictPES)
									self.__sh_den[need_predict_idx] = predictor(self.__sh_pts[need_predict_idx], PredictElementIndex) # change the density of the new element of the backtraced point
		# evolve density
		for iBelongPES in range(pes.NUM_PES):
			for jBelongPES in range(iBelongPES + 1):
				filters: npt.NDArray[np.bool_] = self.__sh_idx == iBelongPES * pes.NUM_PES + jBelongPES
				if indices.size > 0:
					self.__sh_den[filters] = evolve.evolve_density_non_adiabatically(
						self.__sh_den[filters], # same as self.__density[BelongTrilIndex]
						x4[filters],
						p2[filters],
						x2[filters],
						p1[filters],
						mass,
						dt,
						predictor,
						iBelongPES,
						jBelongPES
					)
		# change coordinates
		self.__sh_pts[:, :pes.DIM] = x4
		self.__sh_pts[:, pes.DIM:] = p2
		# evolve gp points
		evolve.evolve(self.__gp_pts, self.__gp_den, mass, dt, predictor)

	def sh_to_gp(self, predictor: typing.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]) -> None:
		"""
		To copy the surface hopping (SH) points to Gaussian Process (GP), and re-sample the extra points for GP

		Parameters
		----------
		predictor : typing.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]
			It predicts the density matrix element based on given coordinates and element index
		"""
		self.__gp_num = np.unique(self.__sh_idx, return_counts=True)[1]
		self.__gp_pts = [np.tile(self.__sh_pts[self.__sh_idx == idx], (1 + self.__extra_ratio, 1)) for idx in pes.tril_element_indices]
		self.__gp_den = [np.tile(self.__sh_den[self.__sh_idx == idx], 1 + self.__extra_ratio) for idx in pes.tril_element_indices]
		for num, pts, den, idx in zip(self.__gp_num, self.__gp_pts, self.__gp_den, pes.tril_element_indices):
			print("rho({}, {})".format(idx // pes.NUM_PES, idx % pes.NUM_PES))
			pts[num:] = __class__.__sample_extra_points(pts[:num], self.__extra_ratio)
			pts[num:] = __class__.__sample_extra_points(sklearn.cluster.KMeans(num, init="k-means++", n_init="auto", random_state=np.random.RandomState(np_rng.bit_generator), algorithm="lloyd").fit(pts).cluster_centers_, self.__extra_ratio)
			den[num:] = predictor(pts[num:], idx)
		print("", end="", flush=True)