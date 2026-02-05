r"""point
=====
Implementation for sampling.
"""
import datetime
import typing

import numpy as np
import sklearn.cluster
import torch
import torch_kmeans

import constant
import evolve
import pes
import plot

SEED: typing.Final = 0
torch.set_default_dtype(constant.DTYPE)
torch.set_default_device(constant.DEVICE)


def normal_sample(
	num_points: int,
	mean: torch.Tensor,
	stddev: torch.Tensor
) -> torch.Tensor:
	r"""To create normally distributed point set based on given mean and variance

	Parameters
	----------
	num_points : int
		The number of points needed
	mean : torch.Tensor, shape of (PHASEDIM,)
		The center of the points
	stddev : torch.Tensor, shape of (PHASEDIM,)
		The standard deviation of the points

	Returns
	-------
	torch.Tensor, shape of (NUM_PTS, PHASEDIM)
		Normally distributed point test
	"""
	return torch.randn((num_points, mean.numel()), device=stddev.device) * stddev + mean


@typing.final
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
	print_belonging(belong_file)
		To print the belonging index of the central points
	evolve(mass, dt, predictor)
		Non-adiabatic dynamics with surface hopping
	"""
	__NUM_PTS: typing.Final = 256
	__NUM_XTR_RATIO: typing.Final = 50
	__config: typing.Final[pes.ModelConfig]
	__dest_row_idx: typing.Final[torch.Tensor]
	__dest_col_idx: typing.Final[torch.Tensor]
	__coup_row_idx: typing.Final[torch.Tensor]
	__coup_col_idx: typing.Final[torch.Tensor]
	__dest_idx: typing.Final[torch.Tensor]
	__num_all_centers: typing.Final[int]
	__coordinate: torch.Tensor
	__density: torch.Tensor
	__index: torch.Tensor
	__tril_element_indices: typing.Final[torch.Tensor]
	__slots__: typing.Final[tuple] = ("__config", "__dest_row_idx", "__dest_col_idx", "__coup_row_idx", "__coup_col_idx", "__dest_idx", "__num_all_centers", "__coordinate", "__density", "__index", "__tril_element_indices")
	
	@staticmethod
	def __init_sampling(center: torch.Tensor, stddev: torch.Tensor, num_pts: int, extra_ratio: int) -> torch.Tensor:
		r"""To sample the initial points by k-means

		Parameters
		----------
		center : torch.Tensor
			Center of the initial distribution
		stddev : torch.Tensor
			Standard deviation of the initial distribution
		num_pts : int
			The number of central (inducing) points
		extra_ratio : int
			Ratio of extra points over central points

		Returns
		-------
		torch.Tensor, shape of (`num_pts` * (1 + `extra_ratio`), PHASEDIM)
			All points, first `num_pts` points are the central points, and the rest are the extra points
		"""
		def print_coordinate_distribution(title: str, coordinates: torch.Tensor) -> None:
			r"""To print the average and standard deviation of the points

			Parameters
			----------
			title : str
				What is the meaning of the given coordinates
			coordinates : torch.Tensor, shape of (NUM_POINTS, PHASEDIM)
				The phase space coordinates
			"""
			print(((title + ": ") if title != "" else "") + f"{plot.format_array(torch.mean(coordinates, 0), "<r>")}, {plot.format_array(torch.std(coordinates, 0), "stddev")}", datetime.datetime.now())

		def sample_extra_points(central_points: torch.Tensor, extra_ratio: int) -> torch.Tensor:
			r"""To create the extra point set

			Parameters
			----------
			central_points : torch.Tensor, shape of (num_point, PHASEDIM)
				Points who will be used as center of sampling
			extra_ratio : int
				The number of points around each central point

			Returns
			-------
			torch.Tensor
				Extra points
			"""
			stddev: typing.Final[torch.Tensor] = torch.std(central_points, 0)
			result: typing.Final[torch.Tensor] = torch.cat([normal_sample(extra_ratio, pt, stddev) for pt in central_points])
			print_coordinate_distribution("Sample Extra Points", result)
			return result

		kmeans_center: typing.Final = torch_kmeans.KMeans(init_method="k-means++", n_clusters=num_pts, seed=SEED)
		kmeans_extra: typing.Final = sklearn.cluster.KMeans(num_pts * extra_ratio, init="k-means++", n_init="auto", random_state=np.random.RandomState(np.random.Generator(np.random.MT19937(SEED)).bit_generator), algorithm="lloyd")
		coordinate: typing.Final[torch.Tensor] = torch.tile(normal_sample(num_pts, center, stddev), (1 + extra_ratio, 1))
		print_coordinate_distribution("Sample Central Points", coordinate[:num_pts])
		coordinate[num_pts:] = sample_extra_points(coordinate[:num_pts], extra_ratio)
		cen_cen: torch.Tensor = kmeans_center(coordinate[torch.newaxis]).centers[0]
		print_coordinate_distribution("KMeans Central Points", cen_cen)
		coordinate[:num_pts] = cen_cen

		xtr_cen: torch.Tensor = torch.from_numpy(kmeans_extra.fit(sample_extra_points(cen_cen.cpu(), extra_ratio ** 2).detach().numpy()).cluster_centers_)
		print_coordinate_distribution("KMeans Extra Points", xtr_cen)
		coordinate[num_pts:] = xtr_cen.to(coordinate.device)

		# cen_cen = kmeans_center(coordinate[torch.newaxis]).centers[0]
		# print_coordinate_distribution("KMeans Central Points", cen_cen)
		# coordinate[:num_pts] = cen_cen
		# xtr_cen = torch.from_numpy(kmeans_extra.fit(sample_extra_points(cen_cen.cpu(), extra_ratio ** 2).detach().numpy()).cluster_centers_)
		# print_coordinate_distribution("KMeans Extra Points", xtr_cen)
		# xtr_cen = torch.from_numpy(kmeans_extra.fit(torch.cat([coordinate[num_pts:].cpu(), xtr_cen]).detach().numpy()).cluster_centers_)
		# coordinate[num_pts:] = xtr_cen.to(coordinate.device)
		# print_coordinate_distribution("KMeans Extra Points", xtr_cen)

		print_coordinate_distribution("Overall", coordinate)
		print("", end="", flush=True)
		return coordinate.to(torch.get_default_device())


	def __init__(
		self,
		init_dist: pes.InitialDistribution,
		num_pts: int = __NUM_PTS,
		extra_ratio: int = __NUM_XTR_RATIO
	) -> None:
		# k = np.arange(n).reshape(n, 1), l = np.arange(n).reshape(1, n)
		# mask_diff_col = (k == i[:, None, None] and l != j[:, None, None])
		# mask_diff_row (k != i[:, None, None] and l == j[:, None, None])
		# all_values = np.tile(np.stack(np.meshgrid(np.arange(n), np.arange(n)), 0).reshape(2, 1, n, n), (1, 10, 1, 1))
		# np.concatenate((all_values[:, mask_diff_row].reshape(2, len(i), n - 1), all_values[:, mask_diff_col].reshape(2, len(i), n - 1)), -1)
		self.__config = init_dist.config
		pes_range: typing.Final[torch.Tensor] = torch.tensor(init_dist.config.PES_RANGE)
		tril_row_indices: typing.Final[torch.Tensor] = torch.tensor(init_dist.config.TRIL_ROW_INDICES)
		tril_col_indices: typing.Final[torch.Tensor] = torch.tensor(init_dist.config.TRIL_COL_INDICES)
		self.__dest_row_idx, self.__dest_col_idx = torch.cat(
			(
				torch.tile(torch.stack(torch.meshgrid(pes_range, pes_range, indexing='ij'), 0)[:, torch.newaxis], (1, init_dist.config.NUM_TRIG, 1, 1))[:, torch.logical_and((pes_range == tril_row_indices[:, torch.newaxis])[..., torch.newaxis], pes_range != tril_col_indices[:, torch.newaxis, torch.newaxis])].reshape(2, init_dist.config.NUM_TRIG, init_dist.config.NUM_PES - 1),
				torch.tile(torch.stack(torch.meshgrid(pes_range, pes_range, indexing='ij'), 0)[:, torch.newaxis], (1, init_dist.config.NUM_TRIG, 1, 1))[:, torch.logical_and((pes_range != tril_row_indices[:, torch.newaxis])[..., torch.newaxis], pes_range == tril_col_indices[:, torch.newaxis, torch.newaxis])].reshape(2, init_dist.config.NUM_TRIG, init_dist.config.NUM_PES - 1)
			),
			-1
		)
		self.__coup_row_idx = torch.cat((self.__dest_col_idx[:, :init_dist.config.NUM_PES - 1], self.__dest_row_idx[:, init_dist.config.NUM_PES - 1:]), -1) # also the diff index of destination
		self.__coup_col_idx = torch.repeat_interleave(torch.cat((tril_col_indices[:, torch.newaxis], tril_row_indices[:, torch.newaxis]), -1), init_dist.config.NUM_PES - 1, -1) # also the diff index of source
		self.__dest_row_idx, self.__dest_col_idx = torch.maximum(self.__dest_row_idx, self.__dest_col_idx), torch.minimum(self.__dest_row_idx, self.__dest_col_idx)
		self.__dest_idx = self.__dest_row_idx * init_dist.config.NUM_PES + self.__dest_col_idx
		self.__num_all_centers = init_dist.config.NUM_TRIG * num_pts
		self.__coordinate = Points.__init_sampling(init_dist.r0, init_dist.sigma_r0, num_pts, extra_ratio)
		self.__density = torch.cat((init_dist(self.__coordinate[:num_pts])[..., init_dist.config.TRIL_ROW_INDICES, init_dist.config.TRIL_COL_INDICES].T.reshape(-1), init_dist(self.__coordinate[num_pts:])[..., init_dist.config.TRIL_ROW_INDICES, init_dist.config.TRIL_COL_INDICES].T.reshape(-1)))
		self.__coordinate = torch.cat([torch.tile(self.__coordinate[:num_pts], (init_dist.config.NUM_TRIG, 1)), torch.tile(self.__coordinate[num_pts:], (init_dist.config.NUM_TRIG, 1))]) # central pts of all elements, then extra pts of all elements
		self.__tril_element_indices = torch.tensor(init_dist.config.TRIL_ELEMENT_INDICES)
		self.__index = torch.cat([torch.repeat_interleave(self.__tril_element_indices, num_pts), torch.repeat_interleave(self.__tril_element_indices, num_pts * extra_ratio)])

	@property
	def num_center(self) -> list[int]:
		r"""To calculate the number of points in the center that corresponding to each element

		Returns
		-------
		torch.Tensor
			Central points cooresponding to each element
		"""
		return (self.__index[:self.__num_all_centers] == self.__tril_element_indices[:, torch.newaxis]).to(torch.int).sum(-1).tolist()

	@property
	def center(self) -> list[torch.Tensor]:
		r"""To give the phase space coordinates of each density matrix element

		Returns
		-------
		list[torch.Tensor]
			Coordinates of each density matrix element
		"""
		return [self.__coordinate[self.__index == iElement] for iElement in self.__config.TRIL_ELEMENT_INDICES]

	@property
	def density(self) -> list[torch.Tensor]:
		r"""To give the density matrix element

		Returns
		-------
		list[torch.Tensor]
			Density matrix element, cooresponding to `center()`
		"""
		return [self.__density[self.__index == iElement] for iElement in self.__config.TRIL_ELEMENT_INDICES]

	@property
	def rescale_factor(self) -> torch.Tensor:
		r"""To give the rescale factor that makes `max|rho|==1`

		Returns
		-------
		torch.Tensor, dtype of `torch.double, shape of (NUM_ELM,)
			Rescale factor that scale up to 1.0; if all samples are 0, return 0
		"""
		result: torch.Tensor = torch.empty((self.__config.NUM_PES, self.__config.NUM_PES))
		for iPES, jPES, iElement in zip(self.__config.TRIL_ROW_INDICES, self.__config.TRIL_COL_INDICES, self.__config.TRIL_ELEMENT_INDICES):
			element_density: torch.Tensor = self.__density[self.__index == iElement]
			if iPES == jPES:
				result[iPES, jPES] = 0 if torch.all(element_density == 0.0) else 1.0 / torch.max(torch.abs(element_density))
			else:
				result[iPES, jPES] = 0 if torch.all(element_density.imag == 0.0) else 1.0 / torch.max(torch.abs(element_density.imag))
				result[jPES, iPES] = 0 if torch.all(element_density.real == 0.0) else 1.0 / torch.max(torch.abs(element_density.real))
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
			np.repeat(np.concatenate([[idx, idx + self.__config.NUM_ELM] for idx in self.__config.TRIL_ELEMENT_INDICES]), np.concatenate([[m, n - m] for m, n in zip(self.num_center, (self.__index == self.__tril_element_indices[:, torch.newaxis]).detach().cpu().numpy().astype(np.int_).sum(-1))]))[np.newaxis],
			footer="\n",
			comments="",
			encoding=constant.ENC
		)

	def evolve(
		self,
		model: pes.Potential,
		mass: torch.Tensor,
		dt: float,
		predictor: constant.Predictor
	) -> None:
		r"""Non-adiabatic dynamics with surface hopping

		Parameters
		----------
		model : pes.Potential
			Quantities derived from potential
		mass : torch.Tensor, shape of (DIM,)
			Mass of classical degree of freedom
		dt : float
			Time interval
		predictor : pes.Predictor
			It predicts the density matrix element based on given coordinates and element index
		purity : torch.Tensor, shape of (NUM_PES, NUM_PES)
			The purity of each element, indicating the transition allowance to other elements
		"""
		# evolve coordinates and hopping
		x0: typing.Final[torch.Tensor] = self.__coordinate[:, :self.__config.DIM] # N * D, slice of centers
		p0: typing.Final[torch.Tensor] = self.__coordinate[:, self.__config.DIM:] # N * D, slice of centers
		x2: typing.Final[torch.Tensor] = torch.empty_like(x0) # N * D
		p1: typing.Final[torch.Tensor] = torch.empty_like(p0) # N * D
		x4: typing.Final[torch.Tensor] = torch.empty_like(x0) # N * D
		p2: typing.Final[torch.Tensor] = torch.empty_like(p0) # N * D
		current_indices: typing.Final[torch.Tensor] = self.__index == self.__tril_element_indices[:, torch.newaxis]
		for iTrig, (iBelongPES, jBelongPES) in enumerate(zip(self.__config.TRIL_ROW_INDICES, self.__config.TRIL_COL_INDICES)):
			indices: torch.Tensor = torch.where(current_indices[iTrig])[0] # NUM_PTS
			num_pts: int = indices.numel()
			if num_pts > 0: # only have elements
				# get x and p, and 2 semi adiabatic steps for all points
				x2[indices, :], p1[indices, :] = evolve.evolve_coordinates_adiabatically(model, x0[indices, :], p0[indices, :], mass, dt / 2.0, evolve.Direction.FORWARD, iBelongPES, jBelongPES)
				x4[indices, :], p2[indices, :] = evolve.evolve_coordinates_adiabatically(model, x2[indices, :], p1[indices, :], mass, dt / 2.0, evolve.Direction.FORWARD, iBelongPES, jBelongPES)
				# surface hopping
				# choose the one to jump to
				idx_of_dest_idx: torch.Tensor = torch.randint(0, 2 * self.__config.NUM_PES - 2, (num_pts,)) # n
				velocity: torch.Tensor = p2[indices, :] / mass # n * D
				potential, couplings = model.adiabatic_potential_and_coupling(x4[indices, :]) # n * N, n * D * N * N
				coupling: torch.Tensor = couplings[torch.arange(num_pts), :, self.__coup_row_idx[iTrig, idx_of_dest_idx], self.__coup_col_idx[iTrig, idx_of_dest_idx]] # n * D
				transition_rate: torch.Tensor = torch.abs(torch.sum(velocity * coupling, -1) * dt) # n
				transition_prob: torch.Tensor = transition_rate / (1.0 + transition_rate) # n
				# energy conservation
				momentum_rescale_factor_sq: torch.Tensor = 1.0\
					+ (potential[torch.arange(num_pts), self.__coup_col_idx[iTrig, idx_of_dest_idx]] - potential[torch.arange(num_pts), self.__coup_row_idx[iTrig, idx_of_dest_idx]]) / torch.sum(p2[indices, :] ** 2 / mass, -1) # n
				# judgment
				transition: torch.Tensor = torch.logical_and(torch.rand(num_pts) < transition_prob, momentum_rescale_factor_sq >= 0.0) # n
				# change index, momentum, and the back propagation
				if torch.any(transition).item():
					transit_indices: torch.Tensor = indices[transition] # indices[transition] equivalent to [current_indices[iTrig]][transition]
					transit_idx_of_dest_idx: torch.Tensor = idx_of_dest_idx[transition]
					self.__index[transit_indices] = self.__dest_idx[iTrig, transit_idx_of_dest_idx]
					p2[transit_indices, :] *= torch.sqrt(momentum_rescale_factor_sq[transition]).reshape(-1, 1)
					# change density
					for iPredictPES, jPredictPES, iPredictElement in zip(self.__config.TRIL_ROW_INDICES, self.__config.TRIL_COL_INDICES, self.__config.TRIL_ELEMENT_INDICES):
						if iBelongPES == iPredictPES and jBelongPES == jPredictPES:
							pass
						need_predict_idx: torch.Tensor = transit_indices[self.__dest_idx[iTrig, transit_idx_of_dest_idx] == iPredictElement] # in case the transition happens to this element
						if need_predict_idx.numel() > 0:
							x2[need_predict_idx, :], p1[need_predict_idx, :] = evolve.evolve_coordinates_adiabatically(model, x4[need_predict_idx, :], p2[need_predict_idx, :], mass, dt / 2.0, evolve.Direction.BACKWARD, iPredictPES, jPredictPES)
							x0[need_predict_idx, :], p0[need_predict_idx, :] = evolve.evolve_coordinates_adiabatically(model, x2[need_predict_idx, :], p1[need_predict_idx, :], mass, dt / 2.0, evolve.Direction.BACKWARD, iPredictPES, jPredictPES)
							self.__density[need_predict_idx] = predictor(self.__coordinate[need_predict_idx, :], iPredictElement)
		# evolve density
		for iPES, jPES, iElement in zip(self.__config.TRIL_ROW_INDICES, self.__config.TRIL_COL_INDICES, self.__config.TRIL_ELEMENT_INDICES):
			filters: torch.Tensor = self.__index == iElement
			if torch.count_nonzero(filters) > 0:
				self.__density[filters] = evolve.evolve_density_non_adiabatically(
					model,
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
		self.__coordinate[:, :self.__config.DIM] = x4
		self.__coordinate[:, self.__config.DIM:] = p2
