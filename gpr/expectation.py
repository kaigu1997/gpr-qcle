r"""expectation
===========
This module evaluates the expectation values (population, <x> and <p>, energy, etc)
"""
import abc
import collections.abc
import math
import typing

import numpy as np
import torch
import torch_kmeans

import constant
import evolve
import gp
import pes
import plot

torch.set_default_dtype(constant.DTYPE)
torch.set_default_device(constant.DEVICE)
torch.manual_seed(constant.SEED)


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



class Averager(abc.ABC):
	r"""To calculate averages

	Parameters
	----------
	config : pes.ModelConfig
		Configuration of the model

	Attributes
	----------
	config : pes.ModelConfig
		Configuration of the model
	PURITY_FACTOR: float
		Pre-factor when calculating purity

	Methods
	----------
	population()
		To calculate the population on each potential energy surfaces
	coordinates()
		To calculate the average positions and momenta
	potential()
		To calculate the average potential energy
	kinetic()
		To calculate the average kinetic energy
	purity()
		To calculate contribution to purity of each element
	"""
	__slots__: typing.ClassVar[tuple] = ("config", "PURITY_FACTOR")
	config: typing.Final[pes.ModelConfig]
	PURITY_FACTOR: typing.Final[float]

	def __init__(self, config: pes.ModelConfig):
		self.config = config
		self.PURITY_FACTOR = (2.0 * math.pi * constant.HBAR) ** config.DIM

	@abc.abstractmethod
	def population(self) -> torch.Tensor:
		r"""To calculate the population on each potential energy surfaces

		Returns
		-------
		torch.Tensor
			Population on each surfaces
		"""

	@abc.abstractmethod
	def coordinates(self) -> torch.Tensor:
		r"""To calculate the average of phase space coordinates

		Returns
		-------
		torch.Tensor
			Average positions and momenta
		"""

	@abc.abstractmethod
	def square_coordinates(self) -> torch.Tensor:
		r"""To calculate averages of product of phase space coordinates

		Returns
		-------
		torch.Tensor
			A PHASEDIM-by-PHASEDIM matrix, whose ij term is <x_i*x_j>
		"""

	def covariance(self) -> torch.Tensor:
		r"""To calculate the covariance between phase space coordinates

		Returns
		-------
		torch.Tensor
			PHASEDIM-by-PHASEDIM covariance matrix
		"""
		coord_ave: typing.Final[torch.Tensor] = self.coordinates()
		return self.square_coordinates() + (self.population().sum() - 2.0) * coord_ave[:, torch.newaxis] * coord_ave[torch.newaxis, :]

	def standard_deviation(self) -> torch.Tensor:
		r"""To calculate the standard deviation of each phase space dimension

		Returns
		-------
		torch.Tensor
			Standard deviation of each dimension
		"""
		return torch.sqrt(torch.diagonal(self.covariance()))

	@abc.abstractmethod
	def potential(self, model: pes.Potential) -> float:
		r"""To calculate the average potential energy

		Parameters
		----------
		model : pes.Potential
			Quantities derived from potential

		Returns
		-------
		float
			Average potential energy
		"""

	def kinetic(self, mass: torch.Tensor) -> float:
		r"""To calculate the average kinetic energy

		Parameters
		----------
		mass : torch.Tensor
			Mass of classical degree of freedom

		Returns
		-------
		float
			Average kinetic energy
		"""
		return (self.square_coordinates()[self.config.P_DIM_RANGE, self.config.P_DIM_RANGE] / mass).sum().item() / 2.0

	@abc.abstractmethod
	def purity(self) -> torch.Tensor:
		r"""To calculate contribution to purity of each element

		Returns
		-------
		torch.Tensor
			Purity of each element
		"""

class MonteCarloAverage(Averager):
	r"""Using Monte Carlo to estimate averages

	Parameters
	----------
	config : pes.ModelConfig
		Configuration of the model
	num_pts : int
		The number of points

	Attributes
	----------
	DIAGONAL_TRIL_INDEX : torch.Tensor
		Lower-triangular index of diagonal elements
	point_set : torch.Tensor, double, shape of (NUM_TRIG, NUM_PTS, PHASEDIM)
		Sample points of each density matrix element for Monte Carlo
	weight : torch.Tensor, double, shape of (NUM_TRIG, NUM_PTS)
		Sample weight of each point
	density : torch.Tensor, complex double, shape of (NUM_TRIG, NUM_PTS)
		Density Matrix element of each point

	Methods
	-------
	update_pts(ref_pts, predictor): To predict density at next time step
	"""
	__slots__: tuple = ("DIAGONAL_TRIL_INDEX", "__num_pts", "point_set", "weight", "density")
	DIAGONAL_TRIL_INDEX: typing.Final[tuple[int, ...]]
	__num_pts: typing.Final[int]
	point_set: torch.Tensor
	weight: torch.Tensor
	density: torch.Tensor

	def __init__(self, config: pes.ModelConfig, num_pts: int):
		super().__init__(config)
		self.DIAGONAL_TRIL_INDEX = tuple(self.config.FLATTEN_TRIL_INDEX[i * (self.config.NUM_PES + 1)] for i in self.config.PES_RANGE)
		self.__num_pts = num_pts
		self.point_set = torch.empty((self.config.NUM_TRIG, self.__num_pts, self.config.PHASEDIM))
		self.weight = torch.empty((self.config.NUM_TRIG, self.__num_pts))
		self.density = torch.empty((self.config.NUM_TRIG, self.__num_pts), dtype=torch.cdouble)

	def _gaussian_weight(self, coord: torch.Tensor, center: torch.Tensor, stddev: torch.Tensor) -> torch.Tensor:
		r"""The multi-dimensional gaussian function

		Parameters
		----------
		coord : torch.Tensor, shape of (..., N)
			The coordinates whose weights are calculated
		center : torch.Tensor, shape of (N,)
			Centers of the normal distributions
		stddev : torch.Tensor, shape of (N,)
			Standard deviation of the normal distributions.
			This function assumes no correlation

		Returns
		-------
		torch.Tensor, shape of (..., N)
			Density at the given coordinates
		"""
		return torch.exp(-torch.sum(((coord - center) / stddev) ** 2, -1) / 2.0) / ((2.0 * math.pi) ** self.config.DIM * stddev.prod())

	@property
	def num_pts(self) -> int:
		r"""_summary_

		Returns
		-------
		int
			_description_
		"""
		return self.__num_pts

	def update_pts(
		self,
		ref_pts: list[torch.Tensor],
		predictor: constant.Predictor
	) -> None:
		r"""To update the point set used

		Parameters
		----------
		ref_pts : list[torch.Tensor], len of NUM_TRIG, each of shape (NUM_PTS, PHASEDIM)
			Current points, used to estimate average and variance
		predictor : self.config.Predictor
			Used to predict the density of the points
		"""
		for iPES, jPES, iTrig in zip(self.config.TRIL_ROW_INDICES, self.config.TRIL_COL_INDICES, self.config.TRIG_RANGE):
			center: torch.Tensor = torch.mean(ref_pts[iTrig], 0)
			stddev: torch.Tensor = 1.5 * torch.std(ref_pts[iTrig], 0)
			self.point_set[iTrig] = normal_sample(self.__num_pts, center, stddev)
			self.weight[iTrig] = self._gaussian_weight(self.point_set[iTrig], center, stddev) # N
			self.density[iTrig] = predictor(self.point_set[iTrig], iPES * self.config.NUM_PES + jPES)

	def population(self) -> torch.Tensor:
		return torch.mean(self.density[self.DIAGONAL_TRIL_INDEX, ...].real / self.weight[self.DIAGONAL_TRIL_INDEX, ...], -1)

	def coordinates(self) -> torch.Tensor:
		return torch.sum(
			torch.mean(
				torch.moveaxis(self.point_set[self.DIAGONAL_TRIL_INDEX, ...], -1, 0)
				* self.density[self.DIAGONAL_TRIL_INDEX, ...].real
				/ self.weight[self.DIAGONAL_TRIL_INDEX, ...],
				-1
			),
			1
		)

	def square_coordinates(self) -> torch.Tensor:
		return torch.sum(
			torch.mean(
				self.density[self.DIAGONAL_TRIL_INDEX, :, torch.newaxis, torch.newaxis].real
				* self.point_set[self.DIAGONAL_TRIL_INDEX, :, :, torch.newaxis]
				* self.point_set[self.DIAGONAL_TRIL_INDEX, :, torch.newaxis, :]
				/ self.weight[self.DIAGONAL_TRIL_INDEX, :, torch.newaxis, torch.newaxis],
				1
			),
			0
		)

	def covariance(self) -> torch.Tensor:
		coord_ave: typing.Final[torch.Tensor] = self.coordinates()
		return torch.sum(
			torch.mean(
				self.density[self.DIAGONAL_TRIL_INDEX, :, torch.newaxis, torch.newaxis].real
				* (self.point_set[self.DIAGONAL_TRIL_INDEX, :, :, torch.newaxis] - coord_ave[:, torch.newaxis])
				* (self.point_set[self.DIAGONAL_TRIL_INDEX, :, torch.newaxis, :] - coord_ave[torch.newaxis, :])
				/ self.weight[self.DIAGONAL_TRIL_INDEX, :, torch.newaxis, torch.newaxis],
				1
			),
			0
		)

	def potential(self, model: pes.Potential) -> float:
		return torch.sum(torch.mean(
			model.adiabatic_potential(self.point_set[self.DIAGONAL_TRIL_INDEX, :, :self.config.DIM])[self.config.PES_RANGE, :, self.config.PES_RANGE]
			* self.density[self.DIAGONAL_TRIL_INDEX, ...].real
			/ self.weight[self.DIAGONAL_TRIL_INDEX, ...],
			-1)
		).item()

	def purity(self) -> torch.Tensor:
		flatten_lower_trig: torch.Tensor = torch.zeros(self.config.NUM_PES, self.config.NUM_PES, dtype=torch.double)
		flatten_lower_trig[self.config.TRIL_ROW_INDICES, self.config.TRIL_COL_INDICES] = torch.mean((self.density.real ** 2 + self.density.imag ** 2) / self.weight, -1)
		return self.PURITY_FACTOR * (flatten_lower_trig + torch.tril(flatten_lower_trig, -1).T)


class EvolvingPointsMCAverage(MonteCarloAverage):
	r"""To calculate average by Monte Carlo estimate too,
	but using points evolving forward with same weights

	Parameters
	----------
	config : pes.ModelConfig
		Configuration of the model
	num_pts : int
		The number of points for monte carlo
	init_dist : pes.InitialDistribution
		Initial distribution to generate points, density, and weights

	Methods
	-------
	evolve(mass, dt, predictor)
		To evolve the coordinates, and density if applicable
	update_density(predictor)
		To update the density using the predictor if the density is not evolved
	"""
	__slots__: tuple = ()

	def __init__(
		self,
		config: pes.ModelConfig,
		num_pts: int,
		init_dist: pes.InitialDistribution,
		stddev: torch.Tensor | None = None
	):
		super().__init__(config, num_pts)
		if stddev is None:
			stddev = init_dist.sigma_r0
		pts: typing.Final[torch.Tensor] = normal_sample(num_pts, init_dist.r0, stddev)
		den: typing.Final[torch.Tensor] = init_dist(pts)
		self.point_set[:] = pts
		self.density = den[:, self.config.TRIL_ROW_INDICES, self.config.TRIL_COL_INDICES].T
		self.weight[:] = self._gaussian_weight(pts, init_dist.r0, stddev)

	def evolve(
		self,
		model: pes.Potential,
		mass: torch.Tensor,
		dt: float,
		predictor: constant.Predictor
	) -> None:
		r"""To evolve the coordinates and density

		Parameters
		----------
		model : pes.Potential
			Quantities derived from potential
		mass : torch.Tensor, shape of (DIM,)
			Mass of classical degree of freedom
		dt : float
			Time interval
		predictor : constant.Predictor
			It predicts the density matrix element based on given coordinates and element index
		"""
		evolve.evolve(model, [*self.point_set], [*self.density], mass, dt, predictor)


@typing.final
class GPRPredictors(Averager):
	r"""Combination of single predictors, and also serves as the analytical averager

	Parameters
	----------
	config : pes.ModelConfig
		Configuration of the model
	num_pts : int
		The number of points for monte carlo and the whole set (`x_all` and `y_all`)
	init_dist : pes.InitialDistribution
		Initial distribution to generate points, density, and weights
	init_stddev : torch.Tensor
		The standard deviation of the initial points, used to generate the initial point set
	num_ind : int
		The number of inducing points
	kernel_initial_value : torch.Tensor
		The initial value of lengthscale for all predictors
	model : pes.Potential
		Quantities derived from potential
	mass : torch.Tensor, shape of (DIM,)
		Mass of classical degree of freedom

	Attributes
	----------
	epmca : EvolvingPointsMCAverage
		The evolving points monte carlo average, used to generate the point set, density, and weights.
		Notice it is evolving, but its built-in `evolve` method is not used.

	Methods
	-------
	inducing_points()
		To get all inducing points
	scale()
		The rescaling factor of each predictor
	predict(x_input, ElementIndex, num_dt)
		To predict test targets based on input and corresponding density matrix element
	get_marginal(dimensions, x_input, ElementIndex, num_dt)
		To get the marginal distribution of current gaussian process regressions
	update(x_all, y_all, num_ind, num_pt)
		To update the training inputs and targets, as well as the rescale factor
	train()
		To train each predictor
	print(f)
		To print hyperparameters to file
	"""
	@staticmethod
	def __check_predictor(predictor: gp.SinglePredictor) -> bool:
		r"""To check if the predictor could be used for training / predicting

		If no label is given, or all the labels are 0, training / predicting is not needed.

		Parameters
		----------
		predictor : SinglePredictor
			The predictor

		Returns
		-------
		bool
			Availability of training / predicting
		"""
		return not torch.all(predictor.y_all == 0).item()

	drc: typing.Final = evolve.Direction.FORWARD
	__JUDGE_INCLUDE_THRESHOLD: typing.Final = 0.1
	__JUDGE_REGISTER_THRESHOLD: typing.Final = 0.1
	__slots__: typing.Final[tuple] = ("__AVERAGE_CONSTANT", "epmca", "__kmeans", "__saved_predictors", "chunk_size")
	__AVERAGE_CONSTANT: typing.Final[float]
	epmca: typing.Final[EvolvingPointsMCAverage]
	__kmeans: typing.Final[torch_kmeans.KMeans]
	__saved_predictors: typing.Final[tuple[list[gp.SinglePredictor], ...]]
	chunk_size: typing.Final[int]

	def __init__(
		self,
		config: pes.ModelConfig,
		num_pts: int,
		init_dist: pes.InitialDistribution,
		init_stddev: torch.Tensor,
		num_ind: int,
		kernel_initial_value: torch.Tensor,
	):
		super().__init__(config)
		self.__AVERAGE_CONSTANT = (2.0 * math.pi) ** self.config.DIM
		self.epmca = EvolvingPointsMCAverage(config, num_pts, init_dist, init_stddev)
		self.__kmeans = torch_kmeans.KMeans(init_method="k-means++", n_clusters=num_ind, seed=constant.SEED, verbose=constant.DEBUG_MODE)
		ind_pt: typing.Final[torch.Tensor] = self.__kmeans(self.epmca.point_set[:1]).centers[0]
		self.__saved_predictors = tuple([] for _ in config.ELEMENT_RANGE)
		for iElement in config.ELEMENT_RANGE:
			print("\tInitial Training " + plot.get_RI_label(iElement, config.NUM_PES))
			TrilIndex: int = self.config.FLATTEN_TRIL_INDEX[iElement]
			y: torch.Tensor = self.epmca.density[TrilIndex].real if iElement // config.NUM_PES <= iElement % config.NUM_PES else self.epmca.density[TrilIndex].imag
			self.__saved_predictors[iElement].append(gp.ResidualPredictor(
				ind_pt,
				self.epmca.point_set[TrilIndex],
				y,
				kernel_initial_value
			)) # this includes training of initial distribution
		# then check for chunk size to avoid OOM in autograd
		self.chunk_size = self.__saved_predictors[0][0].get_chunk_size(normal_sample(num_ind, init_dist.r0, init_dist.sigma_r0))

	def population(self) -> torch.Tensor:
		return torch.stack([sum((pred.lengthscale.prod() * pred.k_inv_y.sum() for pred in self.__saved_predictors[iPES * self.config.NUM_PES + iPES]), start=torch.tensor(0.)) for iPES in self.config.PES_RANGE]) * self.__AVERAGE_CONSTANT

	def coordinates(self) -> torch.Tensor:
		return sum((pred.lengthscale.prod() * (pred.k_inv_y[:, None] * pred.x_ind).sum(0) for iPES in self.config.PES_RANGE for pred in self.__saved_predictors[iPES * self.config.NUM_PES + iPES]), start=torch.zeros(self.config.PHASEDIM)) * self.__AVERAGE_CONSTANT

	def square_coordinates(self) -> torch.Tensor:
		return sum((pred.lengthscale.prod() * ((pred.k_inv_y[:, None, None] * pred.x_ind[:, :, None] * pred.x_ind[:, None, :]).sum(0) + pred.k_inv_y.sum() * torch.diagflat(pred.lengthscale ** 2)) for iPES in self.config.PES_RANGE for pred in self.__saved_predictors[iPES * self.config.NUM_PES + iPES]), start=torch.zeros((self.config.PHASEDIM, self.config.PHASEDIM))) * self.__AVERAGE_CONSTANT

	def covariance(self) -> torch.Tensor:
		return super().covariance()

	def potential(self, model: pes.Potential) -> float:
		return math.nan

	def purity(self) -> torch.Tensor:
		def purity_from_preds(pred_left: gp.SinglePredictor, pred_right: gp.SinglePredictor) -> float:
			r"""To calculate the contribution to purity from two predictors

			Parameters
			----------
			pred_left : gp.SinglePredictor
				The first predictor
			pred_right : gp.SinglePredictor
				The second predictor

			Returns
			-------
			float
				The contribution to purity from the two predictors
			"""
			length_ij: typing.Final[torch.Tensor] = (pred_left.lengthscale ** 2 + pred_right.lengthscale ** 2).sqrt()
			return pred_left.lengthscale.prod().item() * pred_right.lengthscale.prod().item() / length_ij.prod().item() * (pred_left.k_inv_y @ gp.rbf(length_ij, pred_left.x_ind, pred_right.x_ind) @ pred_right.k_inv_y).item()

		result: torch.Tensor = torch.empty(self.config.NUM_PES, self.config.NUM_PES)
		for iPES in range(self.config.NUM_PES):
			for jPES in range(self.config.NUM_PES):
				ElementIndex: int = iPES * self.config.NUM_PES + jPES
				result[iPES, jPES] = sum(purity_from_preds(pred_left, pred_right) for pred_left in self.__saved_predictors[ElementIndex] for pred_right in self.__saved_predictors[ElementIndex])
		return self.PURITY_FACTOR * (2.0 * math.pi) ** self.config.DIM * (result + result.T - torch.diag(torch.diag(result)))

	def __getitem__(self, ElementIndex: int) -> gp.SinglePredictor:
		r"""To get corresponding predictor

		Parameters
		----------
		ElementIndex : int
			Index of the predictor

		Returns
		-------
		SinglePredictor
			Predictor corresponding to the index in density supervector
		"""
		assert 0 <= ElementIndex < self.config.NUM_ELM
		return self.__saved_predictors[ElementIndex][-1]

	@property
	def inducing_points(self) -> torch.Tensor:
		r"""To get all inducing points

		Returns
		-------
		torch.Tensor, shape of (NUM_TRIG, num_ind, PHASEDIM)
			Inducing points of independent, lower triangular elements
		"""
		return torch.stack([self.__saved_predictors[iTrig][-1].x_ind for iTrig in self.config.TRIL_ELEMENT_INDICES], 0)

	@property
	def scale(self) -> torch.Tensor:
		r"""To get the rescaling factor for each element, which is the inverse of the maximum density of the points

		Returns
		-------
		torch.Tensor, shape of (NUM_ELM,)
			The rescaling factor
		"""
		result: torch.Tensor = torch.empty(self.config.NUM_ELM)
		for iTrig, iPES, jPES, iElement in zip(self.config.TRIG_RANGE, self.config.TRIL_ROW_INDICES, self.config.TRIL_COL_INDICES, self.config.TRIL_ELEMENT_INDICES):
			if iPES == jPES:
				result[iElement] = 1.0 / self.epmca.density[iTrig].real.abs().max()
			else:
				result[iElement] = 1.0 / self.epmca.density[iTrig].imag.abs().max()
				result[jPES * self.config.NUM_PES + iPES] = 1.0 / self.epmca.density[iTrig].real.abs().max()
		return result

	@property
	def residual_scale(self) -> torch.Tensor:
		r"""The rescaling factor of each predictor

		Returns
		-------
		torch.Tensor, shape of (NUM_ELM,)
			The rescaling factor
		"""
		return torch.tensor([pred[-1].scale for pred in self.__saved_predictors])

	def __combine_to_complex[**P](
		self,
		x_input: torch.Tensor,
		RowIndex: int,
		ColIndex: int,
		call_single_predictor: collections.abc.Callable[typing.Concatenate[gp.SinglePredictor, torch.Tensor, P], collections.abc.Iterable[torch.Tensor]],
		*args: P.args,
		**kwargs: P.kwargs
	) -> tuple[torch.Tensor, ...]:
		r"""To combine results from single predictor into complex arrays

		Parameters
		----------
		x_input : torch.Tensor, shape of (..., PHASEDIM)
			Test inputs
		RowIndex : int
			Index of row of the element in density matrix
		ColIndex : int
			Index of column of the element in density matrix
		call_single_predictor : collections.abc.Callable[typing.Concatenate[SinglePredictor, torch.Tensor, P], collections.abc.Iterable[torch.Tensor]]
			The function that takes the single predictor and generates some Tensor (prediction, derivatives, marginals, etc)

		Returns
		-------
		tuple[torch.Tensor, ...]
			Combined complex arrays from single predictor
		"""
		assert 0 <= RowIndex < self.config.NUM_PES and 0 <= ColIndex < self.config.NUM_PES
		x_test: typing.Final[torch.Tensor] = x_input.reshape(-1, x_input.shape[-1])
		pred_call: typing.Final[collections.abc.Callable[[gp.SinglePredictor], collections.abc.Iterable[torch.Tensor]]] = lambda pred: call_single_predictor(pred, x_test, *args, **kwargs)
		if RowIndex == ColIndex:
			return tuple(sum(items, start=torch.zeros_like(items[0])).reshape(x_input.shape[:-1] + items[0].shape[1:]) + 0.j for items in zip(*map(pred_call, self.__saved_predictors[RowIndex * self.config.NUM_PES + ColIndex])))
		elif RowIndex > ColIndex:
			return tuple((sum(real, start=torch.zeros_like(real[0])) + 1.j * sum(imag, start=torch.zeros_like(imag[0]))).reshape(x_input.shape[:-1] + real[0].shape[1:]) for real, imag in zip(zip(*map(pred_call, self.__saved_predictors[ColIndex * self.config.NUM_PES + RowIndex])), zip(*map(pred_call, self.__saved_predictors[RowIndex * self.config.NUM_PES + ColIndex]))))
		else: # RowIndex < ColIndex
			return tuple((sum(real, start=torch.zeros_like(real[0])) - 1.j * sum(imag, start=torch.zeros_like(imag[0]))).reshape(x_input.shape[:-1] + real[0].shape[1:]) for real, imag in zip(zip(*map(pred_call, self.__saved_predictors[RowIndex * self.config.NUM_PES + ColIndex])), zip(*map(pred_call, self.__saved_predictors[ColIndex * self.config.NUM_PES + RowIndex]))))

	def predict(
		self,
		x_input: torch.Tensor,
		ElementIndex: int
	) -> torch.Tensor:
		r"""To predict test targets based on input and corresponding density matrix element

		Parameters
		----------
		x_input : torch.Tensor, shape of (..., PHASEDIM)
			Test inputs
		ElementIndex : int
			Index of the element
		num_dt : int
			The number of time steps since epoch

		Returns
		-------
		torch.Tensor, shape of (...)
			Density of the element of all test inputs
		"""
		return self.__combine_to_complex(
			x_input,
			ElementIndex // self.config.NUM_PES,
			ElementIndex % self.config.NUM_PES,
			lambda pred, x_test: (pred.predict(x_test),) if GPRPredictors.__check_predictor(pred) else (torch.zeros(x_test.shape[0]),)
		)[0]

	def get_marginal(
		self,
		x_input: torch.Tensor,
		ElementIndex: int,
		dimensions: int | collections.abc.Iterable[int]
	) -> torch.Tensor:
		r"""To get the marginal distribution of current gaussian process regressions

		Parameters
		----------
		dimensions : int | collections.abc.Iterable[int]
			The dimensions to be kept
		x_input : torch.Tensor, shape of (..., len(dimensions))
			Test inputs
		ElementIndex : int
			Index of the element
		num_dt : int
			The number of time steps since epoch

		Returns
		-------
		torch.Tensor, shape of (...)
			Marginal distribution on the inputs
		"""
		def call_single_predictor(pred: gp.SinglePredictor, x_test: torch.Tensor, dims: collections.abc.Sequence[int]) -> tuple[torch.Tensor]:
			r"""To do prediction of a single predictor

			Parameters
			----------
			pred : SinglePredictor
				The predictor
			x_test : torch.Tensor, shape of (N, len(dimensions))
				Test inputs
			dims : collections.abc.Sequence[int]
				The dims to be kept

			Returns
			-------
			torch.Tensor, shape of (N,)
				Test targets by the predictor
			"""
			if GPRPredictors.__check_predictor(pred):
				return (pred.get_marginal(x_test, dims),)
			else:
				return (torch.zeros(x_test.shape[0]),)

		if isinstance(dimensions, int):
			dimensions = [dimensions]
		else:
			dimensions = tuple(set(dimensions)) # remove duplicate
		assert all(0 <= dim <= self.config.PHASEDIM for dim in dimensions)
		assert x_input.shape[-1] == len(dimensions)
		return self.__combine_to_complex(
			x_input,
			ElementIndex // self.config.NUM_PES,
			ElementIndex % self.config.NUM_PES,
			call_single_predictor,
			dimensions
		)[0]

	def evolve_update(
		self,
		model: pes.Potential,
		mass: torch.Tensor,
		dt: float,
	) -> None:
		r"""To evolve the coordinates and density

		Parameters
		----------
		model : pes.Potential
			Quantities derived from potential
		mass : torch.Tensor, shape of (DIM,)
			Mass of classical degree of freedom
		dt : float
			Time interval
		predictor : constant.Predictor
			It predicts the density matrix element based on given coordinates and element index
		"""
		def coord_derivative(x: torch.Tensor, RowIndex: int, ColIndex: int) -> torch.Tensor:
			r"""To calculate dX/dt and drho_ij/dt

			Parameters
			----------
			x : torch.Tensor, shape of (..., PHASEDIM)
				Phase space coordinates
			RowIndex : int
				Index of row of the element in density matrix
			ColIndex : int
				Index of column of the element in density matrix

			Returns
			-------
			torch.Tensor
				dX/dt, shape of (..., PHASEDIM)
			"""
			r: torch.Tensor = x[..., :self.config.DIM] # position coordinates
			v: torch.Tensor = x[..., self.config.DIM:] / mass # velocity
			F = model.force(r)
			return torch.cat((v, (F[..., RowIndex, RowIndex] + F[..., ColIndex, ColIndex]) / 2.0), -1)

		def density_derivative(
			x: torch.Tensor,
			dXdt: torch.Tensor,
			pred_real: gp.SinglePredictor | None,
			pred_imag: gp.SinglePredictor | None,
			RowIndex: int,
			ColIndex: int
		) -> torch.Tensor:
			r"""To calculate dX/dt and drho_ij/dt

			Parameters
			----------
			x : torch.Tensor, shape of (..., PHASEDIM)
				Phase space coordinates
			dXdt : torch.Tensor, shape of (..., PHASEDIM)
				Time derivative of phase space coordinates
			pred_real : SinglePredictor | None
				Predictor for the real part of the density matrix element; if None, the element will be predicted by residual predictors, and the time derivative will be non-adiabatic. Otherwise, the time derivative will be adiabatic, and the predictor will be used to calculate the time derivative directly.
			pred_imag : SinglePredictor | None
				Predictor for the imaginary part of the density matrix element, if the element is off-diagonal. If the element is diagonal, this should be None.
			RowIndex : int
				Index of row of the element in density matrix
			ColIndex : int
				Index of column of the element in density matrix

			Returns
			-------
			torch.Tensor
				drho_ij/dt, shape of (...)
			"""
			def predict_derivative_over_input_all(x_input: torch.Tensor, RowIndex: int, ColIndex: int) -> gp.SinglePredictor.InputDerivativeReturn:
				r"""To combine the prediction and its derivative of the single predictor into complex arrays

				Parameters
				----------
				x_input : torch.Tensor, shape of (N, PHASEDIM)
					Test inputs
				RowIndex : int
					Index of row of the element in density matrix
				ColIndex : int
					Index of column of the element in density matrix

				Returns
				-------
				gp.SinglePredictor.InputDerivativeReturn
					The prediction, and the derivative of the prediction over input
				"""
				return gp.SinglePredictor.InputDerivativeReturn(*self.__combine_to_complex(
					x_input,
					RowIndex,
					ColIndex,
					lambda pred, x_test: pred.predict_derivative_over_input(x_test)
				))

			r: torch.Tensor = x[..., :self.config.DIM] # position coordinates
			v: torch.Tensor = x[..., self.config.DIM:] / mass # velocity
			E, D = model.adiabatic_potential_and_coupling(r)
			if pred_real is not None:
				pred, grad_input = pred_real.predict_derivative_over_input(x)
				if pred_imag is not None:
					pred_im, grad_input_im = pred_imag.predict_derivative_over_input(x)
					pred = pred + 1.j * pred_im
					grad_input = grad_input + 1.j * grad_input_im
				else:
					pred = pred + 0.j
					grad_input = grad_input + 0.j
			else:
				pred, grad_input = predict_derivative_over_input_all(x, RowIndex, ColIndex)
			# drho/dt at x_test
			time_deriv: torch.Tensor = -(dXdt * grad_input).sum(-1)
			if RowIndex != ColIndex:
				time_deriv -= 1.0j / constant.HBAR * (E[..., RowIndex] - E[..., ColIndex]) * pred
			if pred_real is None:
				for iPES in self.config.PES_RANGE:
					if iPES != RowIndex:
						pred_kj, grad_kj = predict_derivative_over_input_all(x, iPES, ColIndex)
						time_deriv -= (D[..., RowIndex, iPES] * (v * pred_kj[..., np.newaxis] + (E[..., RowIndex] - E[..., iPES])[..., np.newaxis] / 2.0 * grad_kj[..., self.config.DIM:])).sum(-1)
					if iPES != ColIndex:
						pred_ik, grad_ik = predict_derivative_over_input_all(x, RowIndex, iPES)
						time_deriv += (D[..., iPES, ColIndex] * (v * pred_ik[..., np.newaxis] + (E[..., ColIndex] - E[..., iPES])[..., np.newaxis] / 2.0 * grad_ik[..., self.config.DIM:])).sum(-1)
			return time_deriv

		def evolve_parameter(
			weight: torch.Tensor,
			test_den_time_deriv: torch.Tensor,
			grad_inducing: torch.Tensor,
			grad_feature: torch.Tensor,
			grad_label: torch.Tensor,
			grad_param: torch.Tensor,
			inducing_time_deriv: torch.Tensor,
			feature_time_deriv: torch.Tensor,
			label_time_deriv: torch.Tensor,
			param_old: torch.Tensor,
			param_now: torch.Tensor
		) -> torch.Tensor:
			r"""To evolve the parameter by the given information

			Parameters
			----------
			weight : torch.Tensor
				The monte carlo weights of the points
			test_den_time_deriv : torch.Tensor
				Time derivative of the density at the monte carlo points
			grad_inducing: torch.Tensor
				Derivative of prediction at the monte carlo points over inducing points
			grad_feature : torch.Tensor
				Derivative of prediction at the monte carlo points over training features
			grad_label : torch.Tensor
				Derivative of prediction at the monte carlo points over training labels
			grad_param : torch.Tensor
				Derivative of prediction at the monte carlo points over hyperparameters
			inducing_time_deriv : torch.Tensor
				Derivative of inducing points over time
			feature_time_deriv : torch.Tensor
				Derivative of training feature over time
			label_time_deriv : torch.Tensor
				Derivative of training label over time
			param_old : torch.Tensor
				Hyperparameters from last time step
			param_now : torch.Tensor
				Hyperparameters from this time step

			Returns
			-------
			torch.Tensor
				Hyperparameters for the next time step
			"""
			param_ref_epsilon: typing.Final = 0.01 # param_ref(t)=(1-epsilon)*param(t-dt)+epsilon*param(t)
			damp_c: typing.Final = 1.0
			damp_epsilon: typing.Final = 1e-6 # gamma(t)=c*||d(param)/dt||/(||param(t)-param_ref(t)||+epsilon*||param(t)||)
			residual: typing.Final[torch.Tensor] = test_den_time_deriv - grad_inducing.reshape(grad_inducing.shape[0], -1) @ inducing_time_deriv.reshape(-1) - grad_feature.reshape(grad_feature.shape[0], -1) @ feature_time_deriv.reshape(-1) - grad_label @ label_time_deriv
			time_depend: typing.Final[torch.Tensor] = torch.linalg.ldl_solve(*torch.linalg.ldl_factor((grad_param.T * grad_param.T[:, torch.newaxis] / weight).mean(-1)), (grad_param.T * residual / weight).mean(-1, True))[..., 0] # it times dt gives Euler
			param_ref: typing.Final[torch.Tensor] = param_old + param_ref_epsilon * (param_now - param_old)
			param_ref_diff: typing.Final[torch.Tensor] = param_now - param_ref
			damp_coe: typing.Final[float] = damp_c * time_depend.norm().item() / (param_ref_diff.norm().item() + damp_epsilon * param_now.norm().item())
			damp_coe_exp: typing.Final[float] = math.exp(-damp_coe * dt)
			return param_ref + param_ref_diff * damp_coe_exp + (1 - damp_coe_exp) / damp_coe * time_depend

		# evolve saved predictors adiabatically
		for iPES, jPES, iTrig, iElement in zip(self.config.TRIL_ROW_INDICES, self.config.TRIL_COL_INDICES, self.config.TRIG_RANGE, self.config.TRIL_ELEMENT_INDICES):
			SymElmIndex: int = jPES * self.config.NUM_PES + iPES
			# evolve x_all
			x_all: torch.Tensor = self.epmca.point_set[iTrig]
			# get x and p, and 2 semi adiabatic steps
			x0: torch.Tensor = x_all[:, :model.config.DIM] # M * D
			p0: torch.Tensor = x_all[:, model.config.DIM:] # M * D
			x2: torch.Tensor # M * D
			p1: torch.Tensor # M * D
			x2, p1 = evolve.evolve_coordinates_adiabatically(model, x0, p0, mass, dt / 2.0, GPRPredictors.drc, iPES, jPES)
			x4: torch.Tensor # M * D
			p2: torch.Tensor # M * D
			x4, p2 = evolve.evolve_coordinates_adiabatically(model, x2, p1, mass, dt / 2.0, GPRPredictors.drc, iPES, jPES)
			x_all_new: torch.Tensor = torch.cat((x4, p2), -1)
			dx_all_dt: torch.Tensor = coord_derivative(x_all, iPES, jPES)
			pt_ctr: torch.Tensor = torch.mean(x_all, 0)
			pt_std: torch.Tensor = torch.std(self.__saved_predictors[iElement][-1].x_ind, 0)
			pt_for_param_evo: torch.Tensor = normal_sample(self.__saved_predictors[iElement][-1].x_ind.shape[0], pt_ctr, pt_std)
			pt_wt: torch.Tensor = self.epmca._gaussian_weight(pt_for_param_evo, pt_ctr, pt_std)
			dpt_dt: torch.Tensor = coord_derivative(pt_for_param_evo, iPES, jPES)
			# accumulate d(residual))/dt and residual from each predictor
			d_res_dt: torch.Tensor = density_derivative(pt_for_param_evo, dpt_dt, None, None, iPES, jPES)
			d_res_label_dt: torch.Tensor = density_derivative(x_all, dx_all_dt, None, None, iPES, jPES)
			res_label: torch.Tensor = torch.zeros_like(self.epmca.density[iTrig])
			# evolve each saved predictor of the element
			for pred_real, pred_imag in zip(self.__saved_predictors[SymElmIndex][:-1], self.__saved_predictors[iElement][:-1]):
				# evolve parameter first
				dx_ind_dt: torch.Tensor = coord_derivative(pred_real.x_ind, iPES, jPES) # num_ind * PHASEDIM
				test_den_time_deriv: torch.Tensor = density_derivative(pt_for_param_evo, dpt_dt, pred_real, None if iPES == jPES else pred_imag, iPES, jPES)
				label_den_time_deriv: torch.Tensor = density_derivative(x_all, dx_all_dt, pred_real, None if iPES == jPES else pred_imag, iPES, jPES)
				d_res_dt -= test_den_time_deriv
				d_res_label_dt -= label_den_time_deriv
				# deal with real and imag part separately
				pred_real.set_raw_lengthscale(evolve_parameter(
					pt_wt,
					test_den_time_deriv.real,
					*pred_real.predict_derivative_over_internal(pt_for_param_evo, self.chunk_size),
					dx_ind_dt,
					dx_all_dt,
					label_den_time_deriv.real,
					pred_real._old_raw_lengthscale,
					pred_real._raw_lengthscale
				))
				if iPES != jPES:
					pred_imag.set_raw_lengthscale(evolve_parameter(
						pt_wt,
						test_den_time_deriv.imag,
						*pred_imag.predict_derivative_over_internal(pt_for_param_evo, self.chunk_size),
						dx_ind_dt,
						dx_all_dt,
						label_den_time_deriv.imag,
						pred_imag._old_raw_lengthscale,
						pred_imag._raw_lengthscale
					))
				# real/imaginary share inducing points, so they should be evolved together
				saved_ind_new: torch.Tensor = torch.cat(evolve.evolve_coordinates_adiabatically(model, pred_real.x_ind[:, :self.config.DIM], pred_real.x_ind[:, self.config.DIM:], mass, dt, GPRPredictors.drc, iPES, jPES, True), -1)
				if iPES != jPES:
					# density need to be combined and evolved
					saved_den: torch.Tensor = pred_real.y_all + 1.j * pred_imag.y_all
					evolve.evolve_density_adiabatically(model, saved_den, x0, x2, x4, GPRPredictors.drc, dt, iPES, jPES)
					pred_real.update(saved_ind_new, x_all_new, saved_den.real)
					pred_imag.update(saved_ind_new, x_all_new, saved_den.imag)
					res_label -= saved_den
				else:
					# the same predictor, no density evolution in fact
					pred_real.update(saved_ind_new, x_all_new, pred_real.y_all)
					res_label -= pred_real.y_all
			# evolve parameter of the residual predictor
			x_ind: torch.Tensor = self.__saved_predictors[iElement][-1].x_ind
			dx_ind_dt: torch.Tensor = coord_derivative(x_ind, iPES, jPES)
			if iPES == jPES:
				self.__saved_predictors[iElement][-1].set_raw_lengthscale(evolve_parameter(
					pt_wt,
					d_res_dt.real,
					*self.__saved_predictors[iElement][-1].predict_derivative_over_internal(pt_for_param_evo, self.chunk_size),
					dx_ind_dt,
					dx_all_dt,
					d_res_label_dt.real,
					self.__saved_predictors[iElement][-1]._old_raw_lengthscale,
					self.__saved_predictors[iElement][-1]._raw_lengthscale
				))
			else:
				self.__saved_predictors[iElement][-1].set_raw_lengthscale(evolve_parameter(
					pt_wt,
					d_res_dt.imag,
					*self.__saved_predictors[iElement][-1].predict_derivative_over_internal(pt_for_param_evo, self.chunk_size),
					dx_ind_dt,
					dx_all_dt,
					d_res_label_dt.imag,
					self.__saved_predictors[iElement][-1]._old_raw_lengthscale,
					self.__saved_predictors[iElement][-1]._raw_lengthscale
				))
				self.__saved_predictors[SymElmIndex][-1].set_raw_lengthscale(evolve_parameter(
					pt_wt,
					d_res_dt.real,
					*self.__saved_predictors[SymElmIndex][-1].predict_derivative_over_internal(pt_for_param_evo, self.chunk_size),
					dx_ind_dt,
					dx_all_dt,
					d_res_label_dt.real,
					self.__saved_predictors[SymElmIndex][-1]._old_raw_lengthscale,
					self.__saved_predictors[SymElmIndex][-1]._raw_lengthscale
				))
			# evolve inducing points
			x_ind[:, :self.config.DIM], x_ind[:, self.config.DIM:] = evolve.evolve_coordinates_adiabatically(
				model,
				x_ind[:, :self.config.DIM],
				x_ind[:, self.config.DIM:],
				mass,
				dt,
				GPRPredictors.drc,
				iPES,
				jPES,
				True
			)
			# evolve y_all
			self.epmca.density[iTrig] = evolve.evolve_density_non_adiabatically(model, self.epmca.density[iTrig], x4, p2, x2, p1, mass, dt, self.predict, iPES, jPES)
			# finally set up the point coordinates
			self.epmca.point_set[iTrig] = x_all_new
			# and update the predictor
			if iPES == jPES:
				self.__saved_predictors[iElement][-1].update(x_ind, x_all_new, self.epmca.density[iTrig].real + res_label.real)
			else:
				self.__saved_predictors[iElement][-1].update(x_ind, x_all_new, self.epmca.density[iTrig].imag + res_label.imag)
				self.__saved_predictors[SymElmIndex][-1].update(x_ind, x_all_new, self.epmca.density[iTrig].real + res_label.real)
		# then check whether to save the current residual predictor
		for iPES, jPES, iElement, y in zip(self.config.TRIL_ROW_INDICES, self.config.TRIL_COL_INDICES, self.config.TRIL_ELEMENT_INDICES, self.epmca.density):
			SymElmIdx: int = jPES * self.config.NUM_PES + iPES
			x_all: torch.Tensor = self.__saved_predictors[iElement][-1].x_all
			y_res: torch.Tensor = y - self.predict(x_all, iElement)
			pts_include: torch.Tensor = y.abs() > y.abs().max() * GPRPredictors.__JUDGE_INCLUDE_THRESHOLD
			if torch.any(y_res[pts_include].abs() > GPRPredictors.__JUDGE_REGISTER_THRESHOLD * y[pts_include].abs()).item():
				print(f"Register the residual predictor for {plot.get_element_label(iPES, jPES)}; now maximum residual is {y_res.abs().max()} at {plot.format_array(x_all[y_res.abs().argmax()])}")
				# adjust predictor
				self.__saved_predictors[iElement][-1] = super(gp.ResidualPredictor, typing.cast(gp.ResidualPredictor, self.__saved_predictors[iElement][-1])) # update the predictor to `gp.SinglePredictor`, whose lengthscale mapping is now the base one
				# then choose the inducing points
				x_ind: torch.Tensor = self.__kmeans(x_all[torch.newaxis]).centers[0]
				# and add the new predictor (where the parameter is trained)
				if iPES == jPES:
					self.__saved_predictors[iElement].append(gp.ResidualPredictor(
						x_ind,
						x_all,
						y_res.real,
						self.__saved_predictors[iElement][-1].lengthscale
					))
				else: # if iPES < jPES:
					self.__saved_predictors[SymElmIdx][-1] = super(gp.ResidualPredictor, typing.cast(gp.ResidualPredictor, self.__saved_predictors[SymElmIdx][-1])) # update the predictor to `gp.SinglePredictor`
					self.__saved_predictors[iElement].append(gp.ResidualPredictor(
						x_ind,
						x_all,
						y_res.imag,
						self.__saved_predictors[iElement][-1].lengthscale
					))
					self.__saved_predictors[SymElmIdx].append(gp.ResidualPredictor(
						x_ind,
						x_all,
						y_res.real,
						self.__saved_predictors[SymElmIdx][-1].lengthscale
					))

	def train(self, print_log: bool = constant.DEBUG_MODE) -> None:
		r"""To train each residual predictor

		Parameters
		----------
		print_log : bool, optional
			Whether to print the log to console, by default `constant.DEBUG_MODE`
		"""
		# then train the residual predictors
		for iElement in self.config.ELEMENT_RANGE:
			pred = typing.cast(gp.ResidualPredictor, self.__saved_predictors[iElement][-1])
			if __class__.__check_predictor(pred):
				print("\tTraining " + plot.get_RI_label(iElement, self.config.NUM_PES))
				pred.train(print_log)

	def print(self, f: typing.IO) -> None:
		r"""To print the parameters to file

		Parameters
		----------
		f : io.TextIOWrapper
			The file to save the parameters
		"""
		for preds in self.__saved_predictors:
			for pred in preds:
				np.savetxt(f, pred.lengthscale.detach().cpu().numpy().reshape(1, -1), constant.FMT, footer="", comments="", encoding=constant.ENC)
			print("", file=f)
		print("\n", file=f, flush=constant.DEBUG_MODE)
