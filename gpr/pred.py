r"""pred
====
This module deals with evolutions and GP.
"""
import collections.abc
import math
import typing

import numpy as np
import torch
import torch_kmeans

import constant
import expectation
import evolve
import gp
import pes
import plot


@typing.final
class GPRPredictors(expectation.Averager):
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
	epmca : expectation.EvolvingPointsMCAverage
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
	def __check_predictor(predictor: gp.GaussianProcess) -> bool:
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
	__slots__: typing.Final[tuple] = ("epmca", "__kmeans", "ind_pts", "ind_den", "__kernel", "__predictors")
	epmca: typing.Final[expectation.EvolvingPointsMCAverage]
	__kmeans: typing.Final[torch_kmeans.KMeans]
	ind_pts: torch.Tensor
	ind_den: torch.Tensor
	__predictors: typing.Final[list[gp.GaussianProcess]]
	# chunk_size: typing.Final[int]

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
		self.epmca = expectation.EvolvingPointsMCAverage(config, num_pts, init_dist, init_stddev)
		self.__kmeans = torch_kmeans.KMeans(init_method="k-means++", n_clusters=num_ind, seed=constant.SEED, verbose=constant.DEBUG_MODE)
		ind_pt: typing.Final[torch.Tensor] = self.__kmeans(self.epmca.point_set[:1, self.epmca.density[0].real > GPRPredictors.__JUDGE_INCLUDE_THRESHOLD * self.epmca.density[0].real.max()]).centers[0]
		self.ind_pts = torch.repeat_interleave(ind_pt[torch.newaxis], self.config.NUM_TRIG, 0)
		self.ind_den = init_dist(ind_pt)[:, config.TRIL_ROW_INDICES, config.TRIL_COL_INDICES].T
		self.__kernel = gp.wendland_rbf(config.PHASEDIM)
		self.__predictors = []
		for iElement in config.ELEMENT_RANGE:
			RowIndex: int = iElement // config.NUM_PES
			ColIndex: int = iElement % config.NUM_PES
			print("\tInitial Training " + plot.get_RI_label(iElement, config.NUM_PES))
			TrilIndex: int = self.config.FLATTEN_TRIL_INDEX[iElement]
			# this includes training of initial distribution
			if RowIndex <= ColIndex:
				self.__predictors.append(gp.GaussianProcess(
					self.ind_pts[TrilIndex],
					self.ind_den[TrilIndex].real,
					self.epmca.point_set[TrilIndex],
					self.epmca.density[TrilIndex].real,
					self.__kernel,
					kernel_initial_value,
					gp.wendland_rbf.r_c_init,
					2
				))
			else: # RowIndex > ColIndex
				self.__predictors.append(gp.GaussianProcess(
					self.ind_pts[TrilIndex],
					self.ind_den[TrilIndex].real,
					self.epmca.point_set[TrilIndex],
					self.epmca.density[TrilIndex].imag,
					self.__kernel,
					kernel_initial_value,
					gp.wendland_rbf.r_c_init,
					2
				))
		# then check for chunk size to avoid OOM in autograd
		# self.chunk_size = self.__predictors[0].get_chunk_size(expectation.normal_sample(num_ind, init_dist.r0, init_dist.sigma_r0))

	def population(self) -> torch.Tensor:
		return torch.tensor([self[iPES * self.config.NUM_PES + iPES].population for iPES in self.config.PES_RANGE], dtype=torch.get_default_dtype(), device=torch.get_default_device())

	def coordinates(self) -> torch.Tensor:
		return sum((self[iPES * self.config.NUM_PES + iPES].coordinates for iPES in self.config.PES_RANGE), start=torch.zeros(self.config.PHASEDIM))

	def square_coordinates(self) -> torch.Tensor:
		return sum((self[iPES * self.config.NUM_PES + iPES].square_coordinates for iPES in self.config.PES_RANGE), start=torch.zeros((self.config.PHASEDIM, self.config.PHASEDIM)))

	def covariance(self) -> torch.Tensor:
		return super().covariance()

	def potential(self, model: pes.Potential) -> float:
		return math.nan

	def purity(self) -> torch.Tensor:
		result: typing.Final[torch.Tensor] = torch.tensor([pred.purity for pred in self.__predictors]).reshape(self.config.NUM_PES, self.config.NUM_PES)
		return self.PURITY_FACTOR * (result + result.T - torch.diag(torch.diag(result)))

	def __getitem__(self, ElementIndex: int) -> gp.GaussianProcess:
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
		return self.__predictors[ElementIndex]

	@property
	def inducing_points(self) -> torch.Tensor:
		r"""To get all inducing points

		Returns
		-------
		torch.Tensor, shape of (NUM_TRIG, num_ind, PHASEDIM)
			Inducing points of independent, lower triangular elements
		"""
		return self.ind_pts

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

	def __combine_to_complex[**P](
		self,
		x_input: torch.Tensor,
		RowIndex: int,
		ColIndex: int,
		call_single_predictor: collections.abc.Callable[typing.Concatenate[gp.GaussianProcess, torch.Tensor, P], collections.abc.Iterable[torch.Tensor]],
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
		# pred_call: typing.Final[collections.abc.Callable[[gp.GaussianProcess], collections.abc.Iterable[torch.Tensor]]] = lambda pred: call_single_predictor(pred, x_test, *args, **kwargs)
		if RowIndex == ColIndex:
			return tuple(item for item in call_single_predictor(self.__predictors[RowIndex * self.config.NUM_PES + ColIndex], x_test, *args, **kwargs))
		elif RowIndex > ColIndex:
			return tuple(real + 1.j * imag for real, imag in zip(call_single_predictor(self.__predictors[ColIndex * self.config.NUM_PES + RowIndex], x_test, *args, **kwargs), call_single_predictor(self.__predictors[RowIndex * self.config.NUM_PES + ColIndex], x_test, *args, **kwargs)))
		else: # RowIndex < ColIndex
			return tuple(real - 1.j * imag for real, imag in zip(call_single_predictor(self.__predictors[RowIndex * self.config.NUM_PES + ColIndex], x_test, *args, **kwargs), call_single_predictor(self.__predictors[ColIndex * self.config.NUM_PES + RowIndex], x_test, *args, **kwargs)))

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
			x_input.reshape(-1, x_input.shape[-1]),
			ElementIndex // self.config.NUM_PES,
			ElementIndex % self.config.NUM_PES,
			lambda pred, x_test: (pred.predict(x_test),) if GPRPredictors.__check_predictor(pred) else (torch.zeros(x_test.shape[0]),)
		)[0].reshape(x_input.shape[:-1])

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
		def call_single_predictor(pred: gp.GaussianProcess, x_test: torch.Tensor, dims: collections.abc.Sequence[int]) -> tuple[torch.Tensor]:
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
		# evolve full set
		self.epmca.evolve(model, mass, dt, self.predict)
		# evolve inducing points
		evolve.evolve(model, [*self.ind_pts], [*self.ind_den], mass, dt, self.predict)
		# evolve saved predictors adiabatically
		for iPES, jPES, iTrig, iElement in zip(self.config.TRIL_ROW_INDICES, self.config.TRIL_COL_INDICES, self.config.TRIG_RANGE, self.config.TRIL_ELEMENT_INDICES):
			# and update the predictor
			if iPES == jPES:
				self.__predictors[iElement].update(self.ind_pts[iTrig], self.ind_den[iTrig].real, self.epmca.point_set[iTrig], self.epmca.density[iTrig].real)
			else:
				self.__predictors[iElement].update(self.ind_pts[iTrig], self.ind_den[iTrig].imag, self.epmca.point_set[iTrig], self.epmca.density[iTrig].imag)
				self.__predictors[jPES * self.config.NUM_PES + iPES].update(self.ind_pts[iTrig], self.ind_den[iTrig].real, self.epmca.point_set[iTrig], self.epmca.density[iTrig].real)

	def train(self, print_log: bool = constant.DEBUG_MODE) -> None:
		r"""To train each residual predictor

		Parameters
		----------
		print_log : bool, optional
			Whether to print the log to console, by default `constant.DEBUG_MODE`
		"""
		# then train the residual predictors
		for iElement, pred in enumerate(self.__predictors):
			print("\tTraining " + plot.get_RI_label(iElement, self.config.NUM_PES))
			pred.train(2, print_log)

	def print(self, f: typing.IO) -> None:
		r"""To print the parameters to file

		Parameters
		----------
		f : io.TextIOWrapper
			The file to save the parameters
		"""
		for pred in self.__predictors:
			np.savetxt(f, pred.lengthscale.detach().cpu().numpy().reshape(1, -1), constant.FMT, footer="", comments="", encoding=constant.ENC)
		print("\n", file=f, flush=constant.DEBUG_MODE)
