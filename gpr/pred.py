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
import opt
import pes
import plot
import wendland


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
		predictor : gp.GaussianProcess
			The predictor

		Returns
		-------
		bool
			Availability of training / predicting
		"""
		return not torch.all(predictor.y_all == 0).item()

	drc: typing.Final = evolve.Direction.FORWARD
	__JUDGE_INCLUDE_THRESHOLD: typing.Final = 0.1
	__slots__: typing.Final[tuple] = ("epmca", "__kmeans", "ind_pts", "ind_den", "__kernel", "__predictors", "__lr", "__last_ppl", "__last_prt", "chunk_size")
	epmca: typing.Final[expectation.EvolvingPointsMCAverage]
	__kmeans: typing.Final[torch_kmeans.KMeans]
	ind_pts: torch.Tensor
	ind_den: torch.Tensor
	__predictors: typing.Final[list[gp.GaussianProcess]]
	__lr: float
	__last_ppl: float
	__last_prt: float
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
		self.epmca = expectation.EvolvingPointsMCAverage(config, num_pts, init_dist, init_stddev)
		self.__kmeans = torch_kmeans.KMeans(init_method="k-means++", n_clusters=num_ind, seed=constant.SEED, verbose=constant.DEBUG_MODE)
		ind_pt: typing.Final[torch.Tensor] = self.__kmeans(self.epmca.point_set[:1, self.epmca.density[0].real > GPRPredictors.__JUDGE_INCLUDE_THRESHOLD * self.epmca.density[0].real.max()]).centers[0]
		self.ind_pts = torch.repeat_interleave(ind_pt[torch.newaxis], self.config.NUM_TRIG, 0)
		self.ind_den = init_dist(ind_pt)[:, config.TRIL_ROW_INDICES, config.TRIL_COL_INDICES].T
		self.__kernel = wendland.wendland_rbf(config.PHASEDIM)
		self.__predictors = []
		# add the first element
		print("\tInitial Training " + plot.get_RI_label(0, config.NUM_PES))
		self.__predictors.append(gp.GaussianProcess(
			self.ind_pts[0],
			self.ind_den[0].real,
			self.epmca.point_set[0],
			self.epmca.density[0].real,
			self.__kernel,
			kernel_initial_value,
			wendland.wendland_rbf.r_c_init,
			2,
			True
		))
		for iElement in config.ELEMENT_RANGE[1:]:
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
					self[0].r_cutoff,
					2,
					False
				))
			else: # RowIndex > ColIndex
				self.__predictors.append(gp.GaussianProcess(
					self.ind_pts[TrilIndex],
					self.ind_den[TrilIndex].real,
					self.epmca.point_set[TrilIndex],
					self.epmca.density[TrilIndex].imag,
					self.__kernel,
					kernel_initial_value,
					self[0].r_cutoff,
					2,
					False
				))
		self.__lr = 1.0
		self.__last_ppl = 1.0
		self.__last_prt = 1.0
		self.train()
		# then check for chunk size to avoid OOM in autograd
		self.chunk_size = self.__predictors[0].get_chunk_size(expectation.normal_sample(num_ind, init_dist.r0, init_dist.sigma_r0))

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
		gp.GaussianProcess
			Predictor corresponding to the index in density supervector
		"""
		assert 0 <= ElementIndex < self.config.NUM_ELM
		return self.__predictors[ElementIndex]

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
		call_single_predictor : collections.abc.Callable[typing.Concatenate[gp.GaussianProcess, torch.Tensor, P], collections.abc.Iterable[torch.Tensor]]
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
			return tuple(item + 0.j for item in call_single_predictor(self.__predictors[RowIndex * self.config.NUM_PES + ColIndex], x_test, *args, **kwargs))
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
			pred : gp.GaussianProcess
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
		# r"""To evolve the coordinates and density

		# Parameters
		# ----------
		# model : pes.Potential
		# 	Quantities derived from potential
		# mass : torch.Tensor, shape of (DIM,)
		# 	Mass of classical degree of freedom
		# dt : float
		# 	Time interval
		# predictor : constant.Predictor
		# 	It predicts the density matrix element based on given coordinates and element index
		# """
		# def coord_derivative(x: torch.Tensor, RowIndex: int, ColIndex: int) -> torch.Tensor:
		# 	r"""To calculate dX/dt and drho_ij/dt

		# 	Parameters
		# 	----------
		# 	x : torch.Tensor, shape of (..., PHASEDIM)
		# 		Phase space coordinates
		# 	RowIndex : int
		# 		Index of row of the element in density matrix
		# 	ColIndex : int
		# 		Index of column of the element in density matrix

		# 	Returns
		# 	-------
		# 	torch.Tensor
		# 		dX/dt, shape of (..., PHASEDIM)
		# 	"""
		# 	r: torch.Tensor = x[..., :self.config.DIM] # position coordinates
		# 	v: torch.Tensor = x[..., self.config.DIM:] / mass # velocity
		# 	F = model.force(r)
		# 	return torch.cat((v, (F[..., RowIndex, RowIndex] + F[..., ColIndex, ColIndex]) / 2.0), -1)

		# def density_derivative(
		# 	x: torch.Tensor,
		# 	dXdt: torch.Tensor,
		# 	RowIndex: int,
		# 	ColIndex: int
		# ) -> torch.Tensor:
		# 	r"""To calculate dX/dt and drho_ij/dt

		# 	Parameters
		# 	----------
		# 	x : torch.Tensor, shape of (..., PHASEDIM)
		# 		Phase space coordinates
		# 	dXdt : torch.Tensor, shape of (..., PHASEDIM)
		# 		Time derivative of phase space coordinates
		# 	RowIndex : int
		# 		Index of row of the element in density matrix
		# 	ColIndex : int
		# 		Index of column of the element in density matrix

		# 	Returns
		# 	-------
		# 	torch.Tensor
		# 		drho_ij/dt, shape of (...)
		# 	"""
		# 	def predict_derivative_over_input(x_input: torch.Tensor, RowIndex: int, ColIndex: int) -> gp.GaussianProcess.InputDerivativeReturn:
		# 		r"""To combine the prediction and its derivative of the single predictor into complex arrays

		# 		Parameters
		# 		----------
		# 		x_input : torch.Tensor, shape of (N, PHASEDIM)
		# 			Test inputs
		# 		RowIndex : int
		# 			Index of row of the element in density matrix
		# 		ColIndex : int
		# 			Index of column of the element in density matrix

		# 		Returns
		# 		-------
		# 		gp.GaussianProcess.InputDerivativeReturn
		# 			The prediction, and the derivative of the prediction over input
		# 		"""
		# 		return gp.GaussianProcess.InputDerivativeReturn(*self.__combine_to_complex(
		# 			x_input,
		# 			RowIndex,
		# 			ColIndex,
		# 			lambda pred, x_test: pred.predict_derivative_over_input(x_test)
		# 		))

		# 	r: typing.Final[torch.Tensor] = x[..., :self.config.DIM] # position coordinates
		# 	v: typing.Final[torch.Tensor]= x[..., self.config.DIM:] / mass # velocity
		# 	E, D = model.adiabatic_potential_and_coupling(r)
		# 	pred, grad_input = predict_derivative_over_input(x, RowIndex, ColIndex)
		# 	# drho/dt at x_test
		# 	time_deriv: torch.Tensor = -(dXdt * grad_input).sum(-1)
		# 	if RowIndex != ColIndex:
		# 		time_deriv -= 1.0j / constant.HBAR * (E[..., RowIndex] - E[..., ColIndex]) * pred
		# 	for iPES in self.config.PES_RANGE:
		# 		if iPES != RowIndex:
		# 			pred_kj, grad_kj = predict_derivative_over_input(x, iPES, ColIndex)
		# 			time_deriv -= (D[..., RowIndex, iPES] * (v * pred_kj[..., np.newaxis] + (E[..., RowIndex] - E[..., iPES])[..., np.newaxis] / 2.0 * grad_kj[..., self.config.DIM:])).sum(-1)
		# 		if iPES != ColIndex:
		# 			pred_ik, grad_ik = predict_derivative_over_input(x, RowIndex, iPES)
		# 			time_deriv += (D[..., iPES, ColIndex] * (v * pred_ik[..., np.newaxis] + (E[..., ColIndex] - E[..., iPES])[..., np.newaxis] / 2.0 * grad_ik[..., self.config.DIM:])).sum(-1)
		# 	return time_deriv

		# def evolve_parameter(
		# 	weight: torch.Tensor,
		# 	test_den_time_deriv: torch.Tensor,
		# 	grad_inducing: torch.Tensor,
		# 	grad_feature: torch.Tensor,
		# 	grad_label: torch.Tensor,
		# 	grad_param: torch.Tensor,
		# 	inducing_time_deriv: torch.Tensor,
		# 	feature_time_deriv: torch.Tensor,
		# 	label_time_deriv: torch.Tensor,
		# 	param_old: torch.Tensor,
		# 	param_now: torch.Tensor
		# ) -> torch.Tensor:
		# 	r"""To evolve the parameter by the given information

		# 	Parameters
		# 	----------
		# 	weight : torch.Tensor
		# 		The monte carlo weights of the points
		# 	test_den_time_deriv : torch.Tensor
		# 		Time derivative of the density at the monte carlo points
		# 	grad_inducing: torch.Tensor
		# 		Derivative of prediction at the monte carlo points over inducing points
		# 	grad_feature : torch.Tensor
		# 		Derivative of prediction at the monte carlo points over training features
		# 	grad_label : torch.Tensor
		# 		Derivative of prediction at the monte carlo points over training labels
		# 	grad_param : torch.Tensor
		# 		Derivative of prediction at the monte carlo points over hyperparameters
		# 	inducing_time_deriv : torch.Tensor
		# 		Derivative of inducing points over time
		# 	feature_time_deriv : torch.Tensor
		# 		Derivative of training feature over time
		# 	label_time_deriv : torch.Tensor
		# 		Derivative of training label over time
		# 	param_old : torch.Tensor
		# 		Hyperparameters from last time step
		# 	param_now : torch.Tensor
		# 		Hyperparameters from this time step

		# 	Returns
		# 	-------
		# 	torch.Tensor
		# 		Hyperparameters for the next time step
		# 	"""
		# 	param_ref_epsilon: typing.Final = 0.01 # param_ref(t)=(1-epsilon)*param(t-dt)+epsilon*param(t)
		# 	damp_c: typing.Final = 1.0
		# 	damp_epsilon: typing.Final = 1e-6 # gamma(t)=c*||d(param)/dt||/(||param(t)-param_ref(t)||+epsilon*||param(t)||)
		# 	residual: typing.Final[torch.Tensor] = test_den_time_deriv - grad_inducing.reshape(grad_inducing.shape[0], -1) @ inducing_time_deriv.reshape(-1) - grad_feature.reshape(grad_feature.shape[0], -1) @ feature_time_deriv.reshape(-1) - grad_label @ label_time_deriv
		# 	time_depend: typing.Final[torch.Tensor] = torch.linalg.ldl_solve(*torch.linalg.ldl_factor((grad_param.T * grad_param.T[:, torch.newaxis] / weight).mean(-1)), (grad_param.T * residual / weight).mean(-1, True))[:, 0] # it times dt gives Euler
		# 	param_ref: typing.Final[torch.Tensor] = param_old + param_ref_epsilon * (param_now - param_old)
		# 	param_ref_diff: typing.Final[torch.Tensor] = param_now - param_ref
		# 	damp_coe: typing.Final[float] = damp_c * time_depend.norm().item() / (param_ref_diff.norm().item() + damp_epsilon * param_now.norm().item())
		# 	damp_coe_exp: typing.Final[float] = math.exp(-damp_coe * dt)
		# 	return param_ref + param_ref_diff * damp_coe_exp + (1 - damp_coe_exp) / damp_coe * time_depend

		# # evolve saved predictors adiabatically
		# for iPES, jPES, iTrig, iElement in zip(self.config.TRIL_ROW_INDICES, self.config.TRIL_COL_INDICES, self.config.TRIG_RANGE, self.config.TRIL_ELEMENT_INDICES):
		# 	SymElmIndex: int = jPES * self.config.NUM_PES + iPES
		# 	# evolve x_all
		# 	x_all: torch.Tensor = self.epmca.point_set[iTrig]
		# 	# get x and p, and 2 semi adiabatic steps
		# 	x0: torch.Tensor = x_all[:, :model.config.DIM] # M * D
		# 	p0: torch.Tensor = x_all[:, model.config.DIM:] # M * D
		# 	x2: torch.Tensor # M * D
		# 	p1: torch.Tensor # M * D
		# 	x2, p1 = evolve.evolve_coordinates_adiabatically(model, x0, p0, mass, dt / 2.0, GPRPredictors.drc, iPES, jPES)
		# 	x4: torch.Tensor # M * D
		# 	p2: torch.Tensor # M * D
		# 	x4, p2 = evolve.evolve_coordinates_adiabatically(model, x2, p1, mass, dt / 2.0, GPRPredictors.drc, iPES, jPES)
		# 	x_all_new: torch.Tensor = torch.cat((x4, p2), -1)
		# 	dx_all_dt: torch.Tensor = coord_derivative(x_all, iPES, jPES)
		# 	d_rho_dt: torch.Tensor = density_derivative(x_all, dx_all_dt, iPES, jPES)
		# 	x_ind: torch.Tensor = self.ind_pts[iTrig]
		# 	dx_ind_dt: torch.Tensor = coord_derivative(x_ind, iPES, jPES) # num_ind * PHASEDIM
		# 	self[SymElmIndex].update_param(evolve_parameter(
		# 		self.epmca.weight[iTrig],
		# 		d_rho_dt.real,
		# 		*self[SymElmIndex].predict_derivative_over_internal(x_all, self.chunk_size),
		# 		dx_ind_dt,
		# 		dx_all_dt,
		# 		d_rho_dt.real,
		# 		self[SymElmIndex].old_param,
		# 		self[SymElmIndex].raw_param
		# 	))
		# 	if iPES != jPES:
		# 		self[iElement].update_param(evolve_parameter(
		# 			self.epmca.weight[iTrig],
		# 			d_rho_dt.imag,
		# 			*self[iElement].predict_derivative_over_internal(x_all, self.chunk_size),
		# 			dx_ind_dt,
		# 			dx_all_dt,
		# 			d_rho_dt.imag,
		# 			self[iElement].old_param,
		# 			self[iElement].raw_param
		# 		))
		# 	# evolve y_all
		# 	self.epmca.density[iTrig] = evolve.evolve_density_non_adiabatically(model, self.epmca.density[iTrig], x4, p2, x2, p1, mass, dt, self.predict, iPES, jPES)
		# 	# finally set up the point coordinates
		# 	self.epmca.point_set[iTrig] = x_all_new
		# update the inducing points
		self.epmca.evolve(model, mass, dt, self.predict)
		evolve.evolve(model, [*self.ind_pts], [*self.ind_den], mass, dt, self.predict)
		# and update the predictor
		for iPES, jPES, iTrig, iElement in zip(self.config.TRIL_ROW_INDICES, self.config.TRIL_COL_INDICES, self.config.TRIG_RANGE, self.config.TRIL_ELEMENT_INDICES):
			if iPES == jPES:
				self.__predictors[iElement].update(self.ind_pts[iTrig], self.ind_den[iTrig].real, self.epmca.point_set[iTrig], self.epmca.density[iTrig].real)
			else:
				self.__predictors[iElement].update(self.ind_pts[iTrig], self.ind_den[iTrig].imag, self.epmca.point_set[iTrig], self.epmca.density[iTrig].imag)
				self.__predictors[jPES * self.config.NUM_PES + iPES].update(self.ind_pts[iTrig], self.ind_den[iTrig].real, self.epmca.point_set[iTrig], self.epmca.density[iTrig].real)

	def __real_to_raw(self, real_param: torch.Tensor | None = None) -> torch.Tensor:
		r"""To convert real parameters to raw parameters

		Parameters
		----------
		real_param : torch.Tensor | None, optional
			The real parameters, by default None and use the real parameters of each predictor

		Returns
		-------
		torch.Tensor
			The raw parameters
		"""
		if real_param is None:
			return torch.cat([pred.raw_param for pred in self.__predictors])
		else:
			real_param = real_param.reshape(self.config.NUM_ELM, -1)
			return torch.cat([pred.real_to_raw(real_param[i]) for i, pred in enumerate(self.__predictors)])

	def __raw_to_real(self, raw_param: torch.Tensor | None = None) -> torch.Tensor:
		r"""To convert raw parameters to real parameters

		Parameters
		----------
		raw_param : torch.Tensor | None, optional
			The raw parameters, by default None and return the real parameters of each predictor

		Returns
		-------
		torch.Tensor
			The real parameters
		"""
		if raw_param is None:
			return torch.cat([pred.param for pred in self.__predictors])
		else:
			raw_param = raw_param.reshape(self.config.NUM_ELM, -1)
			return torch.cat([pred.raw_to_real(raw_param[i]) for i, pred in enumerate(self.__predictors)])

	def __loss_func(self, raw_param: torch.Tensor) -> torch.Tensor:
		ppl_err: typing.Final[torch.Tensor] = self.__last_ppl - sum((self[iPES * self.config.NUM_PES + iPES].population_with_lengthscale(self[iPES * self.config.NUM_PES + iPES].raw_lengthscale, raw_param[iPES * self.config.NUM_PES + iPES]) for iPES in self.config.PES_RANGE), start=torch.tensor(0.))
		prt: typing.Final[torch.Tensor] = torch.stack([pred.purity_with_lengthscale(pred.raw_lengthscale, raw_param[i]) for i, pred in enumerate(self.__predictors)]).reshape(self.config.NUM_PES, self.config.NUM_PES)
		prt_err: typing.Final[torch.Tensor] = self.__last_prt - self.PURITY_FACTOR * (prt + prt.T - torch.diag(torch.diag(prt))).sum()
		loss: typing.Final[torch.Tensor] = sum((pred.loss_func(pred.raw_lengthscale, raw_param[i]) for i, pred in enumerate(self.__predictors)), start=torch.tensor(0))
		return loss + self.epmca.point_set.shape[1] * (prt_err ** 2 + ppl_err ** 2)

	def train(self, print_log: bool = constant.DEBUG_MODE) -> None:
		r"""To train each residual predictor

		Parameters
		----------
		print_log : bool, optional
			Whether to print the log to console, by default `constant.DEBUG_MODE`
		"""
		indent: typing.Final[int] = 1
		# then train the residual predictors
		for iElement, pred in enumerate(self.__predictors):
			print(f"{indent * "\t"}Training {plot.get_RI_label(iElement, self.config.NUM_PES)}")
			pred.train(2, False, print_log)
		# global training with population and purity penalty
		print(f"{indent * "\t"}Training All")
		raw_param: torch.Tensor = torch.stack([pred.raw_cutoff for pred in self.__predictors])
		print(f"{indent * "\t"}{plot.format_array(self.scale, "scale")}\n{indent * "\t"}", end="")
		opt.Optimizer.print_model(self.__raw_to_real())
		result = opt.GradientDescend(raw_param, self.__loss_func, indent, self.__lr, print_log)
		print(f"{indent * "\t"}Iter = {result.num_iter} - {result.message}")
		opt.Optimizer.print_stuff(result.func_value, self.__raw_to_real(result.param), result.lr, extra_start_str=indent)
		N_param: typing.Final[int] = raw_param.numel() // self.config.NUM_ELM
		raw_param = result.param.reshape(self.config.NUM_ELM, N_param)
		for i, pred in enumerate(self.__predictors):
			pred.raw_cutoff = raw_param[i]
		self.__last_ppl = self.population().sum().item()
		self.__last_prt = self.purity().sum().item()

	def print(self, f: typing.IO) -> None:
		r"""To print the parameters to file

		Parameters
		----------
		f : io.TextIOWrapper
			The file to save the parameters
		"""
		for pred in self.__predictors:
			print(plot.format_array(pred.lengthscale), pred.r_cutoff, end=" ", file=f)
		print("", file=f, flush=constant.DEBUG_MODE)
