r"""pred
====
This module deals with evolutions and GP.
"""
import collections.abc
import math
import typing

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

	@staticmethod
	def __generate_extra_points(pts: torch.Tensor, extra_ratio: int) -> torch.Tensor:
		r"""To generate extra points

		Parameters
		----------
		pts : torch.Tensor, shape of (NUM_PTS, PHASEDIM)
			Inducing points
		extra_ratio : int
			Number of extra points around each point

		Returns
		-------
		torch.Tensor, shape of (NUM_XTR, PHASEDIM)
			Extra points
		"""
		# weights: typing.Final[torch.Tensor] = density.abs().unsqueeze(-1) # NUM_TRIG, NUM_PTS, 1
		# weights_sum: typing.Final[torch.Tensor] = weights.sum(1, True) # NUM_TRIG, 1, 1
		# w_ave: typing.Final[torch.Tensor] = (pts * weights).sum(1, True) / weights_sum # NUM_TRIG, 1, PHASEDIM
		# w_stddev: typing.Final[torch.Tensor] = torch.sqrt(pts.shape[0] / (pts.shape[0] - 1) * (torch.square((pts - w_ave)) * weights).sum(0) / weights_sum)
		# return torch.stack([torch.cat([expectation.normal_sample(extra_ratio, p, w_stddev) for p in pt], 0) for pt in pts], 0)
		std: typing.Final[torch.Tensor] = torch.std(pts, 0) # PHASEDIM
		return torch.cat([expectation.normal_sample(extra_ratio, pt, std) for pt in pts], 0)

	drc: typing.Final = evolve.Direction.FORWARD
	__JUDGE_INCLUDE_THRESHOLD: typing.Final = 0.1
	__JUDGE_RESAMPLE_THRESHOLD: typing.Final = 1e-8
	__slots__: typing.Final[tuple] = ("__kmeans", "__num_pts", "__num_kept", "ind_pts", "ind_den", "__extra_ratio", "xtr_pts", "xtr_den", "__kernel", "__predictors", "__lr", "__last_ppl", "__last_prt", "__last_param", "__purity_weight", "chunk_size")
	__kmeans: typing.Final[torch_kmeans.KMeans]
	__num_pts: typing.Final[int]
	__num_kept: typing.Final[int]
	ind_pts: torch.Tensor # NUM_TRIG, NUM_PTS, PHASEDIM
	ind_den: torch.Tensor # NUM_TRIG, NUM_PTS
	__extra_ratio: typing.Final[int]
	xtr_pts: torch.Tensor # NUM_TRIG, NUM_XTR, PHASEDIM
	xtr_den: torch.Tensor # NUM_TRIG, NUM_XTR
	__predictors: typing.Final[list[gp.GaussianProcess]]
	__lr: float | None
	__last_ppl: float
	__last_prt: float
	__last_param: torch.Tensor
	__purity_weight: typing.Final[torch.Tensor]
	chunk_size: typing.Final[int]

	def __init__(
		self,
		config: pes.ModelConfig,
		init_dist: pes.InitialDistribution,
		epmca: expectation.EvolvingPointsMCAverage,
		num_ind: int,
		kernel_initial_value: torch.Tensor,
	):
		super().__init__(config)
		self.__kmeans = torch_kmeans.KMeans(init_method="k-means++", n_clusters=num_ind, seed=constant.SEED, verbose=constant.DEBUG_MODE)
		self.__num_pts = num_ind
		self.__num_kept = num_ind // 2
		ind_pt: typing.Final[torch.Tensor] = self.__kmeans(epmca.point_set[:1, epmca.density[0].real > GPRPredictors.__JUDGE_INCLUDE_THRESHOLD * epmca.density[0].real.max()]).centers
		self.ind_pts = torch.repeat_interleave(ind_pt, self.config.NUM_TRIG, 0)
		self.ind_den = init_dist(ind_pt[0])[:, config.TRIL_ROW_INDICES, config.TRIL_COL_INDICES].T
		self.__extra_ratio = epmca.density.shape[1] // num_ind
		self.xtr_pts = epmca.point_set.clone()
		self.xtr_den = epmca.density.clone()
		self.__kernel = wendland.wendland_rbf(config.PHASEDIM)
		self.__predictors = []
		# add the first element
		print("\tInitial Training " + plot.get_RI_label(0, config.NUM_PES))
		self.__predictors.append(gp.GaussianProcess(
			self.ind_pts[0],
			self.ind_den[0],
			self.xtr_pts[0],
			self.xtr_den[0],
			self.__kernel,
			kernel_initial_value,
			wendland.wendland_rbf.r_c_init,
			2,
			False
		))
		self.__last_param = torch.full((config.NUM_ELM,), self[0].r_cutoff, dtype=torch.float64)
		for iTrig, iPES, jPES in zip(config.TRIG_RANGE[1:], config.TRIL_ROW_INDICES[1:], config.TRIL_COL_INDICES[1:]):
			print("\tInitial Training " + plot.get_element_label(iPES, jPES))
			# this includes training of initial distribution
			self.__predictors.append(gp.GaussianProcess(
				self.ind_pts[iTrig],
				self.ind_den[iTrig],
				self.xtr_pts[iTrig],
				self.xtr_den[iTrig],
				self.__kernel,
				kernel_initial_value,
				self[0].r_cutoff,
				2,
				False
			))
		self.__lr = 1.0 if config.NUM_ELM > 10 else None
		self.__last_ppl = 1.0
		self.__last_prt = 1.0
		self.__last_param = self.__real_to_raw()
		self.__purity_weight = torch.ones(config.NUM_TRIG) * 2 - torch.eye(config.NUM_PES)[config.TRIL_ROW_INDICES, config.TRIL_COL_INDICES]
		self.train(False)
		# then check for chunk size to avoid OOM in autograd
		self.chunk_size = num_ind # self.__predictors[0].get_chunk_size(expectation.normal_sample(num_ind, init_dist.r0, init_dist.sigma_r0))

	def population(self) -> torch.Tensor:
		return torch.tensor([self[iDiag].population for iDiag in self.config.FLATTEN_TRIL_DIAG_INDEX], dtype=torch.get_default_dtype(), device=torch.get_default_device())

	def coordinates(self) -> torch.Tensor:
		return sum((self[iDiag].coordinates for iDiag in self.config.FLATTEN_TRIL_DIAG_INDEX), start=torch.zeros(self.config.PHASEDIM))

	def square_coordinates(self) -> torch.Tensor:
		return sum((self[iDiag].square_coordinates for iDiag in self.config.FLATTEN_TRIL_DIAG_INDEX), start=torch.zeros((self.config.PHASEDIM, self.config.PHASEDIM)))

	def covariance(self) -> torch.Tensor:
		return super().covariance()

	def potential(self, model: pes.Potential) -> float:
		return math.nan

	def purity(self) -> torch.Tensor:
		result: torch.Tensor = torch.zeros(self.config.NUM_PES, self.config.NUM_PES, dtype=torch.double)
		result[self.config.TRIL_ROW_INDICES, self.config.TRIL_COL_INDICES] = torch.tensor([pred.purity for pred in self.__predictors])
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
		result: torch.Tensor = torch.zeros(self.config.NUM_PES, self.config.NUM_PES)
		result[self.config.TRIL_ROW_INDICES, self.config.TRIL_COL_INDICES] = 1.0 / torch.amax(self.xtr_den.abs(), -1)
		return (result + result.T - torch.diag(torch.diag(result))).reshape(-1)

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
		assert 0 <= ElementIndex < self.config.NUM_ELM
		pred: typing.Final[gp.GaussianProcess] = self.__predictors[self.config.FLATTEN_TRIL_INDEX[ElementIndex]]
		if GPRPredictors.__check_predictor(pred):
			return pred.predict(x_input)
		else:
			return torch.zeros(x_input.shape[:-1], dtype=torch.cdouble)

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
		if isinstance(dimensions, int):
			dimensions = [dimensions]
		else:
			dimensions = tuple(set(dimensions)) # remove duplicate
		assert all(0 <= dim <= self.config.PHASEDIM for dim in dimensions)
		assert x_input.shape[-1] == len(dimensions)
		pred: typing.Final[gp.GaussianProcess] = self.__predictors[self.config.FLATTEN_TRIL_INDEX[ElementIndex]]
		if GPRPredictors.__check_predictor(pred):
			return pred.get_marginal(x_input, dimensions)
		else:
			return torch.zeros(x_input.shape[0], dtype=torch.cdouble)

	def evolve_update(
		self,
		model: pes.Potential,
		mass: torch.Tensor,
		dt: float
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
		"""
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
		# 			time_deriv -= (D[..., RowIndex, iPES] * (v * pred_kj[..., torch.newaxis] + (E[..., RowIndex] - E[..., iPES])[..., torch.newaxis] / 2.0 * grad_kj[..., self.config.DIM:])).sum(-1)
		# 		if iPES != ColIndex:
		# 			pred_ik, grad_ik = predict_derivative_over_input(x, RowIndex, iPES)
		# 			time_deriv += (D[..., iPES, ColIndex] * (v * pred_ik[..., torch.newaxis] + (E[..., ColIndex] - E[..., iPES])[..., torch.newaxis] / 2.0 * grad_ik[..., self.config.DIM:])).sum(-1)
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
		# 	x_all: torch.Tensor = self.xtr_pts[iTrig]
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
		# 	self.xtr_den[iTrig] = evolve.evolve_density_non_adiabatically(model, self.xtr_den[iTrig], x4, p2, x2, p1, mass, dt, self.predict, iPES, jPES)
		# 	# finally set up the point coordinates
		# 	self.xtr_pts[iTrig] = x_all_new
		evolve.evolve(model, [*self.ind_pts], [*self.ind_den], mass, dt, self.predict)
		evolve.evolve(model, [*self.xtr_pts], [*self.xtr_den], mass, dt, self.predict)
		# and update the predictor
		for iTrig in self.config.TRIG_RANGE:
			self.__predictors[iTrig].update(self.ind_pts[iTrig], self.ind_den[iTrig], self.xtr_pts[iTrig], self.xtr_den[iTrig])

	def resample(
		self,
		model: pes.Potential,
		mass: torch.Tensor,
		dt: float
	) -> None:
		r"""To resample the points and density

		After this function, `evolve_update` must be called as the predictors are not updated

		Parameters
		----------
		model : pes.Potential
			Quantities derived from potential
		mass : torch.Tensor, shape of (DIM,)
			Mass of classical degree of freedom
		dt : float
			Time interval
		"""
		if self.__num_kept >= self.__num_pts:
			return
		result_pts: typing.Final[torch.Tensor] = torch.empty_like(self.ind_pts[0])
		result_idx: typing.Final[torch.Tensor] = torch.empty((self.__num_pts,), dtype=torch.int64)
		result_weight: typing.Final[torch.Tensor] = torch.empty((self.__num_pts,))
		ind_hop: list[torch.Tensor] = [torch.empty((0, self.config.PHASEDIM)) for _ in range(self.config.NUM_TRIG)]
		action_hop: list[torch.Tensor] = [torch.empty((0,)) for _ in range(self.config.NUM_TRIG)]
		for iPES, jPES, iTrig in zip(self.config.TRIL_ROW_INDICES, self.config.TRIL_COL_INDICES, self.config.TRIG_RANGE):
			x = self.ind_pts[iTrig, :, :self.config.DIM]
			nac: torch.Tensor = model.coupling(x)
			if nac.abs().max().item() < 1e-2:
				continue
			p = self.ind_pts[iTrig, :, self.config.DIM:]
			flow: torch.Tensor = torch.sum((p / mass)[..., torch.newaxis, torch.newaxis] * nac, -3) * dt * self.ind_den[iTrig, :, torch.newaxis, torch.newaxis] # NUM_PTS, NUM_PES, NUM_PES
			engs: torch.Tensor = model.adiabatic_potential(x) # NUM_PTS, NUM_PES
			double_kinetic: torch.Tensor = torch.sum(p ** 2 / mass, -1, True) # NUM_PTS, 1
			# ij->kj
			momentum_rescale_factor_sq: torch.Tensor = 1.0 + (engs[:, iPES:iPES + 1] - engs) / double_kinetic # NUM_PTS, NUM_PES
			transition: torch.Tensor = momentum_rescale_factor_sq >= 0.0
			for kPES in self.config.PES_RANGE:
				transition_k: torch.Tensor = transition[:, kPES]
				if kPES != iPES and torch.any(transition_k).item():
					DestIdx: int = self.config.FLATTEN_TRIL_INDEX[max(kPES, jPES) * self.config.NUM_PES + min(kPES, jPES)]
					ind_hop[DestIdx] = torch.cat([ind_hop[DestIdx], torch.cat([x[transition_k], p[transition_k] * torch.sqrt(momentum_rescale_factor_sq[transition_k, kPES:kPES + 1])], -1)], 0)
					action_hop[DestIdx] = torch.cat([action_hop[DestIdx], -flow[transition_k, kPES, iPES]], 0)
			if iPES != jPES:
				# ij->ik
				momentum_rescale_factor_sq: torch.Tensor = 1.0 + (engs[:, jPES:jPES + 1] - engs) / double_kinetic
				transition: torch.Tensor = momentum_rescale_factor_sq >= 0.0
				for kPES in self.config.PES_RANGE:
					transition_k: torch.Tensor = transition[:, kPES]
					if kPES != jPES and torch.any(transition_k).item():
						DestIdx: int = self.config.FLATTEN_TRIL_INDEX[max(iPES, kPES) * self.config.NUM_PES + min(iPES, kPES)]
						ind_hop[DestIdx] = torch.cat([ind_hop[DestIdx], torch.cat([x[transition_k], p[transition_k] * torch.sqrt(momentum_rescale_factor_sq[transition_k, kPES:kPES + 1])], -1)], 0)
						action_hop[DestIdx] = torch.cat([action_hop[DestIdx], flow[transition_k, jPES, kPES]], 0)
		for iTrig, pred in enumerate(self.__predictors): # judge, combine
			if ind_hop[iTrig].shape[0] == 0:
				continue
			lengthscale: torch.Tensor = pred.lengthscale
			r_c: torch.Tensor = pred.r_c
			# find new points far from the current points
			new_far_to_ind: torch.Tensor = wendland.cdist2(ind_hop[iTrig] / lengthscale, self.ind_pts[iTrig] / lengthscale).amin(1) > 0.01 # exp(-r^2/2)~0.995 when r=0.1
			den_abs: torch.Tensor = self.ind_den[iTrig].abs()
			flow_abs: torch.Tensor = action_hop[iTrig][new_far_to_ind].abs()
			flow_sum: float = flow_abs.sum().item()
			if flow_sum / (den_abs.sum().item() + flow_sum) < GPRPredictors.__JUDGE_RESAMPLE_THRESHOLD or not new_far_to_ind.any().item(): # too small, or too close
				continue
			result_idx[:self.__num_kept] = torch.argsort(den_abs, descending=True)[:self.__num_kept]
			result_pts[:self.__num_kept] = self.ind_pts[iTrig, result_idx[:self.__num_kept]]
			pts: torch.Tensor = torch.cat([self.ind_pts[iTrig], ind_hop[iTrig][new_far_to_ind]], 0)
			weights: torch.Tensor = torch.cat([den_abs, flow_abs], 0)
			knn: torch.Tensor = pred.kernel(pts, pts, lengthscale=lengthscale, r_c=r_c)
			knm: torch.Tensor = pred.kernel(pts, result_pts[:self.__num_kept], lengthscale=lengthscale, r_c=r_c)
			result_weight[:self.__num_kept] = torch.cholesky_solve(
				knm.T @ weights.unsqueeze(-1),
				torch.linalg.cholesky_ex(pred.kernel(result_pts[:self.__num_kept], result_pts[:self.__num_kept], lengthscale=lengthscale, r_c=r_c) + 1e-12 * torch.eye(self.__num_kept))[0]
			)[:, 0]
			for iPt in range(self.__num_kept, self.__num_pts):
				idx: int = int((knn @ weights - knm @ result_weight[:iPt]).abs().argmax().item())
				result_idx[iPt] = idx
				result_pts[iPt] = pts[idx]
				knm = torch.cat([knm, pred.kernel(pts, result_pts[iPt:iPt + 1], lengthscale=lengthscale, r_c=r_c)], -1)
				result_weight[:iPt + 1] = torch.cholesky_solve(
					knm.T @ weights.unsqueeze(-1),
					torch.linalg.cholesky_ex(pred.kernel(result_pts[:iPt + 1], result_pts[:iPt + 1], lengthscale=lengthscale, r_c=r_c) + 1e-12 * torch.eye(iPt + 1))[0]
				)[:, 0]
			# get density
			new_pts: torch.Tensor = result_idx >= self.__num_pts
			old_pts: torch.Tensor = torch.logical_not(new_pts)
			if constant.DEBUG_MODE:
				print(f"{self.config.TRIL_ROW_INDICES[iTrig]},{self.config.TRIL_COL_INDICES[iTrig]}: {ind_hop[iTrig][new_far_to_ind].shape[0]} new points, maximum flow = {flow_abs.max().item()}, maximum density = {den_abs.max().item()}, choose {torch.count_nonzero(new_pts).item()} new points")
			self.ind_den[iTrig, old_pts] = self.ind_den[iTrig, result_idx[old_pts]]
			self.ind_den[iTrig, new_pts] = pred.predict(result_pts[new_pts], lengthscale=lengthscale, r_c=r_c)
			self.ind_pts[iTrig] = result_pts
			if torch.any(new_pts).item():
				self.xtr_pts[iTrig] = GPRPredictors.__generate_extra_points(self.ind_pts[iTrig], self.__extra_ratio)
				self.xtr_den[iTrig] = pred.predict(self.xtr_pts[iTrig], lengthscale=lengthscale, r_c=r_c)

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
			return torch.stack([pred.raw_cutoff for pred in self.__predictors])
		else:
			return torch.stack([pred.rc_real_to_raw(real_param[i]) for i, pred in enumerate(self.__predictors)])

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
			return torch.stack([pred.r_c for pred in self.__predictors])
		else:
			return torch.stack([pred.rc_raw_to_real(raw_param[i]) for i, pred in enumerate(self.__predictors)])

	def __loss_func(self, raw_param: torch.Tensor, reg: bool) -> torch.Tensor:
		r"""To calculate the loss function for the global training of all predictors, including the population and purity penalty, and the regularization on the change of lengthscale

		Parameters
		----------
		raw_param : torch.Tensor
			The raw parameters of all predictors
		reg : bool
			Whether to use regularization on the change of lengthscale

		Returns
		-------
		torch.Tensor
			The loss value
		"""
		param: typing.Final[torch.Tensor] = self.__raw_to_real(raw_param)
		# strict lower, replace the upper one
		ppl_err: typing.Final[torch.Tensor] = self.__last_ppl - sum((self.__predictors[iDiag].population_with_lengthscale(self.__predictors[iDiag].lengthscale, param[iDiag]) for iDiag in self.config.FLATTEN_TRIL_DIAG_INDEX), start=torch.tensor(0.))
		prt: typing.Final[torch.Tensor] = torch.stack([pred.purity_with_lengthscale(pred.lengthscale, param[i]) for i, pred in enumerate(self.__predictors)])
		prt_err: typing.Final[torch.Tensor] = self.__last_prt - self.PURITY_FACTOR * (prt * self.__purity_weight).sum()
		loss: typing.Final[torch.Tensor] = sum((pred.loss_func(pred.lengthscale, param[i], False) for i, pred in enumerate(self.__predictors)), start=torch.tensor(0))
		return loss + self.xtr_pts.shape[1] * (prt_err ** 2 + ppl_err ** 2 + ((param - self.__last_param).square().sum() if reg else 0))

	def train(self, reg: bool, print_log: bool = constant.DEBUG_MODE) -> None:
		r"""To train each residual predictor

		Parameters
		----------
		reg : bool
			Whether to use regularization on the change of lengthscale
		print_log : bool, optional
			Whether to print the log to console, by default `constant.DEBUG_MODE`
		"""
		indent: typing.Final[int] = 1
		loss_func: typing.Final[typing.Callable[[torch.Tensor], torch.Tensor]] = lambda x: self.__loss_func(x, reg)
		# then train the residual predictors
		for iTrig, iPES, jPES in zip(self.config.TRIG_RANGE, self.config.TRIL_ROW_INDICES, self.config.TRIL_COL_INDICES):
			print(f"{indent * '\t'}Training {plot.get_element_label(iPES, jPES)}")
			self[iTrig].train(
				2,
				reg,
				False,
				print_log
			)
		# global training with population and purity penalty
		print(f"{indent * "\t"}Training All")
		self.__last_ppl = self.population().sum().item()
		self.__last_prt = self.purity().sum().item()
		self.__last_param = self.__raw_to_real().detach().clone()
		raw_param: torch.Tensor = torch.stack([pred.raw_cutoff for pred in self.__predictors])
		print(f"{indent * "\t"}{plot.format_array(self.scale, "scale")}\n{indent * "\t"}", end="")
		opt.Optimizer.print_model(self.__raw_to_real())
		if self.__lr is None:
			loss: float = math.inf
			while True:
				print(f"{indent * "\t"}Optimization with Newton method:")
				result = opt.NewtonMethod(raw_param, loss_func, indent + 1, print_log)
				print(f"{indent * "\t"}Iter = {result.num_iter} - {result.message}")
				opt.Optimizer.print_stuff(result.func_value, self.__raw_to_real(result.param), None, extra_start_str=indent)
				if result.message in (opt.Optimizer.ResultMessage.GRAD,) or result.func_value >= loss:
					break
				else:
					loss = result.func_value
					raw_param = result.param
				print(f"{indent * "\t"}Optimization with Gradient Descend method:")
				result = opt.GradientDescend(raw_param, loss_func, indent + 1, print_log=print_log)
				print(f"{indent * "\t"}Iter = {result.num_iter} - {result.message}")
				opt.Optimizer.print_stuff(result.func_value, self.__raw_to_real(result.param), result.lr, extra_start_str=indent)
				if result.message in (opt.Optimizer.ResultMessage.GRAD,) or result.func_value >= loss:
					break
				else:
					loss = result.func_value
					raw_param = result.param
		else:
			result = opt.GradientDescend(raw_param, loss_func, indent, self.__lr, print_log)
			print(f"{indent * "\t"}Iter = {result.num_iter} - {result.message}")
			opt.Optimizer.print_stuff(result.func_value, self.__raw_to_real(result.param), result.lr, extra_start_str=indent)
		with torch.no_grad():
			for iTrig in self.config.TRIG_RANGE:
				self.__predictors[iTrig].raw_cutoff = result.param[iTrig]

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
