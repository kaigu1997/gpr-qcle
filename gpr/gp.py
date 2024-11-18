"""
gp
==
This module provides support for gaussian process (gp) regression.
"""
import collections.abc
import copy
import io
import math
import os
import sys

import gpytorch
import gpytorch.constraints
import numpy as np
import numpy.typing as npt
import sklearn.neighbors
import torch

sys.path.append(os.path.dirname(__file__))

import opt
import pes
import utility

torch.set_default_dtype(torch.double)
DEBUG_MODE: bool = True


class GP(gpytorch.models.ExactGP):
	"""
	Gaussian process regression

	Parameters
	----------
	x : torch.Tensor
		Training inputs
	y : torch.Tensor
		Training targets
	likelihood : gpytorch.likelihoods.Likelihood
		Likelihood for gaussian
	kernel : gpytorch.kernels.Kernel
		The kernel for gaussian

	Attributes
	----------
	COV_NAME : typing.Literal["cov"]
		Name of the covariance module of GPR

	Methods
	----------
	forward(x)
		The implementation of GPR
	"""
	COV_NAME = "cov"
	__slots__: tuple = ("__mean", "__cov")

	def __init__(self, x: torch.Tensor, y: torch.Tensor, likelihood: gpytorch.likelihoods.Likelihood, kernel: gpytorch.kernels.Kernel):
		super().__init__(x, y, likelihood)
		self.__mean: gpytorch.means.Mean = gpytorch.means.ZeroMean()
		self.__cov: gpytorch.kernels.Kernel = kernel

	@property
	def cov(self) -> gpytorch.kernels.Kernel:
		return self.__cov

	def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
		"""
		The implementation of GPR

		Parameters
		----------
		x : torch.Tensor
			Training inputs

		Returns
		-------
		gpytorch.distributions.MultivariateNormal
			A gaussian process with certain mean and covariance
		"""
		Mean = self.__mean(x)
		assert isinstance(Mean, torch.Tensor)
		return gpytorch.distributions.MultivariateNormal(Mean, self.__cov(x))


class NoConstraint(gpytorch.constraints.Interval):
	"""
	No constraint on the parameter, the value could be any float value

	Parameters
	----------
	initial_value : torch.Tensor | None, optional
		Initial value for the parameter, by default None

	Methods
	-------
	transform(tensor)
		To transform the raw value to the actual value
	inverse_transform(transformed_tensor)
		To transform the actual value to the raw value
	"""
	def __init__(self, initial_value: torch.Tensor | None = None):
		super().__init__(
			lower_bound=-math.inf,
			upper_bound=math.inf,
			transform=lambda x: x,
			inv_transform=lambda x: x,
			initial_value=initial_value,
		)

	def __repr__(self) -> str:
		"""
		The official string representation of an object

		Returns
		-------
		str
			The name of the class with an empty parenthesis
		"""
		return __class__.__name__ + "()"

	def transform(self, tensor: torch.Tensor) -> torch.Tensor:
		"""
		To transform the raw value to the actual value

		Parameters
		----------
		tensor : torch.Tensor
			The raw value

		Returns
		-------
		torch.Tensor
			The transformed, actual value
		"""
		return tensor

	def inverse_transform(self, transformed_tensor: torch.Tensor) -> torch.Tensor:
		"""
		To transform the actual value to the raw value

		Parameters
		----------
		transformed_tensor : torch.Tensor
			The transformed, actual value

		Returns
		-------
		torch.Tensor
			The raw value
		"""
		return transformed_tensor


class SinglePredictor:
	"""
	The instantiation of gaussian process predictor

	Parameters
	----------
	kernel : gpytorch.kernels.Kernel
		The kernel to use
	num_neighbors : int, optional
		The number of neighbors (`k`) searched by kNN for local GP, by default `__NUM_NEIGHBOR`

	Attributes
	----------
	__MAX_ITER : typing.Literal[15000]
		Maximum iteration of optimization
	__NOISE : float
		Extra noise term added for numerical stability in matrix inversion
	__NUM_NEIGHBOR : typing.Literal[32]
		The default number of neighbors to construct local GP

	Methods
	-------
	get_training_features()
		To get the training features of the core subset
	__update_weights()
		To update the weights, :math:`K^{-1}y`
	__update_neighbor()
		To construct the neighbors
	predict(x_test, use_local_GP)
		To predict the average estimation
	error()
		To get the error by comparing label with prediction
	initial_parameter_search(print_log)
		To search for a better initial value
	train(print_log, ftol, xtol)
		To train the parameters
	update(x_all, y_all, num_points)
		To update features and labels
	get_marginal(dimensions)
		To get the marginal distribution over given dimensions
	"""
	__MAX_ITER = 15000
	__NOISE: float = float(gpytorch.settings.min_fixed_noise.value(torch.double) or 1e-8)
	__NUM_NEIGHBOR = 32
	__slots__: tuple = ("__kernel", "__x_all", "__y_all", "__scale", "__model", "__param_indices", "__k_inv_y", "__weights_updated", "__nn", "__neighbor_indices", "__neighbors", "__neighbor_updated")

	def __init__(
		self,
		kernel: gpytorch.kernels.Kernel,
		num_neighbors: int = __NUM_NEIGHBOR
	):
		self.__kernel: gpytorch.kernels.Kernel = copy.deepcopy(kernel)
		likelihood: gpytorch.likelihoods.FixedNoiseGaussianLikelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.full((2,), __class__.__NOISE))
		self.__x_all: torch.Tensor = torch.Tensor()
		self.__y_all: torch.Tensor = torch.Tensor()
		self.__scale: float = 1.0
		self.__model: GP = GP(torch.zeros((2, pes.PHASEDIM)), torch.zeros((2,)), likelihood, self.__kernel)
		self.__param_indices: list[int] = [0]
		with torch.no_grad():
			for name, param, constraint in self.__model.named_parameters_and_constraints():
				# set parameter initial value
				if isinstance(constraint, gpytorch.constraints.Interval) and constraint.initial_value is not None:
						param[...] = constraint.initial_value.expand_as(param).clone()
				else:
					param.fill_(1.0)
				param.detach_().requires_grad = True
				# get indices for each parameter
				self.__param_indices.append(self.__param_indices[-1] + param.numel())
				# set constraint
				name_levels: list[str] = name.split(".")
				attr = self.__model
				for i in range(len(name_levels) - 1):
					attr = getattr(attr, name_levels[i])
				attr.register_constraint(name_levels[-1], NoConstraint())
		self.__k_inv_y: torch.Tensor = torch.Tensor()
		self.__weights_updated: bool = False
		self.__nn: sklearn.neighbors.NearestNeighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=num_neighbors, n_jobs=-1)
		self.__neighbor_indices: torch.Tensor = torch.Tensor()
		self.__neighbors: torch.Tensor = torch.Tensor()
		self.__neighbor_updated: bool = False

	@property
	def model(self) -> GP:
		return self.__model

	def get_training_features(self) -> torch.Tensor:
		"""
		To get the training features of the core subset

		Returns
		-------
		torch.Tensor
			The training features
		"""
		assert self.__model.train_inputs is not None
		return self.__model.train_inputs[0].detach()

	def __update_weights(self) -> None:
		"""
		To update the weights, :math:`K^{-1}y`
		"""
		if not self.__weights_updated:
			self.__k_inv_y = opt.preconditioned_lstsq_solve(self.__model.cov(self.__x_all, self.get_training_features()).to_dense(), self.__y_all)
			self.__weights_updated = True

	@property
	def k_inv_y(self) -> torch.Tensor:
		self.__update_weights()
		return self.__k_inv_y

	def __update_neighbor(self) -> None:
		"""
		To construct the neighbors
		"""
		if not self.__neighbor_updated:
			self.__nn.fit(self.__x_all.detach().numpy())
			self.__neighbor_indices: torch.Tensor = torch.from_numpy(self.__nn.kneighbors(self.__x_all.detach().numpy().reshape(-1, pes.PHASEDIM), self.__nn.get_params()["n_neighbors"] + 1, False)) # N_PT * DIM -> N_PT * (NEIGHBOR+1)
			self.__neighbor_indices = self.__neighbor_indices[self.__neighbor_indices != torch.arange(self.__x_all.shape[0]).reshape(-1, 1)].reshape(self.__x_all.shape[0], self.__nn.get_params()["n_neighbors"]) # N_PT * NEIGHBOR
			self.__neighbors: torch.Tensor = self.__x_all[self.__neighbor_indices] # N_PT * NEIGHBOR * DIM
			self.__neighbor_updated = True

	def predict(
		self,
		x_test: torch.Tensor,
		use_local_GP: bool = True
	) -> torch.Tensor:
		"""
		To predict the average estimation

		Parameters
		----------
		x_test : torch.Tensor, shape of (N, PHASEDIM)
			Validation/Test inputs
		use_local_GP : bool, optional
			Whether use local full GP on selected points or not, by default True

		Returns
		-------
		torch.Tensor, shape of (N,)
			Corresponding validation/test targets based on noise-free SR/PP mean.

		Notes
		-----
		Instance of prediction of projected process (PP)
		"""
		if use_local_GP:
			if x_test.shape == self.__x_all.shape and torch.all(x_test == self.__x_all).item():
				self.__update_neighbor()
				return (self.__model.cov(self.__x_all[:, None, :], self.__neighbors) @ opt.square_solver(self.__model.cov(self.__neighbors).to_dense(), self.__y_all[self.__neighbor_indices, None])).to_dense().reshape(self.__x_all.shape[0])
			else:
				neighbor_ind: torch.Tensor = torch.from_numpy(self.__nn.kneighbors(x_test.detach().numpy().reshape(-1, pes.PHASEDIM), return_distance=False)).reshape(x_test.shape[:-1] + (__class__.__NUM_NEIGHBOR,)) # ... * DIM -> ... * NEIGHBOR
				neighbors: torch.Tensor = self.__x_all[neighbor_ind] # x_all must be M * DIM, this gives ... * NEIGHBOR * DIM
				return (self.__model.cov(x_test[..., None, :], neighbors) @ opt.square_solver(self.__model.cov(neighbors).to_dense(), self.__y_all[neighbor_ind, None])).to_dense().reshape(x_test.shape[:-1])
			"""
				... * 1 * DIM @ ... * NEIGHBOR * DIM -> ... * 1 * NEIGHBOR
				... * NEIGHBOR * DIM -> ... * NEIGHBOR * NEIGHBOR
				[... * NEIGHBOR, 1] -> ... * NEIGHBOR * 1
				-> ... * 1 * 1 -> ...
			"""
		else:
			return (self.__model.cov(x_test, self.get_training_features()) @ self.k_inv_y).to_dense()

	def error(self) -> torch.Tensor:
		"""
		Sum of squared error between prediction vs the known labels

		Returns
		-------
		torch.Tensor
			The sum of squared prediction error
		"""
		return torch.sum((self.__y_all - self.predict(self.__x_all)) ** 2) * (self.__scale ** 2)

	def initial_parameter_search(self, print_log: bool = DEBUG_MODE) -> None:
		"""
		To search for a better initial value

		Parameters
		----------
		print_log : bool, optional
			Whether to print the log to console, by default `DEBUG_MODE`
		"""
		if not self.__neighbor_updated:
			self.__update_neighbor()
		init_params: torch.Tensor = torch.cat([constraint.transform(param) if isinstance(constraint, gpytorch.constraints.Interval) else param for _, param, constraint in self.__model.named_parameters_and_constraints()])
		best_weight: float = 1.0
		last_value: float = math.inf
		for weight in (torch.arange(20.0) + 1.0) / 10.0:
			try:
				with torch.no_grad():
					for iParam, (_, param, constraint) in enumerate(self.__model.named_parameters_and_constraints()):
						param[...] = (init_params[self.__param_indices[iParam]:self.__param_indices[iParam + 1]].reshape_as(param) * weight).detach()
						if isinstance(constraint, gpytorch.constraints.Interval):
							param[...] = constraint.inverse_transform(param).detach()
				loss = self.error()
				if print_log:
					opt.print_stuff(self.__model, loss.item(), indent=1, start_str=f"last = {last_value:.15e},", flush=True)
				if loss.item() < last_value:
					last_value = loss.item()
					best_weight = weight.item()
			except RuntimeError: # NANs
				pass
		with torch.no_grad():
			for iParam, (_, param, constraint) in enumerate(self.__model.named_parameters_and_constraints()):
				param[...] = (init_params[self.__param_indices[iParam]:self.__param_indices[iParam + 1]].reshape_as(param) * best_weight).detach()
				if isinstance(constraint, gpytorch.constraints.Interval):
					param[...] = constraint.inverse_transform(param).detach()
		opt.print_stuff(self.__model, last_value, flush=print_log)

	def train(
		self,
		print_log: bool = DEBUG_MODE,
		ftol: float | None = None,
		xtol: float | None = None
	) -> None:
		"""
		To train the parameters

		Parameters
		----------
		print_log : bool, optional
			Whether to print the log to console, by default `DEBUG_MODE`
		ftol : float | None, optional
			Tolerance of function values, by default None
		xtol : float | None, optional
			Tolerance of gradient / parameter values, by default None
		"""
		def loss_func(cov: gpytorch.kernels.Kernel) -> torch.Tensor:
			"""
			Loss function

			Parameters
			----------
			cov: gpytorch.kernels.Kernel
				Covariance function

			Returns
			-------
			torch.Tensor
				Squared sum of difference between local GP prediction vs known labels
			"""
			return torch.sum((self.__y_all.reshape(-1) - (cov(self.__x_all[:, None, :], self.__neighbors) @ opt.square_solver(cov(self.__neighbors).to_dense(), self.__y_all[self.__neighbor_indices, None])).to_dense().reshape(-1)) ** 2) * (self.__scale ** 2)

		self.__update_neighbor()
		self.__weights_updated = False
		# train model
		self.__model.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.full(self.get_training_features().shape[:-1], __class__.__NOISE))
		self.__model.train()
		self.__model.likelihood.train()
		loss: torch.Tensor = loss_func(self.__model.cov)
		last_value: float = torch.inf
		learning_rate: float = -1.0
		num_iter: int = 0
		grad_opt: opt.GradientOptimization = opt.GradientOptimization(self.__model, self.__param_indices, loss_func, print_log, GP.COV_NAME, ftol, xtol)
		non_grad_opt: opt.NonGradientOptimization | None = None
		while num_iter < __class__.__MAX_ITER:
			num_iter += 1
			if non_grad_opt is None:
				learning_rate, loss, to_end = grad_opt(learning_rate, loss)
				if to_end:
					break
				if math.isnan(learning_rate):
					non_grad_opt = opt.NonGradientOptimization(self.__model, loss_func, print_log, GP.COV_NAME, ftol, xtol)
					if non_grad_opt.converged:
						break
			else:
				loss = torch.tensor(non_grad_opt())
				if non_grad_opt.converged:
					break
			if print_log or num_iter % (__class__.__MAX_ITER // 100) == 0:
				opt.print_stuff(
					self.__model,
					loss.item(),
					learning_rate,
					1,
					start_str=f"Iter {num_iter} - last = {last_value:.15e},",
					flush=True
				)
			last_value = loss.item()
		if num_iter == __class__.__MAX_ITER:
			print("Stop: Maximum number of iterations has been exceeded.")
		opt.print_stuff(
			self.__model,
			loss.item(),
			learning_rate,
			start_str=f"Iter {num_iter} - last = {last_value:.15e},",
			flush=print_log
		)
		# turn to predict mode
		self.__model.eval()
		self.__model.likelihood.eval()

	def update(
		self,
		x_all: torch.Tensor,
		y_all: torch.Tensor,
		scale: float,
		num_points: int
	) -> None:
		"""
		To update the training features and labels of the model

		Parameters
		----------
		x_all : torch.Tensor, of shape (N_ALL_PT, PHASEDIM)
			All training inputs
		y_all : torch.Tensor, of shape (N_ALL_PT)
			All training targets
		scale : float
			The scaling factor to increase
		num_points : int
			The number of points located at the front of all points that is used as the subset
		"""
		assert x_all.shape[-1] == pes.PHASEDIM and x_all.numel() == y_all.numel() * pes.PHASEDIM
		self.__x_all = x_all.reshape(-1, pes.PHASEDIM).clone().detach()
		self.__y_all = y_all.reshape(-1).clone().detach()
		self.__scale = scale
		self.__model.set_train_data(self.__x_all[:num_points].detach(), self.__y_all[:num_points].detach(), False)
		self.__weights_updated = False
		self.__neighbor_updated = False

	def get_marginal(self, dimensions: list[int], x_test: torch.Tensor) -> torch.Tensor:
		"""
		To get the marginal distribution of current gaussian process regression

		Parameters
		----------
		dimensions : list[int]
			The dimensions to be kept, must not have any repeat
		x_test : torch.Tensor, shape of (N, len(dimensions))
			Validation/Test inputs

		Returns
		-------
		torch.Tensor, shape of (N,)
			Corresponding validation/test targets based on noise-free SR/PP mean.
		"""
		marginal_kernel: gpytorch.kernels.Kernel = gpytorch.kernels.RBFKernel(len(dimensions), lengthscale_constraint=NoConstraint(self.__model.cov.lengthscale.reshape(1, pes.PHASEDIM)[:, dimensions]))
		prefactor: float = np.sqrt((2.0 * torch.pi) ** (pes.PHASEDIM - len(dimensions))) * self.__model.cov.lengthscale[:, [i for i in range(pes.PHASEDIM) if i not in dimensions]].prod().item()
		return prefactor * marginal_kernel(x_test, self.get_training_features()[:, dimensions]).to_dense() @ self.k_inv_y


class GPRPredictors:
	"""
	Combination of single predictors

	Parameters
	----------
	kernel : gpytorch.kernels.Kernel, optional
		The kernel of predictors, by default gpytorch.kernels.RBFKernel(pes.PHASEDIM)

	Methods
	-------
	__check_predictor(predictor)
		To check if the predictor could be used for training / predicting
	update(x_all, y_all, num_pt, scale)
		To update the training inputs and targets, as well as the rescale factor
	train()
		To train each predictor
	predict(x_input, ElementIndex)
		To predict test targets based on input and corresponding density matrix element
	get_marginal(dimensions, x_input, ElementIndex)
		To get the marginal distribution over given dimensions
	print(f)
		To print parameters to file
	"""
	@staticmethod
	def __check_predictor(predictor: SinglePredictor) -> bool:
		"""
		To check if the predictor could be used for training / predicting

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
		return isinstance(predictor.model.train_targets, torch.Tensor) and not torch.all(predictor.model.train_targets == 0)

	__slots__: tuple = ("__predictors", "__scale", "__initial_train")

	def __init__(self, parameter_initial_values: npt.NDArray[np.double] | None = None):
		initial_value: torch.Tensor | None = None
		if parameter_initial_values is not None:
			try:
				initial_value = torch.broadcast_to(torch.from_numpy(parameter_initial_values.reshape(-1)), (pes.NUM_ELM, pes.PHASEDIM))
			except RuntimeError: # unable to broadcast to the given shape
				pass
		self.__predictors: list[SinglePredictor] = [SinglePredictor(gpytorch.kernels.RBFKernel(pes.PHASEDIM, lengthscale_constraint=NoConstraint(iv))) for iv in (initial_value if initial_value is not None else [None] * pes.NUM_ELM)]
		self.__scale: npt.NDArray[np.double] = np.ones(pes.NUM_ELM, np.double)
		self.__initial_train: npt.NDArray[np.bool_] = np.ones(pes.NUM_ELM, np.bool_)

	def __getitem__(self, ElementIndex: int) -> SinglePredictor:
		"""
		To get corresponding predictor

		Parameters
		----------
		ElementIndex : int
			Index of the predictor

		Returns
		-------
		SinglePredictor
			Predictor corresponding to the index in density supervector
		"""
		assert 0 <= ElementIndex < pes.NUM_ELM
		return self.__predictors[ElementIndex]

	def update(
		self,
		x_all: list[npt.NDArray[np.double]],
		y_all: list[npt.NDArray[np.cdouble]],
		num_points: int | npt.NDArray[np.int_],
		scale: npt.NDArray[np.double]
	) -> None:
		"""
		To update the training inputs and targets, as well as the rescale factor

		Parameters
		----------
		x_all : list[npt.NDArray[np.double]], len of NUM_TRIG, each of shape (num_points * (1 + NUM_XTR_RATIO), PHASEDIM)
			All training inputs
		y_all : list[npt.NDArray[np.cdouble]], len of NUM_TRIG, each of shape (num_points * (1 + NUM_XTR_RATIO))
			All training targets
		num_points : int | npt.NDArray[np.int_], shape of (NUM_TRIG,)
			The number of points located at the front of all points that is used as the subset
		scale : npt.NDArray[np.double], shape of (NUM_ELM,)
			The rescale factor
		"""
		if isinstance(num_points, int):
			num_points = np.full(pes.NUM_TRIG, num_points, np.int_)
		self.__scale[...] = scale
		for iElement, pred in enumerate(self.__predictors):
			RowIndex: int = iElement // pes.NUM_PES
			ColIndex: int = iElement % pes.NUM_PES
			TrilIndex: int = pes.flatten_tril_index[RowIndex, ColIndex]
			pred.update(
				torch.from_numpy(x_all[TrilIndex]),
				torch.from_numpy(y_all[TrilIndex].real if RowIndex <= ColIndex else y_all[TrilIndex].imag),
				self.__scale[iElement],
				num_points[TrilIndex]
			)

	def train(
		self,
		print_log: bool = DEBUG_MODE,
		ftol: float | None = None,
		xtol: float | None = None
	) -> None:
		"""
		To train each predictor

		Parameters
		----------
		print_log : bool, optional
			Whether to print the log to console, by default `DEBUG_MODE`
		ftol : float | None, optional
			Tolerance of function values, by default None
		xtol : float | None, optional
			Tolerance of gradient / parameter values, by default None
		"""
		for iElement in range(pes.NUM_ELM):
			if __class__.__check_predictor(self.__predictors[iElement]):
				if self.__initial_train[iElement]:
					self.__initial_train[iElement] = False
					self.__predictors[iElement].initial_parameter_search(print_log)
				if print_log:
					print("Training " + utility.get_RI_label(iElement))
				self.__predictors[iElement].train(print_log, ftol, xtol)

	def predict(self, x_input: npt.NDArray[np.double], ElementIndex: int, use_local_gp: bool = True) -> npt.NDArray[np.cdouble]:
		"""
		To predict test targets based on input and corresponding density matrix element

		Parameters
		----------
		x_input : npt.NDArray[np.double], shape of (..., PHASEDIM)
			Test inputs
		ElementIndex : int
			Index of the element
		use_local_GP : bool, optional
			Whether use local full GP on selected points or not, by default True

		Returns
		-------
		npt.NDArray[np.cdouble], shape of (...)
			Density of the element of all test inputs
		"""
		def call_single_predictor(pred: SinglePredictor, x_test: torch.Tensor) -> npt.NDArray[np.double]:
			"""
			To do prediction of a single predictor

			Parameters
			----------
			pred : SinglePredictor
				The predictor
			x_test : torch.Tensor, shape of (N, PHASEDIM)
				Test inputs

			Returns
			-------
			npt.NDArray[np.double], shape of (N,)
				Test targets by the predictor
			"""
			if __class__.__check_predictor(pred):
				return pred.predict(x_test, use_local_gp).detach().numpy()
			else:
				return np.zeros(x_test.shape[0], np.double)

		assert x_input.shape[-1] == pes.PHASEDIM and 0 <= ElementIndex < pes.NUM_ELM
		x_test: torch.Tensor = torch.from_numpy(x_input.reshape(-1, pes.PHASEDIM))
		RowIndex: int = ElementIndex // pes.NUM_PES
		ColIndex: int = ElementIndex % pes.NUM_PES
		result: npt.NDArray[np.cdouble] = np.empty(x_test.shape[0], np.cdouble)
		if RowIndex == ColIndex:
			result.real = call_single_predictor(self.__predictors[ElementIndex], x_test)
			result.imag = 0
		elif RowIndex > ColIndex:
			result.real = call_single_predictor(self.__predictors[ColIndex * pes.NUM_PES + RowIndex], x_test)
			result.imag = call_single_predictor(self.__predictors[ElementIndex], x_test)
		else: # RowIndex < ColIndex
			result.real = call_single_predictor(self.__predictors[ElementIndex], x_test)
			result.imag = -call_single_predictor(self.__predictors[ColIndex * pes.NUM_PES + RowIndex], x_test)
		return result.reshape(x_input.shape[:-1])

	def get_marginal(
		self,
		dimensions: int | collections.abc.Iterable[int],
		x_input: npt.NDArray[np.double],
		ElementIndex: int
	) -> npt.NDArray[np.cdouble]:
		"""
		To get the marginal distribution of current gaussian process regressions

		Parameters
		----------
		dimensions : int | collections.abc.Iterable[int]
			The dimensions to be kept
		x_input : npt.NDArray[np.double], shape of (..., len(dimensions))
			Test inputs
		ElementIndex : int
			Index of the element

		Returns
		-------
		npt.NDArray[np.double], shape of (N,)
			Marginal distribution on the inputs
		"""
		def call_single_predictor(pred: SinglePredictor, dims: list[int], x_test: torch.Tensor) -> npt.NDArray[np.double]:
			"""
			To do prediction of a single predictor

			Parameters
			----------
			pred : SinglePredictor
				The predictor
			dims : list[int]
				The dims to be kept
			x_input : torch.Tensor, shape of (N, len(dimensions))
				Test inputs

			Returns
			-------
			npt.NDArray[np.double], shape of (N,)
				Test targets by the predictor
			"""
			if __class__.__check_predictor(pred):
				return pred.get_marginal(dims, x_test).detach().numpy()
			else:
				return np.zeros(x_test.shape[0], np.double)

		if isinstance(dimensions, int):
			dimensions = [dimensions]
		else:
			dimensions = list(set(dimensions)) # remove duplicate
		assert all(0 <= dim <= pes.PHASEDIM for dim in dimensions)
		assert x_input.shape[-1] == len(dimensions) and 0 <= ElementIndex < pes.NUM_ELM
		x_test: torch.Tensor = torch.from_numpy(x_input.reshape(-1, len(dimensions)))
		RowIndex: int = ElementIndex // pes.NUM_PES
		ColIndex: int = ElementIndex % pes.NUM_PES
		result: npt.NDArray[np.cdouble] = np.empty(x_test.shape[0], np.cdouble)
		if RowIndex == ColIndex:
			result.real = call_single_predictor(self.__predictors[ElementIndex], dimensions, x_test)
			result.imag = 0
		elif RowIndex > ColIndex:
			result.real = call_single_predictor(self.__predictors[ColIndex * pes.NUM_PES + RowIndex], dimensions, x_test)
			result.imag = call_single_predictor(self.__predictors[ElementIndex], dimensions, x_test)
		else: # RowIndex < ColIndex
			result.real = call_single_predictor(self.__predictors[ElementIndex], dimensions, x_test)
			result.imag = -call_single_predictor(self.__predictors[ColIndex * pes.NUM_PES + RowIndex], dimensions, x_test)
		return result.reshape(x_input.shape[:-1])

	def print(self, f: io.TextIOWrapper) -> None:
		"""
		To print the parameters to file

		Parameters
		----------
		f : io.TextIOWrapper
			The file to save the parameters
		"""
		for predictor in self.__predictors:
			np.savetxt(f, predictor.model.cov.lengthscale.detach().numpy().reshape(1, -1))
		print("\n", file=f)
