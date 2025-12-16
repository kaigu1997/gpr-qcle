r"""gp
==
This module provides support for gaussian process (gp) regression.
"""
import collections.abc
import copy
import os
import math
import sys
import typing

import gpytorch
import gpytorch.constraints
import linear_operator
import numpy as np
import numpy.typing as npt
import torch

sys.path.append(os.path.dirname(__file__))

import pes
import utility

torch.set_default_dtype(torch.double)
torch.manual_seed(0)
DEBUG_MODE: bool = True


class GP(gpytorch.models.ExactGP):
	r"""Gaussian process regression

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
	cov()
		The kernel function
	forward(x)
		The implementation of GPR
	"""
	COV_NAME: typing.Final = "cov"
	__slots__: tuple = ("__mean", "__cov")

	def __init__(
		self,
		x: torch.Tensor,
		y: torch.Tensor,
		likelihood: gpytorch.likelihoods.GaussianLikelihood | gpytorch.likelihoods.FixedNoiseGaussianLikelihood,
		kernel: gpytorch.kernels.Kernel
	):
		super().__init__(x, y, likelihood)
		self.__mean: gpytorch.means.Mean = gpytorch.means.ZeroMean()
		self.__cov: gpytorch.kernels.Kernel = kernel

	@property
	def cov(self) -> gpytorch.kernels.Kernel:
		r"""The kernel function

		Returns
		-------
		gpytorch.kernels.Kernel
			`gpytorch` kernel which is callable
		"""
		return self.__cov

	def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
		r"""The implementation of GPR

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
	r"""No constraint on the parameter, the value could be any float value

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
		r"""The official string representation of an object

		Returns
		-------
		str
			The name of the class with an empty parenthesis
		"""
		return __class__.__name__ + "()"

	def transform(self, tensor: torch.Tensor) -> torch.Tensor:
		r"""To transform the raw value to the actual value

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
		r"""To transform the actual value to the raw value

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
	r"""The instantiation of gaussian process predictor

	Parameters
	----------
	kernel : gpytorch.kernels.Kernel
		The kernel to use

	Attributes
	----------
	MAX_ITER : typing.Literal[50000]
		Maximum iteration of optimization
	FTOL : float
		Absolute and relative tolerance of function in optimization
	GTOL : float
		Tolerance of gradient in optimization
	NOISE: float
		Extra noise term added for numerical stability in matrix inversion

	Methods
	-------
	model()
		The exact GPR model
	x_all()
		All features for approximate method
	get_training_features()
		To get the training features of the core subset
	k_inv_y()
		To get the weights, :math:`K^{-1}y`
	predict(x_test)
		To predict the average
	predict_using_given_labels(x_test, y_all_in)
		To predict, but use the provided label instead
	predict_derivative_over_input(x_test)
		To calculate the derivatives over input only
	predict_derivatives(x_test)
		To calculate the derivatives
	update(x_all, y_all_, scale, num_points)
		To update the training features and labels of the model
	update_param(param_change)
		To update parameters with given value
	error()
		To get the error by comparing label with prediction
	train()
		To train the parameters
	"""
	MAX_ITER: typing.Final = 15000
	FTOL: typing.Final = 2.2204460492503131e-09
	GTOL: typing.Final = 1e-5
	NOISE: typing.Final = float(gpytorch.settings.min_fixed_noise.value(torch.double) or 1e-8)
	__slots__: tuple = ("__kernel", "__x_all", "__y_all", "__scale", "__model", "__model_param", "__k_inv_y", "__weights_updated")

	def __init__(self, kernel: gpytorch.kernels.Kernel):
		self.__kernel: gpytorch.kernels.Kernel = copy.deepcopy(kernel)
		likelihood: gpytorch.likelihoods.FixedNoiseGaussianLikelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.full((2,), SinglePredictor.NOISE))
		self.__x_all: torch.Tensor = torch.Tensor()
		self.__y_all: torch.Tensor = torch.Tensor()
		self.__scale: float = 1.0
		self.__model: GP = GP(torch.zeros((2, pes.PHASEDIM)), torch.zeros((2,)), likelihood, self.__kernel)
		self.__model_param: dict[str, torch.Tensor] = copy.deepcopy(self.__model.state_dict())
		self.__k_inv_y: torch.Tensor = torch.Tensor()
		self.__weights_updated: bool = False

	@property
	def model(self) -> GP:
		r"""The exact GPR model

		Returns
		-------
		GP
			The model
		"""
		return self.__model

	@property
	def x_all(self) -> torch.Tensor:
		r"""All features for approximate method

		Returns
		-------
		torch.Tensor, dtype of `torch.double`, shape of (N, PHASEDIM)
			Features
		"""
		return self.__x_all

	def get_training_features(self) -> torch.Tensor:
		r"""To get the training features of the core subset

		Returns
		-------
		torch.Tensor
			The training features
		"""
		assert self.__model.train_inputs is not None
		return self.__model.train_inputs[0]

	def __update_weights(self) -> None:
		r"""To update the weights, :math:`K^{-1}y`
		"""
		if not self.__weights_updated:
			self.__k_inv_y = (linear_operator.utils.stable_pinverse(self.__model.cov(self.__x_all, self.get_training_features()).to_dense()) @ self.__y_all).to_dense()
			self.__weights_updated = True

	@property
	def k_inv_y(self) -> torch.Tensor:
		r"""To get the weights, :math:`K^{-1}y`

		Returns
		-------
		torch.Tensor
			Weights, :math:`K^{-1}y`
		"""
		self.__update_weights()
		return self.__k_inv_y

	def predict(self, x_test: torch.Tensor) -> torch.Tensor:
		r"""Instance of prediction of subset of regressor (SR) / projected process (PP)

		Parameters
		----------
		x_test : torch.Tensor, shape of (N, PHASEDIM)
			Validation/Test inputs

		Returns
		-------
		torch.Tensor, shape of (N,)
			Corresponding validation/test targets based on noise-free SR/PP mean.
		"""
		return (self.__model.cov(x_test, self.get_training_features()) @ self.k_inv_y).to_dense()

	def predict_using_given_labels(self, x_test: torch.Tensor, y_all_in: torch.Tensor) -> torch.Tensor:
		r"""To predict, but use the provided label instead

		Parameters
		----------
		x_test : torch.Tensor, shape of (N, PHASEDIM)
			Validation/Test inputs
		y_all_in : torch.Tensor, shape of (N_ALL,)
			Temporary labels, could be dy/dt for derivative calculation

		Returns
		-------
		torch.Tensor, shape of (N,)
			Corresponding validation/test targets
		"""
		return (self.__model.cov(x_test, self.get_training_features()) @ linear_operator.utils.stable_pinverse(self.__model.cov(self.__x_all, self.get_training_features()).to_dense()) @ y_all_in).to_dense()

	def predict_derivative_over_input(self, x_test: torch.Tensor) -> torch.Tensor:
		r"""To calculate the derivatives over input only

		Parameters
		----------
		x_test : torch.Tensor, dtype of `torch.double`, shape of (..., PHASEDIM)
			Test inputs

		Returns
		-------
		torch.Tensor, of dtype `torch.double` and shape (..., PHASEDIM);
			Derivative of each prediction over each input
		"""
		assert self.__model.train_inputs is not None
		original_shape: typing.Final[tuple] = x_test.shape
		# start with gradient enabled
		with torch.no_grad():
			x_test = x_test.reshape(-1, pes.PHASEDIM).detach().requires_grad_()
		pred: typing.Final[torch.Tensor] = self.predict(x_test) # (N_INPUT,)
		return torch.autograd.grad(pred, x_test, torch.ones_like(pred), True, False, True, True, False, True)[0].detach() # (N_INPUT, PHASEDIM)

	def predict_derivatives(self, x_test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		r"""To calculate the derivatives

		Parameters
		----------
		x_test : torch.Tensor, dtype of `torch.double`, shape of (..., PHASEDIM)
			Test inputs

		Returns
		-------
		tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
			Predictions, of dtype `torch.double` and shape (...);

			Derivative of each prediction over each input, of dtype `torch.double` and shape (..., PHASEDIM);

			Derivative of predictions over the subset of points, of dtype `torch.double` and shape (..., N_SUBSET, PHASEDIM);

			Derivative of predictions over all the points, of dtype `torch.double` and shape (..., N_TOTAL, PHASEDIM);

			Derivative of predictions over the parameters, of dtype `torch.double` and shape (..., N_PARAM).
		"""
		assert self.__model.train_inputs is not None
		original_shape: typing.Final[tuple] = x_test.shape
		# start with gradient enabled
		with torch.no_grad():
			x_test = x_test.reshape(-1, pes.PHASEDIM).detach().requires_grad_() # even if x_test is x_all or subset, this will make them different leaves
			self.__x_all.requires_grad = True
			self.get_training_features().requires_grad = True
			self.__weights_updated = False
		pred: typing.Final[torch.Tensor] = self.predict(x_test) # (N_INPUT,)
		grad_input: typing.Final[torch.Tensor] = torch.autograd.grad(pred, x_test, torch.ones_like(pred), True, False, True, True, False, True)[0] # (N_INPUT, PHASEDIM)
		grad_subset: typing.Final[torch.Tensor] = torch.autograd.grad(pred, self.get_training_features(), torch.eye(pred.numel()), True, False, True, True, True, True)[0] # (N_INPUT, N_SUB, PHASEDIM)
		grad_fullset: typing.Final[torch.Tensor] = torch.autograd.grad(pred, self.__x_all, torch.eye(pred.numel()), True, False, True, True, True, True)[0] # (N_INPUT, N_TOTAL, PHASEDIM)
		grad_param: typing.Final[torch.Tensor] = torch.cat([torch.autograd.grad(pred, param, torch.eye(pred.numel()), True, False, True, True, True, True)[0].reshape(pred.numel(), -1) for param in self.__model.parameters()], -1) # (N_INPUT, N_PARAM)
		# disable gradient for fast calculation
		with torch.no_grad():
			self.__x_all.requires_grad = False
			self.get_training_features().requires_grad = False
		return pred.detach().reshape(original_shape[:-1]),\
			grad_input.reshape(original_shape),\
			grad_subset.reshape(original_shape[:-1] + self.get_training_features().shape),\
			grad_fullset.reshape(original_shape[:-1] + self.__x_all.shape),\
			grad_param.reshape(*original_shape[:-1], -1)

	def update(
		self,
		x_all: torch.Tensor,
		y_all: torch.Tensor,
		scale: float,
		num_points: int
	) -> None:
		r"""To update the training features and labels of the model

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

	def update_param(self, param_change: torch.Tensor) -> None:
		r"""To update parameters with given value

		Parameters
		----------
		param_change : torch.Tensor
			The change concatenated for all parameters
		"""
		length = 0
		for param in self.__model.parameters():
			param = (param + param_change[length:length + param.numel()].reshape(param.shape))
			length += param.numel()

	def error(self, use_weight: bool = True) -> torch.Tensor:
		r"""Error function of subset of regressor (SR) / projected process (PP)

		This function gives the sum of squared error

		Returns
		-------
		torch.Tensor
			The sum of squared prediction error
		"""
		if use_weight:
			return torch.sum((self.__y_all - self.predict(self.__x_all)) ** 2) * (self.__scale ** 2)
		else:
			kmn: torch.Tensor = self.__model.cov(self.__x_all, self.get_training_features()).to_dense()
			return torch.sum((self.__y_all - (kmn @ (linear_operator.utils.stable_pinverse(kmn) @ self.__y_all)).to_dense()) ** 2) * (self.__scale ** 2)

	def train(self, print_log: bool = DEBUG_MODE) -> None:
		r"""To train the parameters

		Parameters
		----------
		print_log : bool, optional
			Whether to print the log to console, by default `DEBUG_MODE`
		"""
		def print_model(model: gpytorch.models.ExactGP, print_grad: bool = False) -> None:
			r"""To print the parameters of the model

			Parameters
			----------
			model : gpytorch.models.ExactGP
				Gaussian process model, containing mean and covariances and their parameters
			print_grad : bool, optional
				Whether to print the gradient or not, by default False
			"""
			def make_tensor_printable(t: torch.Tensor) -> float | npt.NDArray[np.double]:
				r"""To transform a torch Tensor to read-friendly form

				If the tensor contains only 1 element, return the element;
				otherwise, return the flattened numpy array

				Parameters
				----------
				t : torch.Tensor
					The tensor

				Returns
				-------
				float | npt.NDArray[np.double]
					Return the only element or the flattened array
				"""
				if t.dim() == 0 or np.prod(t.shape) == 1:
					return t.item()
				else:
					return t.detach().numpy().ravel()

			fmt: str = "Parameter name: {0:42} value = {1}"
			fmt_grad: str = fmt + " grad = {2}"
			for param_name, param, constraint in model.named_parameters_and_constraints():
				if print_grad and param.grad is not None:
					print(fmt_grad.format(param_name, make_tensor_printable(param), make_tensor_printable(param.grad)))
				else:
					print(fmt.format("".join(param_name.split("raw_")), make_tensor_printable(constraint.transform(param) if isinstance(constraint, gpytorch.constraints.Interval) else param)))

		def get_lr(optimizer: torch.optim.Optimizer) -> float:
			r"""To get the learning rate of the optimizer

			Parameters
			----------
			optimizer : torch.optim.Optimizer
				The optimizer, which contains learning rate

			Returns
			-------
			float
				Learning rate
			"""
			return optimizer.param_groups[0]["lr"]

		def print_stuff(
			loss: torch.Tensor,
			optimizer: torch.optim.Optimizer,
			model: gpytorch.models.ExactGP,
			print_grad: bool = False,
			extra_str="\t"
		) -> None:
			r"""To print all stuffs needed

			Parameters
			----------
			loss : torch.Tensor
				Loss by error function
			optimizer : torch.optim.Optimizer
				The optimizer, containing learning rate
			model : gpytorch.models.ExactGP
				Gaussian process model, containing mean and covariances and their parameters
			print_grad : bool, optional
				Whether to print gradient in the model or not, by default False
			extra_str : str, optional
				An extra string added at the front, by default "\t"
			"""
			print(extra_str + "loss = {:.15e}, lr = {}".format(loss.item(), get_lr(optimizer)))
			print_model(model, print_grad)

		self.__weights_updated = False
		assert isinstance(self.__model.train_targets, torch.Tensor)
		# train model
		self.__model.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.full((self.get_training_features().shape[0],), SinglePredictor.NOISE))
		self.__model.train()
		self.__model.likelihood.train()
		self.__model.load_state_dict(self.__model_param)
		# print_model(self.__model)
		finish_early: bool = False
		optimizer: torch.optim.Optimizer = torch.optim.Rprop(self.__model.parameters(), lr=1.0)
		optimizer.zero_grad()
		loss: torch.Tensor = self.error(False)
		loss.backward()
		last_value: float = loss.item()
		print_stuff(loss, optimizer, self.__model, True, "Init")
		for i in range(1, SinglePredictor.MAX_ITER + 1):
			# adjust lr
			old_prm: dict[str, torch.Tensor] = copy.deepcopy(self.__model.state_dict())
			optimizer.step()
			loss = self.error(False)
			if print_log:
				print_stuff(loss, optimizer, self.__model, True)
			if loss < last_value:
				optimizer = optimizer.__class__(self.__model.parameters(), lr=get_lr(optimizer) * 2.0)
				if print_log:
					print("loss < last_value")
					print_stuff(loss, optimizer, self.__model, True)
			else:
				if print_log:
					print("loss > last_value")
				while loss >= last_value:
					last_loop_value: float = loss.item()
					self.__model.load_state_dict(old_prm)
					optimizer = optimizer.__class__(self.__model.parameters(), lr=get_lr(optimizer) / 2.0)
					optimizer.step()
					loss = self.error(False)
					if print_log:
						print_stuff(loss, optimizer, self.__model, True)
					if last_loop_value == loss.item():
						print("No stepping forward")
						# no stepping forward, but still larger than last, meaning last is the best
						self.__model.load_state_dict(old_prm)
						loss = self.error(False)
						break
			if i % (SinglePredictor.MAX_ITER // 100) == 0:
				print("Iter {} - Loss: {:.15e} - lr: {}".format(i, loss.item(), get_lr(optimizer)))
				print_model(self.__model, True)
				print_model(self.__model)
			# stopping criteria
			if (last_value - loss.item()) / max(abs(last_value), abs(loss.item()), 1.0) < SinglePredictor.FTOL:
				finish_early = True
				print("Convergence: |f_i - f_{i+1}| <= FTOL")
				print("Iter {} - Loss: {:.15e} - lr: {}".format(i, loss.item(), get_lr(optimizer)))
				break
			if np.sqrt(sum(torch.sum(param.grad ** 2).item() if param.grad is not None else 0.0 for param in self.__model.parameters())) < SinglePredictor.GTOL:
				finish_early = True
				print("Convergence: |Gradient| <= GTOL")
				print("Iter {} - Loss: {:.15e} - lr: {}".format(i, loss.item(), get_lr(optimizer)))
				break
			optimizer = optimizer.__class__(self.__model.parameters(), lr=get_lr(optimizer))
			optimizer.zero_grad()
			last_value = loss.item()
			loss.backward()
			if print_log:
				print_stuff(loss, optimizer, self.__model, True, "\tlast = {}, ".format(last_value))
		if not finish_early:
			print("Iter {} - Loss: {:.15e} - lr: {}".format(SinglePredictor.MAX_ITER, last_value, [param["lr"] for param in optimizer.param_groups]))
			print("Stop: Total No. iterations reached limit.")
		print_model(self.__model)
		print("", flush=True)
		self.__model_param = copy.deepcopy(self.__model.state_dict())

	def get_marginal(self, dimensions: collections.abc.Sequence[int], x_test: torch.Tensor) -> torch.Tensor:
		r"""To get the marginal distribution of current gaussian process regression

		Parameters
		----------
		dimensions : collections.abc.Sequence[int]
			The dimensions to be kept, must not have any repeat
		x_test : torch.Tensor, shape of (N, len(dimensions))
			Validation/Test inputs

		Returns
		-------
		torch.Tensor, shape of (N,)
			Corresponding validation/test targets based on noise-free SR/PP mean.
		"""
		marginal_kernel: typing.Final[gpytorch.kernels.RBFKernel] = gpytorch.kernels.RBFKernel(len(dimensions), lengthscale_constraint=NoConstraint(self.__model.cov.lengthscale.reshape(1, pes.PHASEDIM)[:, dimensions]))
		prefactor: typing.Final[float] = np.sqrt((2.0 * torch.pi) ** (pes.PHASEDIM - len(dimensions))) * self.__model.cov.lengthscale[:, [i for i in range(pes.PHASEDIM) if i not in dimensions]].prod().item()
		return prefactor * marginal_kernel(x_test, self.get_training_features()[:, dimensions]).to_dense() @ self.__k_inv_y


class GPRPredictors:
	r"""Combination of single predictors

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
	print(f)
		To print hyperparameters to file
	"""
	@staticmethod
	def __check_predictor(predictor: SinglePredictor) -> bool:
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
		return isinstance(predictor.model.train_targets, torch.Tensor) and not torch.all(predictor.model.train_targets == 0).item() # pyright: ignore[reportArgumentType, reportCallIssue]

	__slots__: tuple = ("__predictors", "__initial_train")

	def __init__(self, kernel: gpytorch.kernels.Kernel = gpytorch.kernels.RBFKernel(pes.PHASEDIM)):
		self.__predictors: list[SinglePredictor] = [SinglePredictor(kernel) for _ in range(pes.NUM_ELM)]

	def __getitem__(self, ElementIndex: int) -> SinglePredictor:
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
		assert 0 <= ElementIndex < pes.NUM_ELM
		return self.__predictors[ElementIndex]

	def __combine_to_complex[**P](
		self,
		x_input: npt.NDArray[np.double],
		RowIndex: int,
		ColIndex: int,
		call_single_predictor: collections.abc.Callable[typing.Concatenate[SinglePredictor, torch.Tensor, P], collections.abc.Iterable[npt.NDArray[np.double]]],
		*args: P.args,
		**kwargs: P.kwargs
	) -> tuple[npt.NDArray[np.cdouble], ...]:
		r"""To combine results from single predictor into complex arrays

		Parameters
		----------
		x_input : npt.NDArray[np.double], shape of (..., PHASEDIM)
			Test inputs
		RowIndex : int
			Index of row of the element in density matrix
		ColIndex : int
			Index of column of the element in density matrix
		call_single_predictor : collections.abc.Callable[typing.Concatenate[SinglePredictor, torch.Tensor, P], collections.abc.Iterable[npt.NDArray[np.double]]]
			The function that takes the single predictor and generates some Tensor (prediction, derivatives, marginals, etc)

		Returns
		-------
		tuple[npt.NDArray[np.cdouble], ...]
			Combined complex arrays from single predictor
		"""
		assert x_input.shape[-1] == pes.PHASEDIM and 0 <= RowIndex < pes.NUM_PES and 0 <= ColIndex < pes.NUM_PES
		x_test: typing.Final[torch.Tensor] = torch.from_numpy(x_input.reshape(-1, pes.PHASEDIM))
		if RowIndex == ColIndex:
			return tuple(item.reshape(x_input.shape[:-1] + item.shape[1:]) + 0.j for item in call_single_predictor(self.__predictors[RowIndex * pes.NUM_PES + ColIndex], x_test, *args, **kwargs))
		elif RowIndex > ColIndex:
			return tuple((real + 1.j * imag).reshape(x_input.shape[:-1] + real.shape[1:]) for real, imag in zip(call_single_predictor(self.__predictors[ColIndex * pes.NUM_PES + RowIndex], x_test, *args, **kwargs), call_single_predictor(self.__predictors[RowIndex * pes.NUM_PES + ColIndex], x_test, *args, **kwargs)))
		else: # RowIndex < ColIndex
			return tuple((real - 1.j * imag).reshape(x_input.shape[:-1] + real.shape[1:]) for real, imag in zip(call_single_predictor(self.__predictors[RowIndex * pes.NUM_PES + ColIndex], x_test, *args, **kwargs), call_single_predictor(self.__predictors[ColIndex * pes.NUM_PES + RowIndex], x_test, *args, **kwargs)))

	def update(
		self,
		x_all: list[npt.NDArray[np.double]],
		y_all: list[npt.NDArray[np.cdouble]],
		num_points: int | npt.NDArray[np.int_],
		scale: npt.NDArray[np.double]
	) -> None:
		r"""To update the training inputs and targets, as well as the rescale factor

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
		for iElement, pred in enumerate(self.__predictors):
			RowIndex: int = iElement // pes.NUM_PES
			ColIndex: int = iElement % pes.NUM_PES
			TrilIndex: int = pes.flatten_tril_index[RowIndex, ColIndex]
			pred.update(
				torch.from_numpy(x_all[TrilIndex]),
				torch.from_numpy(y_all[TrilIndex].real if RowIndex <= ColIndex else y_all[TrilIndex].imag),
				scale[iElement],
				num_points[TrilIndex]
			)

	def evolve_parameter(
		self,
		mass: npt.NDArray[np.double],
		dt: float
	) -> None:
		r"""To evolve parameters based on QCLE and GPR

		Parameters
		----------
		mass : npt.NDArray[np.double]
			_description_
		dt : float
			_description_
		"""
		predict_derivative_over_input: typing.Final[collections.abc.Callable[[npt.NDArray[np.double], int, int], npt.NDArray[np.cdouble]]] = lambda x_input, RowIndex, ColIndex: self.__combine_to_complex(
			x_input,
			RowIndex,
			ColIndex,
			lambda pred, x_test: (pred.predict_derivative_over_input(x_test).detach().numpy(),) if GPRPredictors.__check_predictor(pred) else (np.zeros(x_test.shape),)
		)[0]
		predict_derivatives: typing.Final[collections.abc.Callable[[npt.NDArray[np.double], int, int], tuple[npt.NDArray[np.cdouble], ...]]] = lambda x_input, RowIndex, ColIndex: self.__combine_to_complex(
			x_input,
			RowIndex,
			ColIndex,
			lambda pred, x_test: tuple(t.detach().numpy() for t in pred.predict_derivatives(x_test)) if GPRPredictors.__check_predictor(pred) else (np.zeros(x_test.shape[:-1]), np.zeros(x_test.shape), np.zeros(x_test.shape[:-1] + (1, pes.PHASEDIM)), np.zeros(x_test.shape[:-1] + (1, pes.PHASEDIM)), np.zeros(x_test.shape[:-1] + (1,))) # unknown dims filled by 1
		)

		for iPES, jPES in zip(pes.tril_row_indices, pes.tril_col_indices):
			ElementIndex: int = iPES * pes.NUM_PES + jPES
			x_all: npt.NDArray[np.double] = self[ElementIndex].x_all.detach().numpy()
			num_subset: int = self[ElementIndex].get_training_features().shape[0]
			r: npt.NDArray[np.double] = x_all[..., :pes.DIM] # position coordinates
			v: npt.NDArray[np.double] = x_all[..., pes.DIM:] / mass # velocity
			D: npt.NDArray[np.double] = pes.adiabatic_coupling(r) # coupling
			E: npt.NDArray[np.double] = pes.adiabatic_potential(r) # energy
			F: npt.NDArray[np.double] = pes.adiabatic_force(r) # ``force''
			# dGPR/dt at x_all
			phase_deriv: npt.NDArray[np.double] = np.concat((v, (F[..., iPES, iPES] + F[..., jPES, jPES]) / 2.0), -1) # same shape as x_all
			pred, grad_input, grad_subset, grad_fullset, grad_param = predict_derivatives(x_all, iPES, jPES)
			# drho/dt at x_all
			time_deriv: npt.NDArray[np.cdouble]
			if iPES == jPES:
				time_deriv = -(phase_deriv * grad_input).sum(-1)
			else:
				time_deriv = 1.0j / pes.HBAR * (E[..., jPES] - E[..., iPES]) * pred - (phase_deriv * grad_input).sum(-1)
			for kPES in range(pes.NUM_PES):
				if kPES != iPES:
					time_deriv -= (D[..., iPES, kPES] * (v * self.predict(x_all, kPES * pes.NUM_PES + jPES)[..., np.newaxis] + (E[..., iPES] - E[..., kPES])[..., np.newaxis] / 2.0 * predict_derivative_over_input(x_all, kPES, jPES)[..., pes.DIM:])).sum(-1)
				if kPES != jPES:
					time_deriv += (D[..., kPES, jPES] * (v * self.predict(x_all, iPES * pes.NUM_PES + kPES)[..., np.newaxis] + (E[..., jPES] - E[..., kPES])[..., np.newaxis] / 2.0 * predict_derivative_over_input(x_all, iPES, kPES)[..., pes.DIM:])).sum(-1)
			# get eq for param deriv
			if iPES == jPES:
				self[ElementIndex].update_param(torch.from_numpy(np.linalg.lstsq(
					grad_param.real,
					(time_deriv.real - ((grad_fullset.real * phase_deriv).sum((-1, -2)) + (grad_subset.real * phase_deriv[:num_subset]).sum((-1, -2)) + (self[ElementIndex].predict_using_given_labels(torch.from_numpy(x_all), torch.from_numpy(time_deriv)))))
				)[0]) * dt) # rests of the return from lstsq are residual, rank of matrix, and singular values
			else:
				SymElementIndex: int = jPES * pes.NUM_PES + iPES
				remove_deriv_on_coord: npt.NDArray[np.cdouble] = time_deriv - ((grad_fullset * phase_deriv).sum((-1, -2)) + (grad_subset * phase_deriv[:num_subset]).sum((-1, -2)))
				param_deriv: torch.Tensor = torch.from_numpy(np.linalg.lstsq(
					np.concat((grad_param.real, grad_param.imag), -1),
					np.concat(
						(remove_deriv_on_coord.real - self[SymElementIndex].predict_using_given_labels(torch.from_numpy(x_all), torch.from_numpy(time_deriv.real)),
						remove_deriv_on_coord.imag - self[ElementIndex].predict_using_given_labels(torch.from_numpy(x_all), torch.from_numpy(time_deriv.imag))),
						-1
					)
				)[0])
				n_param: int = param_deriv.numel() // 2
				self[SymElementIndex].update_param(param_deriv[:n_param] * dt) # real part
				self[ElementIndex].update_param(param_deriv[n_param:] * dt) # low trig, imag part

	def train(self, print_log: bool = DEBUG_MODE) -> None:
		r"""To train each predictor

		Parameters
		----------
		print_log : bool, optional
			Whether to print the log to console, by default `DEBUG_MODE`
		"""
		for iElement in range(pes.NUM_ELM):
			if __class__.__check_predictor(self.__predictors[iElement]):
				print("Training " + utility.get_RI_label(iElement))
				self.__predictors[iElement].train(print_log)

	def predict(self, x_input: npt.NDArray[np.double], ElementIndex: int) -> npt.NDArray[np.cdouble]:
		r"""To predict test targets based on input and corresponding density matrix element

		Parameters
		----------
		x_input : npt.NDArray[np.double], shape of (..., PHASEDIM)
			Test inputs
		ElementIndex : int
			Index of the element

		Returns
		-------
		npt.NDArray[np.cdouble], shape of (...)
			Density of the element of all test inputs
		"""
		RowIndex: typing.Final[int] = ElementIndex // pes.NUM_PES
		ColIndex: typing.Final[int] = ElementIndex % pes.NUM_PES
		return self.__combine_to_complex(x_input, RowIndex, ColIndex, lambda pred, x_test: (pred.predict(x_test).detach().numpy(),) if GPRPredictors.__check_predictor(pred) else (np.zeros(x_test.shape[0], np.double),))[0]

	def get_marginal(
		self,
		dimensions: int | collections.abc.Iterable[int],
		x_input: npt.NDArray[np.double],
		ElementIndex: int
	) -> npt.NDArray[np.cdouble]:
		r"""To get the marginal distribution of current gaussian process regressions

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
		def call_single_predictor(pred: SinglePredictor, x_test: torch.Tensor, dims: collections.abc.Sequence[int]) -> npt.NDArray[np.double]:
			r"""To do prediction of a single predictor

			Parameters
			----------
			pred : SinglePredictor
				The predictor
			x_input : torch.Tensor, shape of (N, len(dimensions))
				Test inputs
			dims : collections.abc.Sequence[int]
				The dims to be kept

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
		RowIndex: int = ElementIndex // pes.NUM_PES
		ColIndex: int = ElementIndex % pes.NUM_PES
		return self.__combine_to_complex(
			x_input,
			RowIndex,
			ColIndex,
			call_single_predictor,
			dimensions
		)[0]

	def print(self, f: typing.IO) -> None:
		r"""To print the parameters to file

		Parameters
		----------
		f : io.TextIOWrapper
			The file to save the parameters
		"""
		for predictor in self.__predictors:
			np.savetxt(f, predictor.model.cov.lengthscale.detach().numpy().reshape(1, -1))
		print("\n", file=f)
