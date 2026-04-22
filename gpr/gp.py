r"""gp
==
Implementation for gaussian process (gp) regression.
"""
import abc
import collections.abc
import math
import sys
import typing

import linear_operator
import scipy.optimize
import torch

import constant
import opt

torch.set_default_dtype(constant.DTYPE)
torch.set_default_device(constant.DEVICE)
torch.manual_seed(constant.SEED)


def rbf(lengthscale: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
	r"""To calculate the covariance matrix using RBF kernel

	Parameters
	----------
	lengthscale : torch.Tensor, shape of (D,)
		The characteristic lengths
	x1 : torch.Tensor, shape of (...., M, D)
		The first feature set
	x2 : torch.Tensor, shape of (..., N, D)
		The second feature set

	Returns
	-------
	torch.Tensor, shape of (..., M, N)
		The covariance matrix
	"""
	assert x1.shape[-1] == x2.shape[-1] == lengthscale.numel()
	assert x1.ndim >= 2 and x2.ndim >= 2
	return torch.exp(-torch.square((x1[..., torch.newaxis, :] - x2[..., torch.newaxis, :, :]) / lengthscale.reshape(-1)).sum(-1) / 2.0)


class KernelPredictor(abc.ABC):
	r"""Base class of kernel predictor

	Parameters
	----------
	x_all : torch.Tensor, of shape (N_ALL, PHASEDIM)
		All training inputs
	y_all : torch.Tensor, of shape (N_ALL)
		All training targets
	scale : float
		The scaling factor
	lengthscale_initial_value : torch.Tensor
		Initial value of lengthscale
	indent : int
		The indent for printing log

	Attributes
	----------
	x_all : torch.Tensor, of shape (N_ALL, PHASEDIM)
		All training inputs
	y_all : torch.Tensor, of shape (N_ALL)
		All training targets

	Methods
	-------
	lengthscale()
		To access the real lengthscale
	predict(x_test)
		To predict the average
	predict_derivative_over_input(x_test)
		To give the derivative of prediction over the input
	predict_derivative_over_internal(x_test)
		To calculate the derivative of prediction over all related quantities
	error()
		To get the error by comparing label with prediction
	"""
	@typing.final
	class InputDerivativeReturn(typing.NamedTuple):
		predict: torch.Tensor
		derivative: torch.Tensor

	class InternalDerivativeReturn(typing.NamedTuple):
		feature_derivative: torch.Tensor
		label_derivative: torch.Tensor
		raw_param_derivative: torch.Tensor

	NOISE: typing.Final[float] = torch.finfo(torch.float32).eps
	raw_to_real: collections.abc.Callable[[torch.Tensor], torch.Tensor] = staticmethod(torch.nn.Softplus())
	real_to_raw: collections.abc.Callable[[torch.Tensor], torch.Tensor] = staticmethod(lambda x: torch.where(x * KernelPredictor.raw_to_real.beta > KernelPredictor.raw_to_real.threshold, x, x + torch.log(-torch.expm1(-x)))) # num stable inv softplus
	__slots__: tuple = ("x_all", "y_all", "scale", "_old_raw_lengthscale", "_raw_lengthscale", "_lr")
	x_all: torch.Tensor
	y_all: torch.Tensor
	scale: float
	_old_raw_lengthscale: torch.Tensor
	_raw_lengthscale: torch.Tensor
	_lr: float | None

	def __init__(
		self,
		x_all: torch.Tensor,
		y_all: torch.Tensor,
		lengthscale_initial_value: torch.Tensor,
		indent: int
	) -> None:
		self.x_all = x_all.detach().clone()
		self.y_all = y_all.detach().clone()
		self.scale = 1.0 / y_all.abs().max().item()
		self._old_raw_lengthscale = self.real_to_raw(lengthscale_initial_value).detach()
		self._raw_lengthscale = self.real_to_raw(lengthscale_initial_value).detach()
		self._lr = 1.0 if lengthscale_initial_value.numel() > 10 else None
		self.train(indent)

	def get_chunk_size(self, x_test: torch.Tensor, print_log: bool = constant.DEBUG_MODE) -> int:
		with torch.no_grad():
			x_all = self.x_all.detach().requires_grad_()
			y_all = self.y_all.detach().requires_grad_()
			raw_lengthscale: typing.Final[torch.Tensor] = self._raw_lengthscale.detach().requires_grad_()
		predict: typing.Final[torch.Tensor] = self.predict(x_test, None, True)
		N: typing.Final[int] = predict.numel()
		eye: typing.Final[torch.Tensor] = torch.eye(predict.numel())
		chunk_size: int = N
		chunk_range: range = range(0, N, chunk_size)
		change_to_2_power: bool = False
		power_of_2_max: typing.Final[int] = 64 # wavefront of 64 on AMD and warps of 32 on NV
		while chunk_size > 1:
			try:
				if print_log:
					print("\tChunk size for autograd test:", chunk_size)
				self.InternalDerivativeReturn(
					feature_derivative=torch.cat([torch.autograd.grad(predict, x_all, eye[i:min(i+chunk_size, N)], True, False, True, True, True, True)[0].detach() for i in chunk_range], dim=0),
					label_derivative=torch.cat([torch.autograd.grad(predict, y_all, eye[i:min(i+chunk_size, N)], True, False, True, True, True, True)[0].detach() for i in chunk_range], dim=0),
					raw_param_derivative=torch.cat([torch.autograd.grad(predict, raw_lengthscale, eye[i:min(i+chunk_size, N)], True, False, True, True, True, True)[0].detach() for i in chunk_range], dim=0)
				)
				break
			except torch.OutOfMemoryError:
				chunk_size //= 2
				if chunk_size < power_of_2_max and not change_to_2_power:
					chunk_size = power_of_2_max
					change_to_2_power = True
				chunk_range = range(0, N, chunk_size)
		print(f"Chunk size for autograd: {chunk_size}")
		return chunk_size

	@property
	def lengthscale(self) -> torch.Tensor:
		r"""To access the real lengthscale

		Returns
		-------
		torch.Tensor
			Lengthscale in kernel function
		"""
		return self.raw_to_real(self._raw_lengthscale) # use self to allow subclass to override

	def set_raw_lengthscale(self, value: torch.Tensor) -> None:
		r"""To set the lengthscale, which will update the weights

		This function is designed to be used in optimization and evolution.

		Parameters
		----------
		value : torch.Tensor
			New lengthscale in kernel function
		"""
		self._old_raw_lengthscale = self._raw_lengthscale
		self._raw_lengthscale = value.detach().clone()

	@abc.abstractmethod
	def predict(
		self,
		x_test: torch.Tensor,
		raw_lengthscale: torch.Tensor | None = None,
		requires_grad: bool = False
	) -> torch.Tensor:
		r"""Instance of prediction

		Parameters
		----------
		x_test : torch.Tensor, shape of (N, PHASEDIM)
			Validation/Test inputs
		raw_lengthscale : torch.Tensor | None, optional
			Lengthscale in kernel function, by default None (use current lengthscale)
		requires_grad : bool, optional
			Whether the prediction requires gradient, by default False

		Returns
		-------
		torch.Tensor, shape of (N,)
			Corresponding validation/test targets based on noise-free SR/PP mean.
		"""

	def predict_derivative_over_input(self, x_test: torch.Tensor) -> InputDerivativeReturn:
		r"""To give the derivative of prediction over the input

		Parameters
		----------
		x_test : torch.Tensor, shape of (N, PHASEDIM)
			Validation/Test inputs

		Returns
		-------
		InputDerivativeReturn
			Prediction (shape of (N,)), and its derivative over input (shape of (N, PHASEDIM))
		"""
		with torch.no_grad():
			x_test = x_test.reshape(-1, x_test.shape[-1]).detach().requires_grad_()
		predict: typing.Final[torch.Tensor] = self.predict(x_test, None, True)
		return KernelPredictor.InputDerivativeReturn(predict=predict.detach(), derivative=torch.autograd.grad(predict, x_test, torch.ones_like(predict), False, False, True, True, False, True)[0].detach())

	@abc.abstractmethod
	def predict_derivative_over_internal(self, x_test: torch.Tensor, chunk_size: int = 1) -> InternalDerivativeReturn:
		r"""To calculate the derivative of prediction over all related quantities

		Parameters
		----------
		x_test : torch.Tensor, shape of (N, PHASEDIM)
			Validation/Test inputs
		chunk_size : int, optional
			The number of VJP in parallel, used to avoid OOM, by default 1 (least OOM)

		Returns
		-------
		InternalDerivativeReturn
			derivative over training feature (shape of (N, M, PHASEDIM)),
			derivative over training label (shape of (N, M)),
			and derivative over raw characteristic lengthscale (shape of (N, PHASEDIM))
		"""

	def update(
		self,
		x_all: torch.Tensor,
		y_all: torch.Tensor
	) -> None:
		r"""To update the training features and labels of the model

		Parameters
		----------
		x_all : torch.Tensor, of shape (N_ALL_PT, PHASEDIM)
			All training inputs
		y_all : torch.Tensor, of shape (N_ALL_PT)
			All training targets
		"""
		self.x_all = x_all.reshape(-1, x_all.shape[-1]).detach().clone()
		self.y_all = y_all.reshape(-1).detach().clone()
		self.scale = 1.0 / y_all.abs().max().item()

	def error(self) -> torch.Tensor:
		r"""Error function of subset of regressor (SR) / projected process (PP)

		This function gives the sum of squared error

		Returns
		-------
		torch.Tensor
			The sum of squared prediction error
		"""
		return torch.sum((self.y_all - self.predict(self.x_all)) ** 2) * (self.scale ** 2)

	@abc.abstractmethod
	def loss_func(self, raw_lengthscale: torch.Tensor) -> torch.Tensor:
		"""The default loss function for predictors, for optimization routine to minimize, whose parameter is the raw lengthscale and return a 0-dim Tensor

			Note that as for optimization, the real lengthscale is the transformation of raw lengthscale by `self.raw_to_real`

			This design is for the convenience of optimization, since the lengthscale should be positive, and using raw lengthscale can guarantee the positivity without extra constraints.

		Parameters
		----------
		raw_lengthscale : torch.Tensor
			The raw lengthscale, which will be transformed to real lengthscale by `self.raw_to_real` and used in prediction and error calculation

		Returns
		-------
		torch.Tensor
			The loss, could be squared error, negative log marginal likelihood, or other loss function, as long as it is a 0-dim Tensor and can be optimized by optimization routine
		"""

	def train(
		self,
		indent: int,
		print_log: bool = constant.DEBUG_MODE
	) -> None:
		r"""To train the parameters

		Parameters
		----------
		indent : int
			The indent for printing log
		print_log : bool, optional
			Whether to print the log to console, by default `constant.DEBUG_MODE`
		"""

		# train model
		with torch.no_grad():
			self._raw_lengthscale.requires_grad = True
		print(f"{indent * "\t"}scale = {self.scale}\n{indent * "\t"}", end="")
		opt.Optimizer.print_model(self.lengthscale)
		if self._lr is None:
			loss: float = math.inf
			param: torch.Tensor = self._raw_lengthscale
			while True:
				print(f"{indent * "\t"}Optimization with Newton method:")
				result = opt.NewtonMethod(param, self.loss_func, indent + 1, print_log)
				print(f"{indent * "\t"}Iter = {result.num_iter} - {result.message}")
				opt.Optimizer.print_stuff(result.func_value, self.raw_to_real(result.param), None, extra_start_str=indent)
				if result.message in (opt.Optimizer.ResultMessage.GRAD,) or result.func_value >= loss:
					break
				else:
					loss = result.func_value
					param = result.param
				print(f"{indent * "\t"}Optimization with Gradient Descend method:")
				result = opt.GradientDescend(param, self.loss_func, indent + 1, print_log=print_log)
				print(f"{indent * "\t"}Iter = {result.num_iter} - {result.message}")
				opt.Optimizer.print_stuff(result.func_value, self.raw_to_real(result.param), result.lr, extra_start_str=indent)
				if result.message in (opt.Optimizer.ResultMessage.GRAD,) or result.func_value >= loss:
					break
				else:
					loss = result.func_value
					param = result.param
		else:
			result = opt.GradientDescend(self._raw_lengthscale, self.loss_func, indent, self._lr, print_log)
			print(f"{indent * "\t"}Iter = {result.num_iter} - {result.message}")
			opt.Optimizer.print_stuff(result.func_value, self.raw_to_real(result.param), result.lr, extra_start_str=indent)
		with torch.no_grad():
			self._raw_lengthscale.requires_grad = False
			self._raw_lengthscale = result.param.detach().clone()
			if self._lr is not None:
				self._lr = result.lr


class GaussianProcess(KernelPredictor):
	r"""The instantiation of gaussian process predictor

	Parameters
	----------
	x_ind : torch.Tensor, of shape (N_IND, PHASEDIM)
		Coordinates of inducing points
	x_all : torch.Tensor, of shape (N_ALL, PHASEDIM)
		All training inputs
	y_all : torch.Tensor, of shape (N_ALL)
		All training targets
	scale : float
		The scaling factor
	lengthscale_initial_value : torch.Tensor
		Initial value of lengthscale

	Attributes
	----------
	x_ind : torch.Tensor, of shape (N_IND, PHASEDIM)
		Coordinates of inducing points
	x_all : torch.Tensor, of shape (N_ALL, PHASEDIM)
		All training inputs
	y_all : torch.Tensor, of shape (N_ALL)
		All training targets
	indent : int
		The indent for printing log

	Methods
	-------
	k_inv_y()
		To get the weights, :math:`K^{-1}y`
	predict(x_test)
		To predict the average
	predict_derivative_over_internal(x_test)
		To calculate the derivative of prediction over all related quantities
	get_marginal(x_test, dimensions)
		To get the marginal distribution of current gaussian process regression
	"""
	@typing.final
	class InternalDerivativeReturn(typing.NamedTuple):
		inducing_derivative: torch.Tensor
		feature_derivative: torch.Tensor
		label_derivative: torch.Tensor
		raw_param_derivative: torch.Tensor

	raw_to_real: collections.abc.Callable[[torch.Tensor], torch.Tensor] = staticmethod(torch.exp)
	real_to_raw: collections.abc.Callable[[torch.Tensor], torch.Tensor] = staticmethod(torch.log)
	__slots__: typing.Final[tuple] = ("x_ind", "__k_inv_y", "__weights_updated", "__lr")
	x_ind: torch.Tensor
	x_all: torch.Tensor
	y_all: torch.Tensor
	_old_raw_lengthscale: torch.Tensor
	_raw_lengthscale: torch.Tensor
	scale: float
	__k_inv_y: torch.Tensor
	__weights_updated: bool

	def __init__(
		self,
		x_ind: torch.Tensor,
		x_all: torch.Tensor,
		y_all: torch.Tensor,
		lengthscale_initial_value: torch.Tensor,
		indent: int
	) -> None:
		self.x_ind = x_ind.detach().clone()
		self.__weights_updated = False
		super().__init__(x_all, y_all, lengthscale_initial_value, indent)
		self.__k_inv_y = (linear_operator.utils.stable_pinverse(rbf(self.raw_to_real(self._raw_lengthscale), self.x_all, self.x_ind)) @ self.y_all).detach()
		self._lr = 1.0 if self._raw_lengthscale.numel() >= 10 else None # use gradient descend if dimension is large, otherwise use newton method

	@typing.override
	def get_chunk_size(self, x_test: torch.Tensor, print_log: bool = constant.DEBUG_MODE) -> int:
		with torch.no_grad():
			x_ind = self.x_ind.detach().requires_grad_()
			x_all = self.x_all.detach().requires_grad_()
			y_all = self.y_all.detach().requires_grad_()
			raw_lengthscale: typing.Final[torch.Tensor] = self._raw_lengthscale.detach().requires_grad_()
		predict: typing.Final[torch.Tensor] = rbf(self.raw_to_real(raw_lengthscale), x_test.detach(), x_ind) @ linear_operator.utils.stable_pinverse(rbf(self.raw_to_real(raw_lengthscale), x_all, x_ind)) @ y_all
		N: typing.Final[int] = predict.numel()
		eye: typing.Final[torch.Tensor] = torch.eye(predict.numel())
		chunk_size: int = N
		chunk_range: range = range(0, N, chunk_size)
		change_to_2_power: bool = False
		power_of_2_max: typing.Final[int] = 64 # wavefront of 64 on AMD and warps of 32 on NV
		while chunk_size > 1:
			try:
				if print_log:
					print("\tChunk size for autograd test:", chunk_size)
				GaussianProcess.InternalDerivativeReturn(
					feature_derivative=torch.cat([torch.autograd.grad(predict, x_all, eye[i:min(i+chunk_size, N)], True, False, True, True, True, True)[0].detach() for i in chunk_range], dim=0),
					inducing_derivative=torch.cat([torch.autograd.grad(predict, x_ind, eye[i:min(i+chunk_size, N)], True, False, True, True, True, True)[0].detach() for i in chunk_range], dim=0),
					label_derivative=torch.cat([torch.autograd.grad(predict, y_all, eye[i:min(i+chunk_size, N)], True, False, True, True, True, True)[0].detach() for i in chunk_range], dim=0),
					raw_param_derivative=torch.cat([torch.autograd.grad(predict, raw_lengthscale, eye[i:min(i+chunk_size, N)], True, False, True, True, True, True)[0].detach() for i in chunk_range], dim=0)
				)
				break
			except torch.OutOfMemoryError:
				chunk_size //= 2
				if chunk_size < power_of_2_max and not change_to_2_power:
					chunk_size = power_of_2_max
					change_to_2_power = True
				chunk_range = range(0, N, chunk_size)
		print(f"Chunk size for autograd: {chunk_size}")
		return chunk_size

	def set_raw_lengthscale(self, value: torch.Tensor) -> None:
		r"""To set the lengthscale, which will update the weights

		This function is designed to be used in optimization and evolution.

		Parameters
		----------
		value : torch.Tensor
			New lengthscale in kernel function
		"""
		super().set_raw_lengthscale(value)
		self.__weights_updated = False

	def __calculate_k_inv_y(self, raw_lengthscale: torch.Tensor | None = None) -> torch.Tensor:
		r"""To calculate the weights, :math:`K^{-1}y`

		Parameters
		----------
		raw_lengthscale : torch.Tensor | None, optional
			Lengthscale in kernel function, by default None (use current lengthscale)
		"""
		return linear_operator.utils.stable_pinverse(rbf(self.raw_to_real(self._raw_lengthscale if raw_lengthscale is None else raw_lengthscale), self.x_all, self.x_ind)) @ self.y_all

	def __update_weights(self) -> None:
		r"""To update the weights, :math:`K^{-1}y`
		"""
		if not self.__weights_updated:
			self.__k_inv_y = self.__calculate_k_inv_y().detach()
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
		return self.__k_inv_y.detach()

	@typing.override
	def predict(
		self,
		x_test: torch.Tensor,
		raw_lengthscale: torch.Tensor | None = None,
		requires_grad: bool = False
	) -> torch.Tensor:
		r"""Instance of prediction of subset of regressor (SR) / projected process (PP)

		Parameters
		----------
		x_test : torch.Tensor, shape of (N, PHASEDIM)
			Validation/Test inputs
		raw_lengthscale : torch.Tensor | None, optional
			Lengthscale in kernel function, by default None (use current lengthscale)
		requires_grad : bool, optional
			Whether the prediction requires gradient, by default False

		Returns
		-------
		torch.Tensor, shape of (N,)
			Corresponding validation/test targets based on noise-free SR/PP mean.
		"""
		lengthscale: typing.Final[torch.Tensor] = self.lengthscale if raw_lengthscale is None else self.raw_to_real(raw_lengthscale)
		if requires_grad:
			return rbf(lengthscale, x_test, self.x_ind) @ self.__calculate_k_inv_y(raw_lengthscale)
		else:
			return (rbf(lengthscale, x_test, self.x_ind) @ self.k_inv_y).detach()

	@typing.override
	def predict_derivative_over_internal(self, x_test: torch.Tensor, chunk_size: int = 1) -> InternalDerivativeReturn:
		r"""To calculate the derivative of prediction over all related quantities

		Parameters
		----------
		x_test : torch.Tensor, shape of (N, PHASEDIM)
			Validation/Test inputs
		chunk_size : int, optional
			The number of VJP in parallel, used to avoid OOM, by default 1 (least OOM)

		Returns
		-------
		InternalDerivativeReturn
			derivative over training inducing features (shape of (N, m, PHASEDIM)),
			derivative over training feature (shape of (N, M, PHASEDIM)),
			derivative over training label (shape of (N, M)),
			and derivative over raw characteristic lengthscale (shape of (N, PHASEDIM))
		"""
		with torch.no_grad():
			x_ind: typing.Final[torch.Tensor] = self.x_ind.detach().requires_grad_()
			x_all: typing.Final[torch.Tensor] = self.x_all.detach().requires_grad_()
			y_all: typing.Final[torch.Tensor] = self.y_all.detach().requires_grad_()
			raw_lengthscale: typing.Final[torch.Tensor] = self._raw_lengthscale.detach().requires_grad_()
		predict: typing.Final[torch.Tensor] = rbf(self.raw_to_real(raw_lengthscale), x_test, x_ind) @ self.__calculate_k_inv_y(raw_lengthscale)
		N: typing.Final[int] = predict.numel()
		eye: typing.Final[torch.Tensor] = torch.eye(predict.numel())
		chunk_range: typing.Final[range] = range(0, N, chunk_size)
		return GaussianProcess.InternalDerivativeReturn(
			feature_derivative=torch.cat([torch.autograd.grad(predict, x_all, eye[i:min(i+chunk_size, N)], True, False, True, True, True, True)[0].detach() for i in chunk_range], dim=0),
			inducing_derivative=torch.cat([torch.autograd.grad(predict, x_ind, eye[i:min(i+chunk_size, N)], True, False, True, True, True, True)[0].detach() for i in chunk_range], dim=0),
			label_derivative=torch.cat([torch.autograd.grad(predict, y_all, eye[i:min(i+chunk_size, N)], True, False, True, True, True, True)[0].detach() for i in chunk_range], dim=0),
			raw_param_derivative=torch.cat([torch.autograd.grad(predict, raw_lengthscale, eye[i:min(i+chunk_size, N)], True, False, True, True, True, True)[0].detach() for i in chunk_range], dim=0)
		)

	def variance(self, x_test: torch.Tensor, raw_lengthscale: torch.Tensor | None = None, requires_grad: bool = False) -> torch.Tensor:
		r"""To calculate the variance of prediction of subset of regressor (SR) / projected process (PP)

		Parameters
		----------
		x_test : torch.Tensor, shape of (N, PHASEDIM)
			Validation/Test inputs
		raw_lengthscale : torch.Tensor | None, optional
			Lengthscale in kernel function, by default None (use current lengthscale)
		requires_grad : bool, optional
			Whether to require gradients, by default False

		Returns
		-------
		torch.Tensor, shape of (N,)
			The variance of corresponding validation/test targets based on SR/PP.
		"""
		lengthscale: typing.Final[torch.Tensor] = self.lengthscale if raw_lengthscale is None else self.raw_to_real(raw_lengthscale)
		kxm: typing.Final[torch.Tensor] = rbf(lengthscale, x_test, self.x_ind)
		kmm: typing.Final[torch.Tensor] = rbf(lengthscale, self.x_ind, self.x_ind) + KernelPredictor.NOISE * torch.eye(self.x_ind.shape[0])
		# magnitude, from maximum likelihood
		k_inv_y: typing.Final[torch.Tensor] = self.__calculate_k_inv_y(raw_lengthscale)
		magnitude_square: typing.Final[torch.Tensor] = k_inv_y @ kmm @ k_inv_y / self.y_all.numel()
		# diagonal only, k(x, x) = 1.0 for RBF kernel
		result: typing.Final[torch.Tensor] = (1.0 - torch.einsum("ij,jk,ik->i", kxm, torch.cholesky_inverse(torch.linalg.cholesky_ex(kmm)[0]), kxm)).clamp(0.0, 1.0) * magnitude_square
		if requires_grad:
			return result
		else:
			return result.detach()

	@typing.override
	def loss_func(self, raw_lengthscale: torch.Tensor) -> torch.Tensor:
		return torch.sum(torch.square(self.y_all - self.predict(self.x_all, raw_lengthscale, True))) * (self.scale ** 2)

	@typing.override
	def train(self, indent: int, print_log: bool = constant.DEBUG_MODE) -> None:
		super().train(indent, print_log)
		self.__weights_updated = False

	def get_marginal(self, x_test: torch.Tensor, dimensions: collections.abc.Sequence[int]) -> torch.Tensor:
		r"""To get the marginal distribution of current gaussian process regression

		Parameters
		----------
		x_test : torch.Tensor, shape of (N, len(dimensions))
			Validation/Test inputs
		dimensions : collections.abc.Sequence[int]
			The dimensions to be kept, must not have any repeat

		Returns
		-------
		torch.Tensor, shape of (N,)
			Corresponding validation/test targets based on noise-free SR/PP mean.
		"""
		phasedim: typing.Final[int] = self.x_all.shape[-1]
		prefactor: typing.Final[float] = math.sqrt((2.0 * torch.pi) ** (phasedim - len(dimensions))) * self.lengthscale[[i for i in range(phasedim) if i not in dimensions]].prod().item()
		return prefactor * rbf(self.lengthscale[dimensions], x_test, self.x_ind[:, dimensions]).to_dense() @ linear_operator.utils.stable_pinverse(rbf(self.lengthscale[dimensions], self.x_all[:, dimensions], self.x_ind[:, dimensions])) @ self.y_all

	def update(
		self,
		x_ind: torch.Tensor,
		x_all: torch.Tensor,
		y_all: torch.Tensor
	) -> None:
		r"""To update the training features and labels of the model

		Parameters
		----------
		x_ind : torch.Tensor, of shape (N_IND_PT, PHASEDIM)
			Inducing Points
		x_all : torch.Tensor, of shape (N_ALL_PT, PHASEDIM)
			All training inputs
		y_all : torch.Tensor, of shape (N_ALL_PT)
			All training targets
		"""
		self.x_ind = x_ind.reshape(-1, x_ind.shape[-1]).detach().clone()
		super().update(x_all, y_all)
		self.__weights_updated = False


class NadarayaWatson(KernelPredictor):
	r"""The instantiation of gaussian process predictor

	Parameters
	----------
	x_all : torch.Tensor, of shape (N_ALL, PHASEDIM)
		All training inputs
	y_all : torch.Tensor, of shape (N_ALL)
		All training targets
	scale : float
		The scaling factor
	lengthscale_initial_value : torch.Tensor
		Initial value of lengthscale
	indent : int
		The indent for printing log

	Methods
	-------
	predict(x_test)
		To predict the average
	predict_derivative_over_internal(x_test)
		To calculate the derivative of prediction over all related quantities
	"""
	__slots__: typing.Final[tuple] = ()

	def __init__(
		self,
		x_all: torch.Tensor,
		y_all: torch.Tensor,
		lengthscale_initial_value: torch.Tensor,
		indent: int
	) -> None:
		super().__init__(x_all, y_all, lengthscale_initial_value, indent)

	@typing.override
	def predict(self, x_test: torch.Tensor, raw_lengthscale: torch.Tensor | None = None, requires_grad: bool = False) -> torch.Tensor:
		lengthscale: typing.Final[torch.Tensor] = self.lengthscale if raw_lengthscale is None else self.raw_to_real(raw_lengthscale)
		kxn: typing.Final[torch.Tensor] = rbf(lengthscale, x_test, self.x_all)
		result: typing.Final[torch.Tensor] = (kxn @ self.y_all) / (kxn.sum(-1) + NadarayaWatson.NOISE)
		if requires_grad:
			return result
		else:
			return result.detach()

	@typing.override
	def predict_derivative_over_internal(self, x_test: torch.Tensor, chunk_size: int = 1) -> KernelPredictor.InternalDerivativeReturn:
		with torch.no_grad():
			x_all: typing.Final[torch.Tensor] = self.x_all.detach().requires_grad_()
			y_all: typing.Final[torch.Tensor] = self.y_all.detach().requires_grad_()
			raw_lengthscale: typing.Final[torch.Tensor] = self._raw_lengthscale.detach().requires_grad_()
		kxn: typing.Final[torch.Tensor] = rbf(self.raw_to_real(raw_lengthscale), x_test, x_all)
		predict: typing.Final[torch.Tensor] = (kxn @ y_all) / (kxn.sum(-1) + NadarayaWatson.NOISE)
		N: typing.Final[int] = predict.numel()
		eye: typing.Final[torch.Tensor] = torch.eye(predict.numel())
		chunk_range: typing.Final[range] = range(0, N, chunk_size)
		return KernelPredictor.InternalDerivativeReturn(
			feature_derivative=torch.cat([torch.autograd.grad(predict, x_all, eye[i:min(i+chunk_size, N)], True, False, True, True, True, True)[0].detach() for i in chunk_range], dim=0),
			label_derivative=torch.cat([torch.autograd.grad(predict, y_all, eye[i:min(i+chunk_size, N)], True, False, True, True, True, True)[0].detach() for i in chunk_range], dim=0),
			raw_param_derivative=torch.cat([torch.autograd.grad(predict, raw_lengthscale, eye[i:min(i+chunk_size, N)], True, False, True, True, True, True)[0].detach() for i in chunk_range], dim=0)
		)

	@typing.override
	def loss_func(self, raw_lengthscale: torch.Tensor) -> torch.Tensor:
		knn: typing.Final[torch.Tensor] = rbf(self.raw_to_real(raw_lengthscale), self.x_all, self.x_all)
		num: typing.Final[torch.Tensor] = knn @ self.y_all - self.y_all # subtract self
		den: typing.Final[torch.Tensor] = knn.sum(-1) + NadarayaWatson.NOISE - 1.0 # subtract Kii
		return torch.sum((self.y_all - num / den) ** 2) * (self.scale ** 2) # LOOCV error


class GP_NW_Mix:
	r"""The mix of gaussian process and nadaraya-watson predictor, which is trained by first training gaussian process and then using the result as initialization to train nadaraya-watson

	Parameters
	----------
	x_ind : torch.Tensor, of shape (N_IND, PHASEDIM)
		Coordinates of inducing points
	x_all : torch.Tensor, of shape (N_ALL, PHASEDIM)
		All training inputs
	y_all : torch.Tensor, of shape (N_ALL)
		All training targets
	scale : float
		The scaling factor
	lengthscale_initial_value : torch.Tensor
		Initial value of lengthscale
	indent : int
		The indent for printing log

	Methods
	-------
	predict(x_test)
		To predict the average
	predict_derivative_over_internal(x_test)
		To calculate the derivative of prediction over all related quantities
	update(x_ind, x_all, y_all)
		To update the training features and labels of the model
	"""
	__slots__: typing.Final[tuple] = ("gp", "nw", "var_ref")
	gp: typing.Final[GaussianProcess]
	nw: typing.Final[NadarayaWatson]
	var_ref: float

	def __init__(
		self,
		x_ind: torch.Tensor,
		x_all: torch.Tensor,
		y_all: torch.Tensor,
		lengthscale_initial_value: torch.Tensor,
		indent: int
	) -> None:
		print(f"{indent * "\t"}Training Gaussian Process part:")
		self.gp = GaussianProcess(x_ind, x_all, y_all, lengthscale_initial_value, indent + 1)
		print(f"{indent * "\t"}Training Nadaraya-Watson part:")
		self.nw = NadarayaWatson(x_all, y_all, lengthscale_initial_value, indent + 1)
		print(f"{indent * "\t"}Training variance reference:")
		self.var_ref = 1.0
		self.__self_train(indent + 1) # GP and NM have trained in constructor

	def predict(self, x_test: torch.Tensor, var_param: float | None = None) -> torch.Tensor:
		r"""To predict the average

		Parameters
		----------
		x_test : torch.Tensor
			Test inputs
		var_param : float | None, optional
			The variance to calculate the weight, by default None (use current variance reference)

		Returns
		-------
		torch.Tensor
			Predicted values
		"""
		var_param = self.var_ref if var_param is None else var_param
		var: typing.Final[torch.Tensor] = self.gp.variance(x_test) # diagonal only
		nans: typing.Final[torch.Tensor] = torch.isnan(var)
		if nans.any().item():
			print(f"Number of NaNs in variance: {nans.sum().item()} out of {nans.numel()}")
			exit(0)
		gp_weight: typing.Final[torch.Tensor] = torch.exp(-var / var_param)
		return self.gp.predict(x_test) * gp_weight + self.nw.predict(x_test) * (1.0 - gp_weight)

	def error(self, var_param: float | None = None) -> torch.Tensor:
		r"""Error function. This function gives the sum of squared error

		Parameters
		----------
		var_param : float | None, optional
			The variance to calculate the weight, by default None (use current variance reference)

		Returns
		-------
		torch.Tensor
			The sum of squared prediction error
		"""
		return torch.sum((self.gp.y_all - self.predict(self.gp.x_all, var_param)) ** 2) * (self.gp.scale ** 2)

	def __self_train(self, indent: int, print_log: bool = constant.DEBUG_MODE) -> None:
		r"""To train the parameters of itself only

		Parameters
		----------
		indent : int
			The indent for printing log
		print_log : bool, optional
			Whether to print the log to console, by default `constant.DEBUG_MODE`
		"""
		# binary search for golden
		LOWER_BOUND: typing.Final[float] = -sys.float_info.epsilon / math.log(sys.float_info.epsilon) # exp(-epsilon / LB) approx epsilon , LB approx 6.0e-18
		UPPER_BOUND: typing.Final[float] = -1.0 / math.log(1.0 - sys.float_info.epsilon) # exp(-1.0 / UB) approx 1.0 - epsilon, UB approx 4.5e15
		mid_val: float = self.error(self.var_ref).item()
		if mid_val < opt.Optimizer.FTOL:
			print(f"{indent * "\t"}Initial guess of variance reference = {self.var_ref}, error = {mid_val}, good enough, use it as the variance reference")
			return
		guess_param: float = self.var_ref
		left_val: float = self.error(guess_param / 2.0).item()
		right_val: float = self.error(guess_param * 2.0).item()
		print(f"{indent * "\t"}Initial guess of variance reference = {guess_param}, left error = {left_val}, mid error = {mid_val}, right error = {right_val}")
		bracket: tuple[float, float, float]
		if left_val >= mid_val and right_val >= mid_val:
			bracket = (guess_param / 2.0, guess_param, guess_param * 2.0)
			if print_log:
				print(f"{indent * "\t"}Initial guess is good, use it as the middle point of golden section search")
		elif left_val < mid_val:
			if print_log:
				print(f"{indent * "\t"}Initial guess is too large, search left")
			while left_val <= mid_val and guess_param / 2.0 > LOWER_BOUND:
				guess_param /= 2.0
				right_val = mid_val
				mid_val = left_val
				left_val = self.error(guess_param / 2.0).item()
				if print_log:
					print(f"{(indent + 2) * "\t"}Guess of variance reference: {guess_param}, left error: {left_val}, mid error: {mid_val}, right error: {right_val}")
			if guess_param / 2.0 <= LOWER_BOUND:
				right_val = mid_val
				mid_val = left_val
				left_val = self.error(LOWER_BOUND).item()
				if print_log:
					print(f"{(indent + 2) * "\t"}Guess of variance reference: {guess_param}, left error: {left_val}, mid error: {mid_val}, right error: {right_val}")
				if left_val <= mid_val:
					print(f"{(indent + 1) * "\t"}Lower bound reached, use it as the variance reference")
					self.var_ref = guess_param
					return
				else:
					print(f"{(indent + 1) * "\t"}Lower bound reached, use it as the middle point of golden section search")
					bracket = (LOWER_BOUND, guess_param, guess_param * 2.0)
			else:
				print(f"{(indent + 1) * "\t"}Guess of variance reference: {guess_param}, left error: {left_val}, mid error: {mid_val}, right error: {right_val}")
				bracket = (guess_param / 2.0, guess_param, guess_param * 2.0)
		else: # right_val < init_val
			if print_log:
				print(f"{indent * "\t"}Initial guess is too small, search right")
			while right_val <= mid_val and guess_param * 2.0 < UPPER_BOUND:
				guess_param *= 2.0
				left_val = mid_val
				mid_val = right_val
				right_val = self.error(guess_param * 2.0).item()
				if print_log:
					print(f"{(indent + 2) * "\t"}Guess of variance reference: {guess_param}, left error: {left_val}, mid error: {mid_val}, right error: {right_val}")
			if guess_param * 2.0 >= UPPER_BOUND:
				left_val = mid_val
				mid_val = right_val
				right_val = self.error(UPPER_BOUND).item()
				if print_log:
					print(f"{(indent + 2) * "\t"}Guess of variance reference: {guess_param}, left error: {left_val}, mid error: {mid_val}, right error: {right_val}")
				if right_val <= mid_val:
					print(f"{(indent + 1) * "\t"}Upper bound reached, use it as the variance reference")
					self.var_ref = UPPER_BOUND
					return
				else:
					print(f"{(indent + 1) * "\t"}Upper bound reached, use it as the middle point of golden section search")
					bracket = (guess_param / 2.0, guess_param, UPPER_BOUND)
			else:
				print(f"{(indent + 1) * "\t"}Guess of variance reference: {guess_param}, left error: {left_val}, mid error: {mid_val}, right error: {right_val}")
				bracket = (guess_param / 2.0, guess_param, guess_param * 2.0)
		result: typing.Final[scipy.optimize.OptimizeResult] = scipy.optimize.minimize_scalar(lambda var_ref: self.error(var_ref).item(), bracket, tol=opt.Optimizer.FTOL, options={"maxiter": opt.Optimizer.MAX_ITER, "disp": print_log})
		print(f"{indent * "\t"}{result.message.replace("\n", " ")}\n{indent * "\t"}Optimal variance reference: {result.x}, error: {result.fun}")
		self.var_ref = result.x

	def train(self, indent: int, print_log: bool = constant.DEBUG_MODE) -> None:
		r"""To train the parameters

		Parameters
		----------
		indent : int
			The indent for printing log
		print_log : bool, optional
			Whether to print the log to console, by default `constant.DEBUG_MODE`
		"""
		print(f"{indent * "\t"}Training Gaussian Process part:")
		self.gp.train(indent + 1, print_log)
		print(f"{indent * "\t"}Training Nadaraya-Watson part:")
		self.nw.train(indent + 1, print_log)
		print(f"{indent * "\t"}Training variance reference:")
		self.__self_train(indent + 1, print_log)

	def update(
		self,
		x_ind: torch.Tensor,
		x_all: torch.Tensor,
		y_all: torch.Tensor
	) -> None:
		r"""To update the training features and labels of the model

		Parameters
		----------
		x_ind : torch.Tensor, of shape (N_IND_PT, PHASEDIM)
			Inducing Points
		x_all : torch.Tensor, of shape (N_ALL_PT, PHASEDIM)
			All training inputs
		y_all : torch.Tensor, of shape (N_ALL_PT)
			All training targets
		"""
		self.gp.update(x_ind, x_all, y_all)
		self.nw.update(x_all, y_all)
