r"""gp
==
Implementation for gaussian process (gp) regression.
"""
import abc
import collections.abc
import math
import typing

import linear_operator
import torch

import constant
import opt

torch.set_default_dtype(constant.DTYPE)
torch.set_default_device(constant.DEVICE)
torch.manual_seed(constant.SEED)


def rbf(x1: torch.Tensor, x2: torch.Tensor, /, *, lengthscale: torch.Tensor) -> torch.Tensor:
	r"""To calculate the covariance matrix using RBF kernel

	Parameters
	----------
	x1 : torch.Tensor, shape of (...., M, D)
		The first feature set
	x2 : torch.Tensor, shape of (..., N, D)
		The second feature set
	lengthscale : torch.Tensor, shape of (D,)
		The characteristic lengths

	Returns
	-------
	torch.Tensor, shape of (..., M, N)
		The covariance matrix
	"""
	assert x1.shape[-1] == x2.shape[-1] == lengthscale.numel()
	assert x1.ndim >= 2 and x2.ndim >= 2
	return torch.exp(-torch.square((x1[..., torch.newaxis, :] - x2[..., torch.newaxis, :, :]) / lengthscale.reshape(-1)).sum(-1) / 2.0)

def wendland_rbf(x1: torch.Tensor, x2: torch.Tensor, /, *, lengthscale: torch.Tensor, rho: torch.Tensor = torch.tensor(2.5)) -> torch.Tensor:
	r"""To calculate the covariance matrix using Wendland kernel with k=1

	Parameters
	----------
	x1 : torch.Tensor, shape of (...., M, D)
		The first feature set
	x2 : torch.Tensor, shape of (..., N, D)
		The second feature set
	lengthscale : torch.Tensor, shape of (D,)
		The characteristic lengths
	rho : torch.Tensor, optional
		The support radius, by default torch.tensor(2.5)

	Returns
	-------
	torch.Tensor, shape of (..., M, N)
		The covariance matrix
	"""
	assert x1.shape[-1] == x2.shape[-1] == lengthscale.numel()
	half_dim: typing.Final[int] = lengthscale.numel() // 2
	assert x1.ndim >= 2 and x2.ndim >= 2
	distance: typing.Final[torch.Tensor] = torch.sqrt(torch.square((x1[..., torch.newaxis, :] - x2[..., torch.newaxis, :, :]) / lengthscale.reshape(-1)).sum(-1))
	distance_wendland: typing.Final[torch.Tensor] = distance / rho
	return torch.where(distance_wendland < 1.0, (1.0 - distance_wendland) ** (half_dim + 3) * ((half_dim + 3) * distance_wendland + 1.0), torch.zeros_like(distance)) * torch.exp(-distance ** 2 / 2.0)


class KernelPredictor(abc.ABC):
	r"""Base class of kernel predictor

	Parameters
	----------
	x_ind : torch.Tensor, of shape (N_IND_PT, PHASEDIM)
		Inducing Points
	y_ind : torch.Tensor, of shape (N_IND_PT)
		Targets of inducing points
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
	__slots__: tuple = ("x_ind", "y_ind", "x_all", "y_all", "scale", "_old_raw_lengthscale", "_raw_lengthscale", "_lr")
	x_ind: torch.Tensor
	y_ind: torch.Tensor
	x_all: torch.Tensor
	y_all: torch.Tensor
	scale: float
	_old_raw_lengthscale: torch.Tensor
	_raw_lengthscale: torch.Tensor
	_lr: float | None

	def __init__(
		self,
		x_ind: torch.Tensor,
		y_ind: torch.Tensor,
		x_all: torch.Tensor,
		y_all: torch.Tensor,
		lengthscale_initial_value: torch.Tensor,
		indent: int
	) -> None:
		self.x_ind = x_ind.detach().clone()
		self.y_ind = y_ind.detach().clone()
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
		chunk_size: int = N
		change_to_2_power: bool = False
		power_of_2_max: typing.Final[int] = 64 # wavefront of 64 on AMD and warps of 32 on NV
		while chunk_size > 1:
			try:
				if print_log:
					print("\tChunk size for autograd test:", chunk_size)
				self.predict_derivative_over_internal(x_all, chunk_size)
				break
			except torch.OutOfMemoryError:
				chunk_size //= 2
				if chunk_size < power_of_2_max and not change_to_2_power:
					chunk_size = power_of_2_max
					change_to_2_power = True
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
		x_ind: torch.Tensor,
		y_ind: torch.Tensor,
		x_all: torch.Tensor,
		y_all: torch.Tensor
	) -> None:
		r"""To update the training features and labels of the model

		Parameters
		----------
		x_ind : torch.Tensor, of shape (N_IND_PT, PHASEDIM)
			Inducing Points
		y_ind : torch.Tensor, of shape (N_IND_PT)
			Targets of inducing points
		x_all : torch.Tensor, of shape (N_ALL_PT, PHASEDIM)
			All training inputs
		y_all : torch.Tensor, of shape (N_ALL_PT)
			All training targets
		"""
		self.x_ind = x_ind.reshape(-1, x_ind.shape[-1]).detach().clone()
		self.y_ind = y_ind.reshape(-1).detach().clone()
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
	x_ind : torch.Tensor, of shape (N_IND_PT, PHASEDIM)
		Inducing Points
	y_ind : torch.Tensor, of shape (N_IND_PT)
		Targets of inducing points
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
	__slots__: typing.Final[tuple] = ("__k_inv_y", "__weights_updated", "__lr")
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
		y_ind: torch.Tensor,
		x_all: torch.Tensor,
		y_all: torch.Tensor,
		lengthscale_initial_value: torch.Tensor,
		indent: int
	) -> None:
		self.__weights_updated = False
		super().__init__(x_ind, y_ind, x_all, y_all, lengthscale_initial_value, indent)
		self.__k_inv_y = (linear_operator.utils.stable_pinverse(wendland_rbf(self.x_all, self.x_ind, lengthscale=self.raw_to_real(self._raw_lengthscale))) @ self.y_all).detach()
		self._lr = 1.0 if self._raw_lengthscale.numel() >= 10 else None # use gradient descend if dimension is large, otherwise use newton method

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
		return linear_operator.utils.stable_pinverse(wendland_rbf(self.x_all, self.x_ind, lengthscale=self.raw_to_real(self._raw_lengthscale if raw_lengthscale is None else raw_lengthscale))) @ self.y_all

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
			return wendland_rbf(x_test, self.x_ind, lengthscale=lengthscale) @ self.__calculate_k_inv_y(raw_lengthscale)
		else:
			return (wendland_rbf(x_test, self.x_ind, lengthscale=lengthscale) @ self.k_inv_y).detach()

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
		predict: typing.Final[torch.Tensor] = wendland_rbf(x_test, x_ind, lengthscale=self.raw_to_real(raw_lengthscale)) @ self.__calculate_k_inv_y(raw_lengthscale)
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
		kxm: typing.Final[torch.Tensor] = wendland_rbf(x_test, self.x_ind, lengthscale=lengthscale)
		kmm: typing.Final[torch.Tensor] = wendland_rbf(self.x_ind, self.x_ind, lengthscale=lengthscale) + KernelPredictor.NOISE * torch.eye(self.x_ind.shape[0])
		# diagonal only, k(x, x) = 1.0 for RBF kernel
		result: typing.Final[torch.Tensor] = (1.0 - torch.einsum("ij,jk,ik->i", kxm, torch.cholesky_inverse(torch.linalg.cholesky_ex(kmm)[0]), kxm)).clamp(0.0, 1.0)
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
		return prefactor * wendland_rbf(x_test, self.x_ind[:, dimensions], lengthscale=self.lengthscale[dimensions]).to_dense() @ linear_operator.utils.stable_pinverse(wendland_rbf(self.x_all[:, dimensions], self.x_ind[:, dimensions], lengthscale=self.lengthscale[dimensions])) @ self.y_all

	@typing.override
	def update(
		self,
		x_ind: torch.Tensor,
		y_ind: torch.Tensor,
		x_all: torch.Tensor,
		y_all: torch.Tensor
	) -> None:
		r"""To update the training features and labels of the model

		Parameters
		----------
		x_ind : torch.Tensor, of shape (N_IND_PT, PHASEDIM)
			Inducing Points
		y_ind : torch.Tensor, of shape (N_IND_PT)
			Targets of inducing points
		x_all : torch.Tensor, of shape (N_ALL_PT, PHASEDIM)
			All training inputs
		y_all : torch.Tensor, of shape (N_ALL_PT)
			All training targets
		"""
		super().update(x_ind, y_ind, x_all, y_all)
		self.__weights_updated = False
