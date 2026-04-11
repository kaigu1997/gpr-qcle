r"""gp
==
Implementation for gaussian process (gp) regression.
"""
import abc
import enum
import collections.abc
import math
import typing

import linear_operator
import torch

import constant
import plot

torch.set_default_dtype(constant.DTYPE)
torch.set_default_device(constant.DEVICE)
torch.manual_seed(constant.SEED)


class Optimizer(abc.ABC):
	r"""Base class of callable optimizer

	Parameters
	----------
	param : torch.Tensor
		Initial parameter
	loss_func : collections.abc.Callable[[torch.Tensor], torch.Tensor]
		Loss function to minimize, whose parameter is `param` and return a 0-dim Tensor

	Attributes
	----------
	MAX_ITER : typing.Literal[15000]
		Maximum iteration of optimization
	FTOL : float
		Absolute and relative tolerance of function in optimization
	GTOL : float
		Tolerance of gradient in optimization

	Methods
	-------
	print_model(param, grad, hessian)
		To print the parameters of the model
	print_stuff(loss, param, lr, grad, hessian, extra_start_str, extra_model_str)
		To print all stuffs needed
	"""
	MAX_ITER: typing.Final = 15000
	FTOL: typing.Final = 2.2204460492503131e-09
	GTOL: typing.Final = 1e-5

	class ResultMessage(enum.StrEnum):
		GRAD = "Convergence: |Gradient| <= GTOL"
		FVAL = "Convergence: |f_i - f_{i+1}| <= FTOL"
		ITER = "Stop: Total No. iterations reached limit."
		STUCK = "No stepping forward"

	class Result(typing.NamedTuple):
		param: torch.Tensor
		grad: torch.Tensor
		hessian: torch.Tensor | None
		lr: float
		func_value: float
		num_iter: int
		message: "Optimizer.ResultMessage"

	@staticmethod
	def print_model(param: torch.Tensor, grad: torch.Tensor | None = None, hessian: torch.Tensor | None = None) -> None:
		r"""To print the parameters of the model

		Parameters
		----------
		param : torch.Tensor
			Parameters
		grad : torch.Tensor | None
			Gradient of the parameters
		hessian : torch.Tensor | None, optional
			Hessian matrix of the parameters, by default None
		"""
		print(f"param = {plot.format_array(param)}{f" grad = {plot.format_array(grad)}" if grad is not None else ""}{f" hessian = {plot.format_array(hessian)}" if hessian is not None else ""}")

	@staticmethod
	def print_stuff(
		loss: float,
		param: torch.Tensor,
		lr: float | None,
		grad: torch.Tensor | None = None,
		hessian: torch.Tensor | None = None,
		extra_start_str: str = "\t\t\t\t",
		extra_model_str: str | None = None,
	) -> None:
		r"""To print all stuffs needed

		Parameters
		----------
		loss : float
			Loss from error function
		param : torch.Tensor
			Parameters
		lr : float | None
			Learning rate
		grad : torch.Tensor
			Gradient of the parameters
		hessian : torch.Tensor | None, optional
			Hessian matrix of the parameters, by default None
		extra_str : str, optional
			An extra string added at the front, by default "\t"s
		extra_start_str : str | None, optional
			An extra string added at the front of model printing, by default None
		"""
		if extra_model_str is None:
			extra_model_str = extra_start_str
		print(f"{extra_start_str}loss = {loss:.15e}{f" - lr = {lr}" if lr is not None else ""}\n{extra_model_str}raw ", end="")
		Optimizer.print_model(param, grad, hessian)

	@abc.abstractmethod
	def __new__(
		cls,
		param: torch.Tensor,
		loss_func: collections.abc.Callable[[torch.Tensor], torch.Tensor],
		*args,
		**kwargs
	) -> Result:
		...

class GradientDescend(Optimizer):
	r"""Gradient Descend

	Parameters
	----------
	param : torch.Tensor
		Initial parameter
	loss_func : collections.abc.Callable[[torch.Tensor], torch.Tensor]
		Loss function to minimize, whose parameter is `param` and return a 0-dim Tensor
	lr : float, optional
		Initial learning rate, by default 1.0
	print_log : bool, optional
		Whether to print log or not, by default constant.DEBUG_MODE

	Returns
	-------
	Optimizer.Result
		Optimization result
	"""
	def __new__(
		cls,
		param: torch.Tensor,
		loss_func: collections.abc.Callable[[torch.Tensor], torch.Tensor],
		lr: float = 1.0,
		print_log: bool = constant.DEBUG_MODE
	) -> Optimizer.Result:
		param = param.reshape(-1).detach().requires_grad_()
		loss: torch.Tensor = loss_func(param)
		grad: torch.Tensor = torch.autograd.grad(loss, param, allow_unused=True, materialize_grads=True)[0]
		lr = min(lr, (param.abs() / grad.abs()).max().item())
		last_value: float = loss.item()
		Optimizer.print_stuff(loss.item(), param, lr, grad, None, "\t\tInit ")
		for i in range(1, Optimizer.MAX_ITER + 1):
			if torch.norm(grad).item() < Optimizer.GTOL:
				return Optimizer.Result(param, grad, None, lr, loss.item(), i - 1, Optimizer.ResultMessage.GRAD)
			# adjust lr
			loss = loss_func((param - lr * grad).detach())
			if print_log:
				Optimizer.print_stuff(loss.item(), param - lr * grad, lr, grad)
			if loss < last_value:
				param = (param - lr * grad).detach().requires_grad_()
				lr *= 2.0
				if print_log:
					print("\t\t\t\tloss < last_value")
			else:
				if print_log:
					print("\t\t\t\tloss > last_value or loss is NaN")
				while loss >= last_value or loss.isnan().item():
					last_loop_value: float = loss.item()
					lr /= 2.0
					loss = loss_func((param - lr * grad).detach())
					if print_log:
						Optimizer.print_stuff(loss.item(), param - lr * grad, lr, grad, None, "\t\t\t\t\t")
					if last_loop_value == loss.item():
						# no stepping forward, but still larger than last, meaning last is the best
						return Optimizer.Result(param, grad, None, lr, loss_func(param).item(), i, Optimizer.ResultMessage.STUCK)
				param = (param - lr * grad).detach().requires_grad_()
			if i % (Optimizer.MAX_ITER // 1000) == 0 or print_log:
				Optimizer.print_stuff(loss.item(), param, lr, grad, None, f"\t\t\tIter {i} - last = {last_value} - ", "\t\t\t")
			# stopping criteria
			if (last_value - loss.item()) / max(abs(last_value), abs(loss.item()), 1.0) < Optimizer.FTOL:
				return Optimizer.Result(param, grad, None, lr, loss.item(), i, Optimizer.ResultMessage.FVAL)
			# update parameter and loss
			last_value = loss.item()
			loss = loss_func(param) # in fact this has been updated. Here is just for gradient
			grad = torch.autograd.grad(loss, param, allow_unused=True, materialize_grads=True)[0]
		return Optimizer.Result(param, grad, None, lr, loss.item(), Optimizer.MAX_ITER, Optimizer.ResultMessage.ITER)


class NewtonMethod(Optimizer):
	def __new__(
		cls,
		param: torch.Tensor,
		loss_func: collections.abc.Callable[[torch.Tensor], torch.Tensor],
		print_log: bool = constant.DEBUG_MODE
	) -> Optimizer.Result:
		param = param.reshape(-1).detach().requires_grad_()
		loss: torch.Tensor = loss_func(param)
		grad: torch.Tensor = torch.autograd.grad(loss, param, retain_graph=True, create_graph=True, allow_unused=True, materialize_grads=True)[0] + 0.0 * param
		hessian: torch.Tensor = torch.autograd.grad(grad, param, torch.eye(param.numel()), allow_unused=True, is_grads_batched=True, materialize_grads=True)[0].detach()
		hessian = (hessian + hessian.mH) / 2.0
		mu: float = 1e-6 * torch.max(torch.diag(hessian)).item()
		v: float = 2.0 # rate to adjust mu
		last_value: float = loss.item()
		Optimizer.print_stuff(loss.item(), param, mu, grad, hessian, "\t\tInit ")
		for i in range(1, Optimizer.MAX_ITER + 1):
			if torch.norm(grad).item() < Optimizer.GTOL:
				return Optimizer.Result(param, grad, None, mu, loss.item(), i - 1, Optimizer.ResultMessage.GRAD)
			# adjust lr
			while True:
				try:
					L = torch.linalg.cholesky(hessian + mu * torch.eye(param.numel()))
					break
				except RuntimeError:
					mu *= v
					if math.isinf(mu) or math.isnan(mu):
						return Optimizer.Result(param, grad, None, mu, loss.item(), i, Optimizer.ResultMessage.STUCK)
			update = torch.cholesky_solve(grad.reshape(-1, 1), L).reshape(-1)
			loss = loss_func((param - update).detach())
			if print_log:
				Optimizer.print_stuff(loss.item(), param - update, mu, grad, hessian + mu * torch.eye(param.numel()))
			if loss.item() < last_value:
				param = (param - update).detach().requires_grad_()
				rho: float = (last_value - loss.item()) / (grad @ update - 0.5 * update @ hessian @ update + 1e-12).item()
				mu = mu * max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
				v = 2.0
				if print_log:
					print("\t\t\t\tloss < last_value")
			else:
				if print_log:
					print("\t\t\t\tloss > last_value or loss is NaN")
				while loss >= last_value or loss.isnan().item():
					last_loop_value: float = loss.item()
					mu = mu * v
					v *= 2.0
					while True:
						try:
							L = torch.linalg.cholesky(hessian + mu * torch.eye(param.numel()))
							break
						except RuntimeError:
							mu *= v
							if math.isinf(mu) or math.isnan(mu):
								return Optimizer.Result(param, grad, None, mu, loss.item(), i, Optimizer.ResultMessage.STUCK)
					update = torch.cholesky_solve(grad.reshape(-1, 1), L).reshape(-1)
					loss = loss_func((param - update).detach())
					if print_log:
						Optimizer.print_stuff(loss.item(), param - update, mu, grad, hessian + mu * torch.eye(param.numel()), "\t\t\t\t\t")
					if last_loop_value == loss.item():
						# no stepping forward, but still larger than last, meaning last is the best
						return Optimizer.Result(param, grad, hessian, mu, loss_func(param).item(), i, Optimizer.ResultMessage.STUCK)
				# once loss is less than last, change the parameter
				param = (param - update).detach().requires_grad_()
				rho: float = (last_value - loss.item()) / (grad @ update - 0.5 * update @ hessian @ update + 1e-12).item()
				mu = mu * max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
				v = 2.0
			if i % (Optimizer.MAX_ITER // 1000) == 0 or print_log:
				Optimizer.print_stuff(loss.item(), param, mu, grad, hessian, f"\t\t\tIter {i} - last = {last_value} - ", "\t\t\t")
			# stopping criteria
			if (last_value - loss.item()) / max(abs(last_value), abs(loss.item()), 1.0) < Optimizer.FTOL:
				return Optimizer.Result(param, grad, hessian, mu, loss.item(), i, Optimizer.ResultMessage.FVAL)
			# update parameter and loss
			last_value = loss.item()
			loss = loss_func(param) # in fact this has been updated. Here is just for gradient
			grad = torch.autograd.grad(loss, param, retain_graph=True, create_graph=True, allow_unused=True, materialize_grads=True)[0] + 0.0 * param
			hessian = torch.autograd.grad(grad, param, torch.eye(param.numel()), allow_unused=True, is_grads_batched=True, materialize_grads=True)[0].detach()
			hessian = (hessian + hessian.mH) / 2.0
		return Optimizer.Result(param, grad, hessian, mu, loss.item(), Optimizer.MAX_ITER, Optimizer.ResultMessage.ITER)


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


class SinglePredictor:
	r"""The instantiation of gaussian process predictor

	Parameters
	----------
	RowIndex : int
		The row index of the predictor in the whole density matrix
	ColIndex : int
		The column index of the predictor in the whole density matrix
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

	Methods
	-------
	lengthscale()
		To access the real lengthscale
	k_inv_y()
		To get the weights, :math:`K^{-1}y`
	predict(x_test)
		To predict the average
	error()
		To get the error by comparing label with prediction
	predict_derivative_over_input(x_test)
		To give the derivative of prediction over the input
	predict_derivative_over_internal(x_test)
		To calculate the derivative of prediction over all related quantities
	get_marginal(x_test, dimensions)
		To get the marginal distribution of current gaussian process regression
	"""
	@typing.final
	class InputDerivativeReturn(typing.NamedTuple):
		predict: torch.Tensor
		derivative: torch.Tensor

	@typing.final
	class InternalDerivativeReturn(typing.NamedTuple):
		inducing_derivative: torch.Tensor
		feature_derivative: torch.Tensor
		label_derivative: torch.Tensor
		raw_param_derivative: torch.Tensor

	raw_to_real: collections.abc.Callable[[torch.Tensor], torch.Tensor] = staticmethod(torch.exp)
	real_to_raw: collections.abc.Callable[[torch.Tensor], torch.Tensor] = staticmethod(torch.log)
	__slots__: tuple = ("x_ind", "x_all", "y_all", "scale", "_old_raw_lengthscale", "_raw_lengthscale", "__k_inv_y", "__weights_updated")
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
		lengthscale_initial_value: torch.Tensor
	) -> None:
		self.x_ind = x_ind.detach().clone()
		self.x_all = x_all.detach().clone()
		self.y_all = y_all.detach().clone()
		self._old_raw_lengthscale = self.real_to_raw(lengthscale_initial_value).detach()
		self._raw_lengthscale = self.real_to_raw(lengthscale_initial_value).detach()
		self.scale = 1.0 / y_all.abs().max().item()
		self.__k_inv_y = (linear_operator.utils.stable_pinverse(rbf(self.raw_to_real(self._raw_lengthscale), self.x_all, self.x_ind)) @ self.y_all).detach()
		self.__weights_updated = True

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
				SinglePredictor.InternalDerivativeReturn(
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
		self.__weights_updated = False

	def __update_weights(self) -> None:
		r"""To update the weights, :math:`K^{-1}y`
		"""
		if not self.__weights_updated:
			self.__k_inv_y = (linear_operator.utils.stable_pinverse(rbf(self.raw_to_real(self._raw_lengthscale), self.x_all, self.x_ind)) @ self.y_all).detach()
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

	def predict(self, x_test: torch.Tensor, requires_grad: bool = False) -> torch.Tensor:
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
		result: typing.Final[torch.Tensor] = rbf(self.lengthscale, x_test, self.x_ind) @ self.k_inv_y
		if requires_grad:
			return result
		else:
			return result.detach()

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
		predict: typing.Final[torch.Tensor] = self.predict(x_test, True)
		return SinglePredictor.InputDerivativeReturn(predict=predict.detach(), derivative=torch.autograd.grad(predict, x_test, torch.ones_like(predict), False, False, True, True, False, True)[0].detach())

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
		predict: typing.Final[torch.Tensor] = rbf(self.raw_to_real(raw_lengthscale), x_test, x_ind) @ linear_operator.utils.stable_pinverse(rbf(self.raw_to_real(raw_lengthscale), x_all, x_ind)) @ y_all
		N: typing.Final[int] = predict.numel()
		eye: typing.Final[torch.Tensor] = torch.eye(predict.numel())
		chunk_range: typing.Final[range] = range(0, N, chunk_size)
		return SinglePredictor.InternalDerivativeReturn(
			feature_derivative=torch.cat([torch.autograd.grad(predict, x_all, eye[i:min(i+chunk_size, N)], True, False, True, True, True, True)[0].detach() for i in chunk_range], dim=0),
			inducing_derivative=torch.cat([torch.autograd.grad(predict, x_ind, eye[i:min(i+chunk_size, N)], True, False, True, True, True, True)[0].detach() for i in chunk_range], dim=0),
			label_derivative=torch.cat([torch.autograd.grad(predict, y_all, eye[i:min(i+chunk_size, N)], True, False, True, True, True, True)[0].detach() for i in chunk_range], dim=0),
			raw_param_derivative=torch.cat([torch.autograd.grad(predict, raw_lengthscale, eye[i:min(i+chunk_size, N)], True, False, True, True, True, True)[0].detach() for i in chunk_range], dim=0)
		)

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
		self.x_all = x_all.reshape(-1, x_all.shape[-1]).detach().clone()
		self.y_all = y_all.reshape(-1).detach().clone()
		self.scale = 1.0 / y_all.abs().max().item()
		self.__weights_updated = False

	def error(self) -> torch.Tensor:
		r"""Error function of subset of regressor (SR) / projected process (PP)

		This function gives the sum of squared error

		Returns
		-------
		torch.Tensor
			The sum of squared prediction error
		"""
		return torch.sum((self.y_all - self.predict(self.x_all)) ** 2) * (self.scale ** 2)


@typing.final
class ResidualPredictor(SinglePredictor):
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

	Methods
	-------
	train()
		To train the parameters
	"""
	raw_to_real: collections.abc.Callable[[torch.Tensor], torch.Tensor] = staticmethod(torch.nn.Softplus())
	real_to_raw: collections.abc.Callable[[torch.Tensor], torch.Tensor] = staticmethod(lambda x: torch.where(x * ResidualPredictor.raw_to_real.beta > ResidualPredictor.raw_to_real.threshold, x, x + torch.log(-torch.expm1(-x)))) # num stable inv softplus
	__slots__: typing.Final[tuple] = ("__lr",)
	__lr: float | None

	def __init__(
		self,
		x_ind: torch.Tensor,
		x_all: torch.Tensor,
		y_all: torch.Tensor,
		lengthscale_initial_value: torch.Tensor
	) -> None:
		super().__init__(x_ind, x_all, y_all, lengthscale_initial_value)
		self.__lr = 1.0 if self._raw_lengthscale.numel() >= 10 else None # use gradient descend if dimension is large, otherwise use newton method
		self.train()

	def train(self, print_log: bool = constant.DEBUG_MODE) -> None:
		r"""To train the parameters

		Parameters
		----------
		print_log : bool, optional
			Whether to print the log to console, by default `constant.DEBUG_MODE`
		"""
		def loss_func(length: torch.Tensor) -> torch.Tensor:
			kmn: torch.Tensor = rbf(self.raw_to_real(length), self.x_all, self.x_ind)
			return torch.sum((self.y_all - (kmn @ (linear_operator.utils.stable_pinverse(kmn) @ self.y_all)).to_dense()) ** 2) * (self.scale ** 2)

		# train model
		with torch.no_grad():
			self._raw_lengthscale.requires_grad = True
		print(f"\t\tscale = {self.scale}\n\t\t", end="")
		Optimizer.print_model(self.lengthscale)
		if self.__lr is None:
			loss: float = math.inf
			param: torch.Tensor = self._raw_lengthscale
			while True:
				result = NewtonMethod(param, loss_func, print_log)
				print(f"\t\tIter = {result.num_iter} - {result.message}")
				Optimizer.print_stuff(result.func_value, self.raw_to_real(result.param), None, extra_start_str="\t\t")
				if result.message in (Optimizer.ResultMessage.GRAD,) or result.func_value >= loss:
					break
				else:
					loss = result.func_value
					param = result.param
				result = GradientDescend(param, loss_func, print_log=print_log)
				print(f"\t\tIter = {result.num_iter} - {result.message}")
				Optimizer.print_stuff(result.func_value, self.raw_to_real(result.param), result.lr, extra_start_str="\t\t")
				if result.message in (Optimizer.ResultMessage.GRAD,) or result.func_value >= loss:
					break
				else:
					loss = result.func_value
					param = result.param
		else:
			result = GradientDescend(self._raw_lengthscale, loss_func, self.__lr, print_log)
			print(f"\t\tIter = {result.num_iter} - {result.message}")
			Optimizer.print_stuff(result.func_value, self.raw_to_real(result.param), result.lr, extra_start_str="\t\t")
		with torch.no_grad():
			self._raw_lengthscale.requires_grad = False
			self._raw_lengthscale = result.param.detach().clone()
			if self.__lr is not None:
				self.__lr = result.lr
