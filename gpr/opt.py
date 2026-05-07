r"""opt
===

This module contains the optimizers used in the project.
"""
import abc
import collections.abc
import enum
import math
import typing

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
		extra_start_str: str | int = 4,
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
		if isinstance(extra_start_str, int):
			extra_start_str = "\t" * extra_start_str
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
	indent : int
		The indent for printing log
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
		indent: int,
		lr: float = 1.0,
		print_log: int = 0
	) -> Optimizer.Result:
		param = param.reshape(-1).detach().requires_grad_()
		loss: torch.Tensor = loss_func(param)
		grad: torch.Tensor = torch.autograd.grad(loss, param, allow_unused=True, materialize_grads=True)[0]
		lr = min(lr, (param.abs() / grad.abs()).max().item())
		last_value: float = loss.item()
		Optimizer.print_stuff(loss.item(), param, lr, grad, None, f"{indent * "\t"}Init ")
		for i in range(1, Optimizer.MAX_ITER + 1):
			if torch.isnan(grad).any().item():
				return Optimizer.Result(param, grad, None, lr, loss.item(), i - 1, Optimizer.ResultMessage.STUCK)
			if torch.norm(grad).item() < Optimizer.GTOL:
				return Optimizer.Result(param, grad, None, lr, loss.item(), i - 1, Optimizer.ResultMessage.GRAD)
			# adjust lr
			loss = loss_func((param - lr * grad).detach())
			if print_log:
				Optimizer.print_stuff(loss.item(), param - lr * grad, lr, grad, extra_start_str=indent + 2)
			if loss < last_value:
				param = (param - lr * grad).detach().requires_grad_()
				lr *= 2.0
				if print_log:
					print(f"{(indent + 2) * "\t"}loss < last_value")
			else:
				if print_log:
					print(f"{(indent + 2) * "\t"}loss > last_value or loss is NaN")
				while loss >= last_value or loss.isnan().item():
					last_loop_value: float = loss.item()
					lr /= 2.0
					loss = loss_func((param - lr * grad).detach())
					if print_log:
						Optimizer.print_stuff(loss.item(), param - lr * grad, lr, grad, None, indent + 3)
					if last_loop_value == loss.item():
						# no stepping forward, but still larger than last, meaning last is the best
						return Optimizer.Result(param, grad, None, lr, loss_func(param).item(), i, Optimizer.ResultMessage.STUCK)
				param = (param - lr * grad).detach().requires_grad_()
			if i % (Optimizer.MAX_ITER // 1000) == 0 or print_log:
				Optimizer.print_stuff(loss.item(), param, lr, grad, None, f"{(indent + 1) * "\t"}Iter {i} - last = {last_value} - ", (indent + 1) * "\t")
			# stopping criteria
			if (last_value - loss.item()) / max(abs(last_value), abs(loss.item()), 1.0) < Optimizer.FTOL:
				return Optimizer.Result(param, grad, None, lr, loss.item(), i, Optimizer.ResultMessage.FVAL)
			# update parameter and loss
			last_value = loss.item()
			loss = loss_func(param) # in fact this has been updated. Here is just for gradient
			grad = torch.autograd.grad(loss, param, allow_unused=True, materialize_grads=True)[0]
		return Optimizer.Result(param, grad, None, lr, loss.item(), Optimizer.MAX_ITER, Optimizer.ResultMessage.ITER)


class NewtonMethod(Optimizer):
	r"""Using Newton's method to optimize the parameters

	Parameters
	----------
	param : torch.Tensor
		Initial parameter
	loss_func : collections.abc.Callable[[torch.Tensor], torch.Tensor]
		The loss function to minimize, whose parameter is `param` and return a 0-dim Tensor
	indent : int
		The indent for printing log
	print_log : bool, optional
		Whether to print log, by default constant.DEBUG_MODE

	Returns
	-------
	Optimizer.Result
		Optimization result
	"""
	def __new__(
		cls,
		param: torch.Tensor,
		loss_func: collections.abc.Callable[[torch.Tensor], torch.Tensor],
		indent: int,
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
		Optimizer.print_stuff(loss.item(), param, mu, grad, hessian, f"{indent * "\t"}Init ")
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
				Optimizer.print_stuff(loss.item(), param - update, mu, grad, hessian + mu * torch.eye(param.numel()), indent + 2)
			if loss.item() < last_value:
				param = (param - update).detach().requires_grad_()
				rho: float = (last_value - loss.item()) / (grad @ update - 0.5 * update @ hessian @ update + 1e-12).item()
				mu = mu * max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
				v = 2.0
				if print_log:
					print(f"{(indent + 2) * "\t"}loss < last_value")
			else:
				if print_log:
					print(f"{(indent + 2) * "\t"}loss > last_value or loss is NaN")
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
						Optimizer.print_stuff(loss.item(), param - update, mu, grad, hessian + mu * torch.eye(param.numel()), indent + 3)
					if last_loop_value == loss.item():
						# no stepping forward, but still larger than last, meaning last is the best
						return Optimizer.Result(param, grad, hessian, mu, loss_func(param).item(), i, Optimizer.ResultMessage.STUCK)
				# once loss is less than last, change the parameter
				param = (param - update).detach().requires_grad_()
				rho: float = (last_value - loss.item()) / (grad @ update - 0.5 * update @ hessian @ update + 1e-12).item()
				mu = mu * max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
				v = 2.0
			if i % (Optimizer.MAX_ITER // 1000) == 0 or print_log:
				Optimizer.print_stuff(loss.item(), param, mu, grad, hessian, f"{(indent + 1) * "\t"}Iter {i} - last = {last_value} - ", f"{(indent + 1) * "\t"}")
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

