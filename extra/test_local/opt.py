import collections.abc
import math
import os
import sys
import traceback
import typing

import gpytorch
import gpytorch.constraints
import numpy as np
import numpy.typing as npt
import torch

sys.path.append(os.path.dirname(__file__))

import gp

torch.set_default_dtype(torch.float64)
PARAM_UPLIM = 10


def format_array(arr_name : str | None, arr: typing.Any) -> str:
	result: str = (arr_name + " = ") if arr_name else ""
	if isinstance(arr, torch.Tensor) or isinstance(arr, np.ndarray):
		result += " ".join(format_array(None, val.item()) for val in arr.ravel())
	elif isinstance(arr, collections.abc.Iterable):
		result += " ".join(format_array(None, item) for item in arr)
	elif isinstance(arr, complex):
		result += f"{arr.real} + {arr.imag}i"
	else:
		result += str(arr)
	return result


def print_stuff(
	model: gpytorch.models.ExactGP,
	loss: float,
	learning_rate: float = math.nan,
	indent: int = 0,
	print_grad: bool = False,
	hessian: torch.Tensor | None = None,
	start_str: str="",
	end_str: str="",
	flush: bool=False
) -> None:
	def print_model(model_print_grad: bool = False) -> None:
		param_name_fmt_str: str = f"{"\t" * indent}Parameter name: {{}}"
		param_name: str
		param: torch.nn.Parameter
		constraint: gpytorch.constraints.Interval | None
		for param_name, param, constraint in model.named_parameters_and_constraints():
			if param.numel() < PARAM_UPLIM:
				if model_print_grad and param.grad is not None:
					print(
						param_name_fmt_str.format(param_name),
						format_array("value", param),
						format_array("grad", param.grad)
					)
				else:
					print(
						param_name_fmt_str.format("".join(param_name.split("raw_"))),
						format_array("value", constraint.transform(param) if isinstance(constraint, gpytorch.constraints.Interval) else param)
					)
			else:
				if model_print_grad and param.grad is not None:
					print(param_name_fmt_str.format(param_name), f"|grad| = {param.grad.norm().item()}")

	print(f"{"\t" * indent}{start_str + " " if start_str != "" else ""}loss = {loss:.15e}, lr = {learning_rate}")
	print_model()
	if print_grad:
		print_model(True)
		if hessian is not None:
			print("\t" * indent, format_array("hessian", hessian.reshape(-1)), sep="")
			print(f"{"\t" * indent}Cond(hessian): {torch.linalg.cond(hessian).item()}")
	print(("\t" * indent + end_str + "\n") if end_str != "" else "", end="", flush=flush)


LossFuncType = collections.abc.Callable[[gpytorch.kernels.Kernel, torch.Tensor, torch.Tensor], torch.Tensor]


class GradientOptimization:
	__FTOL: float = 2.2204460492503131e-09
	__GTOL: float = 1e-5
	__slots__ = ("__model", "__param_indices", "__loss_func", "__x", "__y", "__print_log", "__cov", "__ftol", "__xtol")
	def __init__(
		self,
		model: gpytorch.models.ExactGP,
		param_indices: list[int],
		loss_func: LossFuncType,
		x: torch.Tensor,
		y: torch.Tensor,
		print_log: bool,
		cov_name: str = "cov",
		ftol: float | None = None,
		xtol: float | None = None,
	):
		self.__model: gpytorch.models.ExactGP = model
		self.__param_indices: list[int] = param_indices
		self.__loss_func: LossFuncType = loss_func
		self.__x: torch.Tensor = x
		self.__y: torch.Tensor = y
		self.__print_log: bool = print_log
		assert hasattr(model, cov_name)
		self.__cov = getattr(model, cov_name)
		assert isinstance(self.__cov, gpytorch.kernels.Kernel)
		self.__ftol: float = ftol or __class__.__FTOL
		self.__xtol: float = xtol or __class__.__GTOL

	def __call__(self, learning_rate: float, loss: torch.Tensor) -> tuple[float, torch.Tensor, bool]:
		def optimize_with_learning_rate(
			combined_parameters: torch.Tensor,
			change: torch.Tensor,
			last_loss: float,
			method_name: str,
			initial_learning_rate: float = 1.0
		) -> tuple[torch.Tensor, float]:
			def change_param(target: torch.Tensor) -> None:
				with torch.no_grad():
					for iParam, param in enumerate(self.__model.parameters()):
						param[...] = target[self.__param_indices[iParam]:self.__param_indices[iParam + 1]].reshape_as(param).detach()

			def return_back() -> tuple[torch.Tensor, float]:
				if self.__print_log:
					print(f"\tNo stepping Forward for {method_name}.")
				with torch.no_grad():
					for iParam, param in enumerate(self.__model.parameters()):
						param[...] = combined_parameters[self.__param_indices[iParam]:self.__param_indices[iParam + 1]].reshape_as(param).detach()
				return torch.tensor(last_loss), initial_learning_rate

			learning_rate: float = initial_learning_rate
			change_param(combined_parameters - learning_rate * change)
			learning_rate_threshold: float = sys.float_info.epsilon * (combined_parameters.norm() / change.norm()).item() # lr * |change| should > double precision * |param|
			loss: torch.Tensor | None = None
			while learning_rate >= learning_rate_threshold:
				try:
					loss = self.__loss_func(self.__cov, self.__x, self.__y)
					break
				except RuntimeError: # NANs
					learning_rate /= 2.0
			if learning_rate < learning_rate_threshold:
				# based on current learning rate, all parameters have changes smaller than double precision
				return return_back()
			assert loss is not None
			if self.__print_log:
				print_stuff(self.__model, loss.item(), learning_rate, 2, start_str=f"last = {last_loss:.15e},", flush=True)
			while loss.isnan().item() or loss.item() >= last_loss:
				learning_rate /= 2.0
				change_param(combined_parameters - learning_rate * change)
				try:
					loss = self.__loss_func(self.__cov, self.__x, self.__y)
				except RuntimeError: # NANs
					return return_back()
				if self.__print_log:
					print_stuff(self.__model, loss.item(), learning_rate, 2, start_str=f"last = {last_loss:.15e},", flush=True)
				if learning_rate < learning_rate_threshold:
					# based on current learning rate, all parameters being the same
					if loss.item() >= last_loss:
						return return_back()
					else:
						if self.__print_log:
							print(f"\tNo stepping Forward for {method_name}.")
						break
			return loss, learning_rate

		# calculate gradient and hessian
		hessian: torch.Tensor | None = torch.eye(self.__param_indices[-1]) if self.__param_indices[-1] <= PARAM_UPLIM else None
		try:
			for iParam, param in enumerate(self.__model.parameters()):
				param.grad = torch.autograd.grad(loss, param, None, True, True, True, True, False, True)[0] # same shape of param
				if torch.any(torch.isnan(param.grad)).item() or torch.any(torch.isinf(param.grad)).item():
					return math.nan, loss, False
				if hessian is not None:
					for iGrad, grad_elm in enumerate(param.grad.reshape(-1)):
						for jParam, param_for_grad in enumerate(self.__model.parameters()):
							hessian[self.__param_indices[iParam] + iGrad, self.__param_indices[jParam]:self.__param_indices[jParam + 1]] = torch.autograd.grad(grad_elm, param_for_grad, None, True, False, True, True, False, True)[0].reshape(-1)
		except Exception as e:
			traceback.print_exception(e)
			return math.nan, loss, False
		if hessian is not None:
			if torch.any(torch.isnan(hessian)).item() or torch.any(torch.isinf(hessian)).item():
				return math.nan, loss, False
			hessian = (hessian + hessian.T) / 2.0 # symmetrize
		grad_combined: torch.Tensor = torch.cat([(param.grad if param.grad is not None else torch.zeros_like(param)).reshape(-1) for param in self.__model.parameters()])
		param_combined: torch.Tensor = torch.cat([param.reshape(-1) for param in self.__model.parameters()])
		if learning_rate < 0.0:
			learning_rate = min((param_combined.norm() / grad_combined.norm()).item(), 1.0)
		# stopping criteria
		grad_sqnm: float = torch.sum(grad_combined ** 2).item()
		if grad_sqnm < self.__xtol ** 2:
			print(f"Convergence: |Gradient| = {math.sqrt(grad_sqnm)} <= GTOL = {self.__xtol}")
			return learning_rate, loss, True
		# change parameter and log
		newton_change: torch.Tensor | None = gp.square_solver(hessian, grad_combined) if hessian is not None else None
		if learning_rate < 1.0:
			learning_rate *= 2.0
		elif learning_rate >= 1.0:
			learning_rate = 1.0
		if self.__print_log:
			print_stuff(
				self.__model,
				loss.item(),
				learning_rate,
				1,
				True,
				hessian,
				flush=True
			)
		last_value: float = loss.item()
		old_learning_rate: float = learning_rate
		changes: list[torch.Tensor | None] = [newton_change, grad_combined]
		methods: list[str] = ["Newton Method", "Gradient Descent"]
		not_none_changes: list[torch.Tensor] = [c for c in changes if c is not None]
		not_none_methods: list[str] = [methods[i] for i in range(min(len(changes), len(methods))) if changes[i] is not None]
		for change, method in zip(not_none_changes, not_none_methods):
			loss, learning_rate = optimize_with_learning_rate(
				param_combined,
				change,
				last_value,
				method,
				old_learning_rate
			)
			if loss.item() < last_value:
				break
		if loss.item() >= last_value:
			print("Stop: No stepping Forward for Gradient-Based Optimization")
			return math.nan, loss, False # tried all method and no one can step forward
		f_diff: float = abs(last_value - loss.item())
		f_denom: float = max(abs(last_value), abs(loss.item()), 1.0)
		if f_diff / f_denom < self.__ftol:
			print(f"Convergence: |f_i - f_{{i+1}}| = {f_diff} / {f_denom} <= FTOL = {self.__ftol}")
			return learning_rate, loss, True
		return learning_rate, loss, False # generally, doing next step with gradient again


class NonGradientOptimization:
	__FTOL = 1e-4
	__XTOL = 1e-4
	__RHO = 1.0
	__CHI = 2.0
	__PSI = 0.5
	__SIGMA = 0.5
	__NONZDELT = 0.05
	__ZDELT = 0.00025
	__slots__ = ("__model", "__loss_func", "__x", "__y", "__print_log", "__ftol", "__xtol", "__cov", "__N", "__sim", "__one2np1", "__fsim")

	def __assign_value_to_model(self, x: npt.NDArray[np.double]) -> None:
		idx: int = 0
		for param in self.__model.parameters():
			param.ravel()[...] = torch.from_numpy(x[idx:idx + param.numel()])
			idx += param.numel()

	def __func(self, x: npt.NDArray[np.double]) -> float:
		self.__assign_value_to_model(x)
		return self.__loss_func(self.__cov, self.__x, self.__y).item()

	def __init__(
		self,
		model: gpytorch.models.ExactGP,
		loss_func: LossFuncType,
		x: torch.Tensor,
		y: torch.Tensor,
		print_log: bool,
		cov_name: str = "cov",
		ftol: float | None = None,
		xtol: float | None = None
	) -> None:
		self.__model: gpytorch.models.ExactGP = model
		self.__loss_func: LossFuncType = loss_func
		self.__x: torch.Tensor = x
		self.__y: torch.Tensor = y
		self.__print_log: bool = print_log
		self.__ftol: float = ftol or __class__.__FTOL
		self.__xtol: float = xtol or __class__.__XTOL
		assert hasattr(model, cov_name)
		self.__cov = getattr(model, cov_name)
		assert isinstance(self.__cov, gpytorch.kernels.Kernel)
		# remove gradient
		with torch.no_grad():
			for param in self.__model.parameters():
				param.grad = None
				param.requires_grad = False
		x0: npt.NDArray[np.double] = np.concatenate([p.data.detach().numpy().ravel() for p in model.parameters()])
		self.__N: int = x0.size
		# function values
		self.__sim: npt.NDArray[np.double] = np.empty((self.__N + 1, self.__N))
		self.__sim[0] = x0
		for k in range(self.__N):
			x1: npt.NDArray[np.double] = np.copy(x0)
			if x1[k] != 0:
				x1[k] = (1 + __class__.__NONZDELT) * x1[k]
			else:
				x1[k] = __class__.__ZDELT
			self.__sim[k + 1] = x1
		# initial evaluate
		self.__one2np1: list[int] = list(range(1, self.__N + 1))
		self.__fsim: npt.NDArray[np.double] = np.full((self.__N + 1,), np.inf)
		for k in range(self.__N + 1):
			try:
				self.__fsim[k] = self.__func(self.__sim[k])
			finally:
				pass
		# sort so sim[0,:] has the lowest function value
		ind: npt.NDArray[np.int_] = np.argsort(self.__fsim)
		self.__sim = np.take(self.__sim, ind, 0)
		self.__fsim = np.take(self.__fsim, ind, 0)

	@property
	def converged(self) -> bool:
		f_diff: float = np.max(np.abs(self.__fsim[0] - self.__fsim[1:]))
		f_denom: float = max(np.max(np.abs(self.__fsim)), 1.0)
		if f_diff / f_denom <= self.__ftol:
			print(f"Convergence: |f_0 - f_i| = {f_diff} / {f_denom} <= FTOL = {self.__ftol}")
			return True
		x_diff: float = np.max(np.linalg.norm(self.__sim[1:] - self.__sim[0], axis=1))
		if x_diff <= self.__xtol:
			print(f"Convergence: ||x_0 - x_i|| = {x_diff} <= XTOL = {self.__xtol}")
			return True
		return False

	def __call__(self) -> None:
		xbar: npt.NDArray[np.double] = np.add.reduce(self.__sim[:-1], 0) / self.__N
		xr: npt.NDArray[np.double] = (1.0 + __class__.__RHO) * xbar - __class__.__RHO * self.__sim[-1]
		fxr: float = self.__func(xr)
		doshrink: bool = False
		if fxr < self.__fsim[0]:
			xe: npt.NDArray[np.double] = (1.0 + __class__.__RHO * __class__.__CHI) * xbar - __class__.__RHO * __class__.__CHI * self.__sim[-1]
			fxe: float = self.__func(xe)
			if fxe < fxr:
				self.__sim[-1] = xe
				self.__fsim[-1] = fxe
			else:
				self.__sim[-1] = xr
				self.__fsim[-1] = fxr
		else: # fsim[0] <= fxr
			if fxr < self.__fsim[-2]:
				self.__sim[-1] = xr
				self.__fsim[-1] = fxr
			else: # fxr >= fsim[-2]
				# Perform contraction
				if fxr < self.__fsim[-1]:
					xc: npt.NDArray[np.double] = (1.0 + __class__.__PSI * __class__.__RHO) * xbar - __class__.__PSI * __class__.__RHO * self.__sim[-1]
					fxc: float = self.__func(xc)
					if fxc <= fxr:
						self.__sim[-1] = xc
						self.__fsim[-1] = fxc
					else:
						doshrink = True
				else:
					# Perform an inside contraction
					xcc: npt.NDArray[np.double] = (1.0 - __class__.__PSI) * xbar + __class__.__PSI * self.__sim[-1]
					fxcc: float = self.__func(xcc)
					if fxcc < self.__fsim[-1]:
						self.__sim[-1] = xcc
						self.__fsim[-1] = fxcc
					else:
						doshrink = True
				if doshrink:
					for j in self.__one2np1:
						self.__sim[j] = self.__sim[0] + __class__.__SIGMA * (self.__sim[j] - self.__sim[0])
						self.__fsim[j] = self.__func(self.__sim[j])

		ind = np.argsort(self.__fsim)
		self.__sim = np.take(self.__sim, ind, 0)
		self.__fsim = np.take(self.__fsim, ind, 0)
		self.__assign_value_to_model(self.__sim[0])
		if self.__print_log:
			print_stuff(self.__model, self.__fsim[0], indent=1, flush=True)
		return self.__fsim[0]

	def __del__(self) -> None:
		# add gradient back for next iteration
		with torch.no_grad():
			for param in self.__model.parameters():
				param.requires_grad = True


def gpytorch_train(
	x: torch.Tensor,
	y: torch.Tensor,
	kernel: gpytorch.kernels.Kernel,
	loss_func: LossFuncType,
	print_log: bool = True,
	initial_values: list[torch.Tensor] = [],
	initial_value_search: bool = True
) -> gp.GPR | None:
	try:
		MAX_ITER: typing.Literal[50000] = 50000
		NOISE: float = float(gpytorch.settings.min_fixed_noise.value(torch.double) or 1e-8)
		model: gp.GPR = gp.GPR(x, y, gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.full(x.shape[:-1], NOISE)), kernel)
		param_indices: list[int] = [0]
		with torch.no_grad():
			for iParam, (name, param, constraint) in enumerate(model.named_parameters_and_constraints()):
				try:
					param[...] = initial_values[iParam].expand_as(param).clone()
				except (IndexError, RuntimeError): # unable to expand:
					if isinstance(constraint, gpytorch.constraints.Interval) and constraint.initial_value is not None:
						param[...] = constraint.initial_value.expand_as(param).clone()
					else:
						param.fill_(1.0)
				param.detach_().requires_grad = True
				param_indices.append(param_indices[-1] + param.numel())
				# set constraint
				# name_levels: list[str] = name.split(".")
				# num_name_levels: int = len(name_levels)
				# attr = model
				# for i in range(num_name_levels - 1):
				# 	attr = getattr(attr, name_levels[i])
				# attr.register_constraint(name_levels[-1], gp.NoConstraint())
		# train model
		assert model.likelihood is not None
		model.train()
		model.likelihood.train()
		# initial scan for [param/10, param*2]
		loss: torch.Tensor | None = None
		last_value: float = torch.inf
		if initial_value_search:
			init_params: torch.Tensor = torch.cat([constraint.transform(param) if isinstance(constraint, gpytorch.constraints.Interval) else param for _, param, constraint in model.named_parameters_and_constraints()])
			best_weight: float = 0.0
			for weight in (torch.arange(20.0) + 1.0) / 10.0:
				try:
					with torch.no_grad():
						for iParam, (_, param, constraint) in enumerate(model.named_parameters_and_constraints()):
							param[...] = (init_params[param_indices[iParam]:param_indices[iParam + 1]].reshape_as(param) * weight).detach()
							if isinstance(constraint, gpytorch.constraints.Interval):
								param[...] = constraint.inverse_transform(param).detach()
					loss = loss_func(model.cov, x, y)
					if print_log:
						print_stuff(model, loss.item(), math.nan, 1, start_str=f"last = {last_value:.15e},", flush=True)
					if loss.item() < last_value:
						last_value = loss.item()
						best_weight = weight.item()
				except RuntimeError: # NANs
					pass
			if best_weight == 0.0: # no non-NAN value
				print("Bad choice of initial value!", flush=True)
				return None
			else:
				with torch.no_grad():
					for iParam, (_, param, constraint) in enumerate(model.named_parameters_and_constraints()):
						param[...] = (init_params[param_indices[iParam]:param_indices[iParam + 1]].reshape_as(param) * best_weight).detach()
						if isinstance(constraint, gpytorch.constraints.Interval):
							param[...] = constraint.inverse_transform(param).detach()
				last_value = torch.inf
				loss = loss_func(model.cov, x, y)
		else:
			try:
				loss = loss_func(model.cov, x, y)
			except RuntimeError: # NANs
				print("Bad choice of initial value!", flush=True)
				return None
		# start optimization
		assert loss is not None
		learning_rate: float = -1.0
		num_iter: int = 0
		grad_opt: GradientOptimization = GradientOptimization(model, param_indices, loss_func, x, y, print_log, model.COV_NAME)
		non_grad_opt: NonGradientOptimization | None = None
		while num_iter < MAX_ITER:
			num_iter += 1
			if non_grad_opt is None:
				learning_rate, loss, to_end = grad_opt(learning_rate, loss)
				if to_end:
					break
				if math.isnan(learning_rate):
					non_grad_opt = NonGradientOptimization(model, loss_func, x, y, print_log, model.COV_NAME)
					if non_grad_opt.converged:
						break
			else:
				loss = torch.tensor(non_grad_opt())
				if non_grad_opt.converged:
					break
			if print_log or num_iter % (MAX_ITER // 100) == 0:
				print_stuff(
					model,
					loss.item(),
					learning_rate,
					1,
					start_str=f"Iter {num_iter} - last = {last_value:.15e},",
					flush=True
				)
			last_value = loss.item()
		if num_iter == MAX_ITER:
			print("Stop: Maximum number of iterations has been exceeded.")
		print_stuff(
			model,
			loss.item(),
			learning_rate,
			start_str=f"Iter {num_iter} - last = {last_value:.15e},",
			flush=print_log
		)
		# predict
		model.eval()
		model.likelihood.eval()
		return model
	except Exception as e:
		traceback.print_exception(e)
		return None
