#!/usr/bin/env python

# imports
import datetime
import gc
import math
import sys
import traceback
import typing
import warnings

import collections.abc
import gpytorch
import gpytorch.constraints

import matplotlib.axes
import matplotlib.colors
import matplotlib.cm
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import numpy.typing as npt
import scipy.optimize
import sklearn.cluster
import sklearn.exceptions
import sklearn.neighbors
import torch

torch.set_default_dtype(torch.float64)
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)
warnings.filterwarnings("ignore", category=gpytorch.utils.warnings.NumericalWarning)


class PNLogNorm(matplotlib.colors.Normalize):
	__slots__ = ("__abs_min", "__abs_max", "__log2_abs_min", "__log2_abs_max", "__log2_diff")

	def __init__(self, abs_min: float = sys.float_info.min, abs_max: float = sys.float_info.max, clip: bool = False) -> None:
		assert 0 < abs_min < abs_max
		super().__init__(-abs_max, abs_max, clip)
		self.__abs_min: float = abs_min
		self.__abs_max: float = abs_max
		self.__log2_abs_min: float = math.log2(abs_min)
		self.__log2_abs_max: float = math.log2(abs_max)
		self.__log2_diff: float = self.__log2_abs_max - self.__log2_abs_min

	@property
	def abs_min(self):
		return self.__abs_min

	@property
	def abs_max(self):
		return self.__abs_max

	def __call__(self, value: typing.Any, clip: bool | None = None) -> float | npt.NDArray[np.double]:
		def process_single_value(value: float, clip: bool) -> float:
			if abs(value) <= self.__abs_min:
				return 0.5
			sgn_half: float = 0.5 if value > 0 else -0.5
			if clip and abs(value) >= self.__abs_max:
				return sgn_half + 0.5
			else:
				return (math.log2(abs(value)) - self.__log2_abs_min) / self.__log2_diff * sgn_half + 0.5

		if clip is None:
			clip = self.clip
		result, is_scalar = super().process_value(value)
		if is_scalar == 1:
			return process_single_value(result.data.ravel()[0], clip)
		else:
			return np.ma.array(np.vectorize(process_single_value, otypes=[float])(result.data, clip), mask=result.mask)

	def __repr__(self) -> str:
		return __class__.__name__ + "({}, {})".format(self.__abs_min, self.__abs_max)
		
	def inverse(self, value: typing.Any) -> float | npt.NDArray[np.double]:
		def process_single_value(value: float) -> float:
			if value == 0.5:
				return 0.0
			return (-1.0 if value < 0.5 else 1.0) * math.exp2(abs(value - 0.5) * 2 * self.__log2_diff + self.__log2_abs_min)

		result, is_scalar = super().process_value(value)
		if is_scalar == 1:
			return process_single_value(result.data.ravel()[0])
		else:
			return np.ma.array(np.vectorize(process_single_value, otypes=[float])(result.data), mask=result.mask)


def get_centered_norm_and_levels(abs_max: float, cmap: str | matplotlib.colors.Colormap) -> tuple[matplotlib.colors.CenteredNorm, npt.NDArray[np.double], npt.NDArray[np.double]]:
	abs_max = abs(abs_max)
	if isinstance(cmap, str):
		cmap = matplotlib.colormaps[cmap]
	abs_max_10_level: float = math.pow(10, math.floor(math.log10(abs_max)))
	centered_norm: matplotlib.colors.CenteredNorm = matplotlib.colors.CenteredNorm(0.0, math.ceil(abs_max / abs_max_10_level) * abs_max_10_level, True)
	centered_levels: npt.NDArray[np.double] = np.linspace(-centered_norm.halfrange, centered_norm.halfrange, cmap.N // 8 * 8 + 1, True)
	return centered_norm, centered_levels, centered_levels[::cmap.N // 8]


def get_posneg_log_norm_and_levels(abs_min: float, abs_max: float, cmap: str | matplotlib.colors.Colormap) -> tuple[PNLogNorm, npt.NDArray[np.double], npt.NDArray[np.double]]:
	abs_min = abs(abs_min)
	abs_max = abs(abs_max)
	if abs_max == 0:
		abs_max = sys.float_info.min * 2
	if abs_min == 0:
		abs_min = sys.float_info.min
	abs_min = max(abs_min, abs_max * sys.float_info.epsilon)
	lb_log10: int = int(math.floor(math.log10(abs_min)))
	ub_log10: int = int(math.ceil(math.log10(abs_max)))
	lb_log10 += (ub_log10 - lb_log10) % 4
	if isinstance(cmap, str):
		cmap = matplotlib.colormaps[cmap]
	num_ticks: int = cmap.N // 8 * 8 + 3
	pn_log_norm: matplotlib.colors.Normalize = PNLogNorm(math.pow(10, lb_log10), math.pow(10, ub_log10), True)
	pn_log_levels: npt.NDArray[np.double] = np.zeros(num_ticks)
	pn_log_levels[cmap.N // 8 * 4 + 2:] = np.exp2(np.linspace(math.log2(pn_log_norm.abs_min), math.log2(pn_log_norm.abs_max), cmap.N // 8 * 4 + 1, True))
	pn_log_levels[cmap.N // 8 * 4::-1] = -pn_log_levels[cmap.N // 8 * 4 + 2:]
	return pn_log_norm, pn_log_levels, np.concatenate([pn_log_levels[0:cmap.N // 8 * 4:cmap.N // 8], pn_log_levels[cmap.N // 8 * 4 + 1:cmap.N // 8 * 4 + 2], pn_log_levels[cmap.N // 8 * 5 + 2::cmap.N // 8]])

class NoConstraint(gpytorch.constraints.Interval):
	def __init__(self, initial_value=None):
		super().__init__(
			lower_bound=-math.inf,
			upper_bound=math.inf,
			transform=lambda x: x,
			inv_transform=lambda x: x,
			initial_value=initial_value,
		)

	def __repr__(self) -> str:
		return __class__.__name__ + "()"

	def transform(self, tensor: torch.Tensor) -> torch.Tensor:
		return tensor

	def inverse_transform(self, transformed_tensor: torch.Tensor) -> torch.Tensor:
		return transformed_tensor


class PredType(typing.Protocol):
	def __call__(self, model: gpytorch.models.ExactGP, x_test: torch.Tensor, **kwargs: typing.Any) -> torch.Tensor:
		...


class MyGPR(gpytorch.models.ExactGP):
	def __init__(self, x: torch.Tensor, y: torch.Tensor, likelihood: gpytorch.likelihoods.Likelihood, kernel: gpytorch.kernels.Kernel):
		super().__init__(x, y, likelihood)
		self.mean: gpytorch.means.Mean = gpytorch.means.ZeroMean()
		self.cov: gpytorch.kernels.Kernel = kernel

	def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
		Mean = self.mean(x)
		assert isinstance(Mean, torch.Tensor)
		return gpytorch.distributions.MultivariateNormal(Mean, self.cov(x))


def get_covariance_matrix(model: gpytorch.models.ExactGP, x1: torch.Tensor, x2: torch.Tensor | None = None, cov_name: str = "cov", **kwargs) -> torch.Tensor:
	if x2 is None:
		x2 = x1
	assert hasattr(model, cov_name)
	cov = getattr(model, cov_name)
	assert isinstance(cov, gpytorch.kernels.Kernel)
	return cov(x1, x2, **kwargs).to_dense()


def square_solver(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
	val, vec = torch.linalg.eigh(A) # A = vec @ val.diag() @ vec.T
	return vec @ (1.0 / val).diag_embed() @ vec.mH @ B


def preconditioned_lstsq_solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
	_, s, vh = torch.linalg.svd(A) # A: m-n, _: m-m, s: n, vh: n-n
	q = vh.mH @ (1.0 / s.abs()).diag_embed()
	return q @ torch.linalg.lstsq(A @ q, B, rcond=0.0, driver="gels").solution


def default_predict(model: gpytorch.models.ExactGP, x_test: torch.Tensor, cov_name: str = "cov") -> torch.Tensor:
	assert model.train_inputs is not None
	x: torch.Tensor = model.train_inputs[0].detach()
	return get_covariance_matrix(model, x_test, x, cov_name) @ square_solver(get_covariance_matrix(model, x, cov_name=cov_name), model.train_targets.detach())


class LossFuncType(typing.Protocol):
	def __call__(self, model: gpytorch.models.ExactGP, **kwargs: typing.Any) -> torch.Tensor:
		...


def gpytorch_train(
	x: torch.Tensor,
	y: torch.Tensor,
	kernel: gpytorch.kernels.Kernel,
	loss_func: LossFuncType,
	print_log: bool = True,
	initial_values: list[torch.Tensor] = [],
	initial_value_search: bool = True,
	**kwargs
) -> MyGPR | None:
	def print_stuff(
		model: gpytorch.models.ExactGP,
		loss: float,
		learning_rate: float,
		indent: int = 0,
		print_grad: bool = False,
		hessian: torch.Tensor | None = None,
		start_str: str = "",
		end_str: str = "",
		flush: bool = print_log
	) -> None:
		def format_array(arr_name : str | None, arr: typing.Any) -> str:
			result: str = (arr_name + " = ") if arr_name else ""
			if isinstance(arr, torch.Tensor) or isinstance(arr, np.ndarray):
				result += " ".join(format_array(None, val.item()) for val in arr.ravel())
			elif isinstance(arr, collections.abc.Iterable):
				result += " ".join(format_array(None, item) for item in arr)
			elif isinstance(arr, complex):
				result += "{} + {}i".format(arr.real, arr.imag)
			else:
				result += str(arr)
			return result

		def print_model(model_print_grad: bool = False) -> None:
			param_name_fmt_str: str = "{}Parameter name: {{}}".format("\t" * indent)
			param_name: str
			param: torch.nn.Parameter
			constraint: gpytorch.constraints.Interval | None
			for param_name, param, constraint in model.named_parameters_and_constraints():
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

		print("{}{}loss = {:.15e}, lr = {}".format("\t" * indent, (start_str + " ") if start_str != "" else "", loss, learning_rate))
		print_model()
		if print_grad:
			print_model(True)
			if hessian is not None:
				print("\t" * indent, format_array("hessian", hessian.reshape(-1)), sep="")
				print("{}Cond(hessian): {}".format("\t" * indent, torch.linalg.cond(hessian).item()))
		print(("\t" * indent + end_str + "\n") if end_str != "" else "", end="", flush=flush)

	def non_gradient_optimize(model: MyGPR) -> MyGPR:
		def assign_value_to_model(x: npt.NDArray[np.double]) -> None:
			idx: int = 0
			for param in model.parameters():
				param.ravel()[...] = torch.from_numpy(x[idx:idx + param.numel()])
				idx += param.numel()

		def fun(x: npt.NDArray[np.double]) -> float:
			assign_value_to_model(x)
			return loss_func(model, **kwargs).item()
		
		def nongrad_print_stuff(intermediate_result: scipy.optimize.OptimizeResult) -> None:
			assign_value_to_model(intermediate_result.x)
			print_stuff(model, intermediate_result.fun, 0.0, 2)

		# initialize: remove gradient
		with torch.no_grad():
			for param in model.parameters():
				param.grad = None
				param.requires_grad = False
		# optimize
		result: scipy.optimize.OptimizeResult = scipy.optimize.minimize(
			fun,
			np.concatenate([p.data.detach().numpy().ravel() for p in model.parameters()]),
			method="Nelder-Mead",
			callback=nongrad_print_stuff if print_log else None,
			options={"maxiter": MAX_ITER, "fatol": FTOL, "xatol": GTOL_SQRT}
		)
		assign_value_to_model(result.x)
		print_stuff(model, result.fun, 0.0, 0, start_str="Iter {},".format(result.nit), end_str=result.message)
		# finalize
		with torch.no_grad():
			for param in model.parameters():
				param.requires_grad = True
		model.eval()
		assert model.likelihood is not None
		model.likelihood.eval()
		return model

	def optimize_with_learning_rate(
		model: gpytorch.models.ExactGP,
		combined_parameters: torch.Tensor,
		change: torch.Tensor,
		param_indices: list[int],
		last_loss: float,
		method_name: str,
		initial_learning_rate: float
	) -> tuple[torch.Tensor, float]:
		def change_param(target: torch.Tensor) -> None:
			with torch.no_grad():
				for iParam, param in enumerate(model.parameters()):
					param[...] = target[param_indices[iParam]:param_indices[iParam + 1]].reshape_as(param).detach()

		def return_back() -> tuple[torch.Tensor, float]:
			if print_log:
				print("\tNo stepping Forward for {}.".format(method_name))
			change_param(combined_parameters)
			return torch.tensor(last_loss), initial_learning_rate

		learning_rate: float = initial_learning_rate
		change_param(combined_parameters - learning_rate * change)
		learning_rate_threshold: float = sys.float_info.epsilon * (combined_parameters.norm() / change.norm()).item() # lr * |change| should > double precision * |param|
		loss: torch.Tensor
		while learning_rate >= learning_rate_threshold:
			try:
				loss = loss_func(model, **kwargs)
				break
			except RuntimeError: # NANs
				learning_rate /= 2.0
		if learning_rate < learning_rate_threshold:
			# based on current learning rate, all parameters have changes smaller than double precision
			return return_back()
		if print_log:
			print_stuff(model, loss.item(), learning_rate, 2, start_str="last = {:.15e},".format(last_loss), flush=True)
		while loss.isnan().item() or loss.item() >= last_loss:
			learning_rate /= 2.0
			change_param(combined_parameters - learning_rate * change)
			try:
				loss = loss_func(model, **kwargs)
			except RuntimeError: # NANs
				return return_back()
			if print_log:
				print_stuff(model, loss.item(), learning_rate, 2, start_str="last = {:.15e},".format(last_loss), flush=True)
			if learning_rate < learning_rate_threshold:
				# based on current learning rate, all parameters have changes smaller than double precision
				if loss.item() >= last_loss:
					return return_back()
				else:
					if print_log:
						print("\tNo stepping Forward for {}.".format(method_name))
					break
		return loss, learning_rate

	MAX_ITER: typing.Literal[50000] = 50000
	FTOL: float = 1e-09
	GTOL_SQRT: float = 1e-5
	GTOL: float = GTOL_SQRT ** 2
	NOISE: float = gpytorch.settings.min_fixed_noise.value(torch.double) or 1e-8
	model: MyGPR = MyGPR(x, y, gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.full(x.shape[:-1], NOISE)), kernel)
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
			name_levels: list[str] = name.split(".")
			num_name_levels: int = len(name_levels)
			attr = model
			for i in range(num_name_levels - 1):
				attr = getattr(attr, name_levels[i])
			attr.register_constraint(name_levels[-1], NoConstraint())
	# train model
	assert model.likelihood is not None
	model.train()
	model.likelihood.train()
	finish_early: bool = False
	# initial scan for [param/10, param*2]
	loss: torch.Tensor
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
				loss = loss_func(model, **kwargs)
				if print_log:
					print_stuff(model, loss.item(), 1.0, 1, start_str="last = {:.15e},".format(last_value), flush=True)
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
			loss = loss_func(model, **kwargs)
	else:
		try:
			loss = loss_func(model, **kwargs)
		except RuntimeError: # NANs
			print("Bad choice of initial value!", flush=True)
			return None
	# start optimization
	learning_rate: float = -1.0
	hessian: torch.Tensor = torch.eye(param_indices[-1], dtype=torch.double)
	num_iter: int = 0
	for i in range(1, MAX_ITER + 1):
		# calculate gradient and hessian
		for iParam, param in enumerate(model.parameters()):
			param.grad = torch.autograd.grad(loss, param, None, True, True, True, True, False, True)[0] # same shape of param
			if torch.any(torch.isnan(param.grad)).item() or torch.any(torch.isinf(param.grad)).item():
				return non_gradient_optimize(model) # gradient failed
			for iGrad, grad_elm in enumerate(param.grad.reshape(-1)):
				for jParam, param_for_grad in enumerate(model.parameters()):
					hessian[param_indices[iParam] + iGrad, param_indices[jParam]:param_indices[jParam + 1]] = torch.autograd.grad(grad_elm, param_for_grad, None, True, False, True, True, False, True)[0]
		if torch.any(torch.isnan(hessian)).item() or torch.any(torch.isinf(hessian)).item():
			return non_gradient_optimize(model) # gradient failed
		hessian = (hessian + hessian.T) / 2.0 # symmetrize
		grad_combined: torch.Tensor = torch.cat([(param.grad if param.grad is not None else torch.zeros_like(param)).reshape(-1) for param in model.parameters()])
		param_combined: torch.Tensor = torch.cat([param.data.reshape(-1) for param in model.parameters()])
		if learning_rate == -1.0:
			learning_rate = (param_combined.norm() / grad_combined.norm()).item()
		# stopping criteria
		grad_sqnm: float = torch.sum(grad_combined ** 2).item()
		if grad_sqnm < GTOL:
			finish_early = True
			num_iter = i
			print("Convergence: |Gradient| = {} <= GTOL = {}".format(math.sqrt(grad_sqnm), GTOL_SQRT))
			break
		# prepare
		newton_change: torch.Tensor = square_solver(hessian, grad_combined)
		if learning_rate < 1.0:
			learning_rate *= 2.0
		if print_log:
			print_stuff(
				model,
				loss.item(),
				learning_rate,
				1,
				True,
				hessian,
				"Iter {} - last = {:.15e},".format(i, last_value),
				"",
				True
			)
		# Newton
		last_value = loss.item()
		old_learning_rate: float = learning_rate
		for change, method in zip([newton_change, grad_combined], ["Newton Method", "Gradient Descent"]):
			loss, learning_rate = optimize_with_learning_rate(
				model,
				param_combined,
				change,
				param_indices,
				last_value,
				method,
				old_learning_rate
			)
			if loss.item() < last_value:
				break
		if loss.item() >= last_value: # tried all method and no one can step forward
			finish_early = True
			num_iter = i
			print("Stop: No stepping Forward")
			break
		if abs(last_value - loss.item()) / max(abs(last_value), abs(loss.item()), 1.0) < FTOL:
			finish_early = True
			num_iter = i
			print("Convergence: |f_i - f_{{i+1}}| = {} / {} <= FTOL = {}".format(abs(last_value - loss.item()), max(abs(last_value), abs(loss.item()), 1.0), FTOL))
			break
	if not finish_early:
		print("Stop: Total No. iterations reached limit.")
		num_iter = MAX_ITER
	print_stuff(
		model,
		loss.item(),
		learning_rate,
		0,
		True,
		hessian,
		"Iter {} - last = {:.15e},".format(num_iter, last_value)
	)
	# predict
	model.eval()
	model.likelihood.eval()
	return model


def loocv_se(
	model: gpytorch.models.ExactGP,
	x_all: torch.Tensor,
	y_all: torch.Tensor,
	cov_name: str = "cov",
	**kwargs
) -> torch.Tensor:
	assert model.train_inputs is not None
	x: torch.Tensor = model.train_inputs[0]
	# k_cho = torch.linalg.cholesky(get_covariance_matrix(model, x, cov_name=cov_name) + NOISE * torch.eye(x.shape[-2]))
	# return ((torch.cholesky_solve(model.train_targets[..., None], k_cho)[..., 0] / torch.diagonal(torch.cholesky_inverse(k_cho), 0, -2, -1)) ** 2).sum() + ((default_predict(model, x_all, cov_name) - y_all) ** 2).sum()
	cov_mat: torch.Tensor = get_covariance_matrix(model, x, cov_name=cov_name)
	return ((square_solver(cov_mat, model.train_targets) / torch.diagonal(square_solver(cov_mat, torch.eye(x.shape[-2])), 0, -2, -1)) ** 2).sum() + ((default_predict(model, x_all, cov_name) - y_all) ** 2).sum()


class ApproxPredType(typing.Protocol):
	def __call__(
		self,
		model: gpytorch.models.ExactGP,
		x_test: torch.Tensor,
		x_all: torch.Tensor,
		y_all: torch.Tensor,
		**kwargs: typing.Any
	) -> torch.Tensor:
		...


def approx_se(
	model: gpytorch.models.ExactGP,
	pred: ApproxPredType,
	x_all: torch.Tensor,
	y_all: torch.Tensor,
	**kwargs
) -> torch.Tensor:
	return ((y_all - pred(model, x_all, x_all, y_all, **kwargs)) ** 2).sum()


def pinv_pred(
	model: gpytorch.models.ExactGP,
	x_test: torch.Tensor,
	x_all: torch.Tensor,
	y_all: torch.Tensor,
	cov_name: str = "cov",
	**kwargs
) -> torch.Tensor:
	assert model.train_inputs is not None
	x: torch.Tensor = model.train_inputs[0]
	return (get_covariance_matrix(model, x_test, x, cov_name) @ preconditioned_lstsq_solve(get_covariance_matrix(model, x_all, x, cov_name), y_all[..., None]))[..., 0]


def main() -> None:
	gc.set_debug(gc.DEBUG_UNCOLLECTABLE | gc.DEBUG_SAVEALL | gc.DEBUG_STATS)
	if gc.isenabled():
		gc.disable()
	# constants
	FIGSIZE: list[float] = matplotlib.rcParams["figure.figsize"]
	N_GAUSS = 20
	DIM: typing.Literal[2] = 2
	MAX: float = 20.0
	MIN: float = -MAX
	N_GRIDS = 201
	COLOR_RATIO: float = 1.1
	x1, x2 = np.meshgrid(np.linspace(MIN, MAX, N_GRIDS, dtype=np.double), np.linspace(MIN, MAX, N_GRIDS, dtype=np.double))
	x_test: npt.NDArray[np.double] = np.concatenate((x1[..., np.newaxis], x2[..., np.newaxis]), axis=-1).reshape(-1, DIM)
	# distribution
	rng: np.random.Generator = np.random.Generator(np.random.MT19937(0))
	# CENTER: npt.NDArray[np.double] = rng.uniform(-MAX / 10, MAX / 10, (N_GAUSS, DIM))
	# WIDTH: npt.NDArray[np.double] = rng.uniform(0.1, MAX / 10, (N_GAUSS, DIM))
	# WEIGHT: npt.NDArray[np.double] = rng.uniform(-1.0, 1.0, N_GAUSS)
	# print("Centers are at\n", CENTER, sep=None)
	# print("Weights of Gaussian are\n", WEIGHT, sep=None, flush=True)
	WIDTH: npt.NDArray[np.double] = np.array([1.0, 0.5])
	print("Widths are\n", WIDTH, sep=None)

	def dist_func(x: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
		if x.ndim == 1:
			x = x[np.newaxis]
		# return np.sum(WEIGHT * np.exp(-np.sum(np.square((x[:, np.newaxis, :] - CENTER) / WIDTH), axis=-1) / 2.0), axis=-1)
		return np.exp(-np.sum((x / WIDTH) ** 2, -1) / 2.0)

	y_test: npt.NDArray[np.double] = np.reshape(dist_func(x_test), (N_GRIDS, N_GRIDS))
	assert np.count_nonzero(y_test == 0.0) != y_test.size
	ABS_MAX_LIMIT: float = np.max(np.abs(y_test))
	ABS_MIN_LIMIT: float = np.min(np.abs(y_test[y_test != 0.0]))
	CMAP: str = 'seismic'
	CTR_NORM: matplotlib.colors.Normalize
	CTR_LEVELS: npt.NDArray[np.double]
	CTR_TICKS: npt.NDArray[np.double]
	CTR_NORM, CTR_LEVELS, CTR_TICKS = get_centered_norm_and_levels(ABS_MAX_LIMIT * COLOR_RATIO, CMAP)
	LOG_NORM: matplotlib.colors.Normalize
	LOG_LEVELS: npt.NDArray[np.double]
	LOG_TICKS: npt.NDArray[np.double]
	LOG_NORM, LOG_LEVELS, LOG_TICKS = get_posneg_log_norm_and_levels(ABS_MIN_LIMIT, ABS_MAX_LIMIT, CMAP)

	def limited_region(arr2d: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
		return arr2d[N_GRIDS // 8 * 3:N_GRIDS // 8 * 5 + 1, N_GRIDS // 8 * 3:N_GRIDS // 8 * 5 + 1]

	def draw_function() -> None:
		fig: matplotlib.figure.Figure = plt.figure(figsize=(FIGSIZE[0] * 3, FIGSIZE[1]))
		ax0, ax1, ax2 = fig.subplots(ncols=3)
		ax0.contourf(limited_region(x1), limited_region(x2), limited_region(y_test), CTR_LEVELS, cmap=CMAP, norm=CTR_NORM)
		# ax0.scatter(CENTER[:, 0], CENTER[:, 1], 20 * np.abs(WEIGHT), 'black')
		ax1.contourf(x1, x2, y_test, CTR_LEVELS, cmap=CMAP, norm=CTR_NORM)
		# ax1.scatter(CENTER[:, 0], CENTER[:, 1], 10, 'black')
		fig.colorbar(matplotlib.cm.ScalarMappable(CTR_NORM, CMAP), ax=[ax0, ax1], ticks=CTR_TICKS)
		ax2.contourf(x1, x2, y_test, LOG_LEVELS, cmap=CMAP, norm=LOG_NORM)
		fig.colorbar(matplotlib.cm.ScalarMappable(LOG_NORM, CMAP), ax=ax2, ticks=LOG_TICKS, format="%+.1e")
		fig.savefig("distribution.png")
		plt.close(fig)
		del fig

	draw_function()
	print(datetime.datetime.now(), flush=True)
	gc.collect()

	# sample points
	def random_walk() -> tuple[npt.NDArray[np.double], npt.NDArray[np.double]]:
		num_plot: int = 0
		def draw_sample_pts(central_pts: npt.NDArray[np.double], extra_pts: npt.NDArray[np.double]) -> None:
			fig: matplotlib.figure.Figure = plt.figure(figsize=(FIGSIZE[0] * 3, FIGSIZE[1]))
			ax0, ax1, ax2 = fig.subplots(ncols=3)
			ax0.contourf(limited_region(x1), limited_region(x2), limited_region(y_test), CTR_LEVELS, cmap=CMAP, norm=CTR_NORM)
			ax0.scatter(np.clip(extra_pts[:, 0], MIN / 4, MAX / 4), np.clip(extra_pts[:, 1], MIN / 4, MAX / 4), 1, 'green')
			ax0.scatter(np.clip(central_pts[:, 0], MIN / 4, MAX / 4), np.clip(central_pts[:, 1], MIN / 4, MAX / 4), 10, 'black')
			ax1.contourf(x1, x2, y_test, CTR_LEVELS, cmap=CMAP, norm=CTR_NORM)
			fig.colorbar(matplotlib.cm.ScalarMappable(CTR_NORM, CMAP), ax=[ax0, ax1], ticks=CTR_TICKS)
			ax2.contourf(x1, x2, y_test, LOG_LEVELS, cmap=CMAP, norm=LOG_NORM)
			fig.colorbar(matplotlib.cm.ScalarMappable(LOG_NORM, CMAP), ax=ax2, ticks=LOG_TICKS, format="%.1e")
			for ax in [ax1, ax2]:
				ax.scatter(extra_pts[:, 0], extra_pts[:, 1], 1, 'green')
				ax.scatter(central_pts[:, 0], central_pts[:, 1], 10, 'black')
			fig.savefig("sample_points_{}.png".format(num_plot))
			plt.close(fig)
			del fig

		# PTS_RATIO = 10
		# EXTRA_RATIO = 50
		# STEP_SIZE: float = np.sqrt(np.max(scipy.spatial.distance.pdist(CENTER, 'sqeuclidean')))
		# NUM_STEP = 1000
		# BOX_SIZE: float = MAX - MIN
		# print("Step size for random walk is {}".format(STEP_SIZE))
		# print("Number of steps for random walk is {}".format(NUM_STEP))
		# central_pts: npt.NDArray[np.double] = np.repeat(CENTER, PTS_RATIO, 0)
		# weights: npt.NDArray[np.double] = dist_func(central_pts)
		# acc_ratio: npt.NDArray[np.double] = np.zeros(weights.shape)
		# for _ in range(NUM_STEP):
		# 	new_pts: npt.NDArray[np.double] = central_pts + rng.uniform(-STEP_SIZE, STEP_SIZE, central_pts.shape)
		# 	new_pts = np.where(new_pts < MIN, new_pts + BOX_SIZE, new_pts)
		# 	new_pts = np.where(new_pts > MAX, new_pts - BOX_SIZE, new_pts)
		# 	new_weights: npt.NDArray[np.double] = dist_func(new_pts)
		# 	acc: npt.NDArray[np.bool_] = np.power(np.abs(new_weights / weights), 2.0) > rng.uniform(0, 1, weights.shape)
		# 	acc_ratio += acc
		# 	central_pts = np.where(acc[:, np.newaxis], new_pts, central_pts)
		# 	weights = np.where(acc, new_weights, weights)
		# acc_ratio /= NUM_STEP
		# print("Acceptance ratio for {} central points is within [{}, {}]".format(central_pts.size // DIM, np.min(acc_ratio), np.max(acc_ratio)), flush=True)
		N_PTS = 256
		EXTRA_RATIO = 50
		central_pts: npt.NDArray[np.double] = rng.multivariate_normal(np.zeros(DIM), np.diag(WIDTH ** 2), N_PTS, "raise", method="cholesky")
		var: npt.NDArray[np.double] = np.diag(np.var(central_pts, 0))
		extra_pts: npt.NDArray[np.double] = np.concatenate([rng.multivariate_normal(pt, var, EXTRA_RATIO, "raise", method="cholesky") for pt in central_pts])
		draw_sample_pts(central_pts, extra_pts)
		num_plot += 1
		kmeans_center: sklearn.cluster.KMeans = sklearn.cluster.KMeans(central_pts.shape[0], init="k-means++", n_init="auto", random_state=np.random.RandomState(rng.bit_generator), algorithm="lloyd").fit(np.concatenate([central_pts, extra_pts]))
		central_pts[...] = kmeans_center.cluster_centers_
		var = np.diag(np.var(kmeans_center.cluster_centers_, 0))
		kmeans_extra: sklearn.cluster.KMeans = sklearn.cluster.KMeans(central_pts.shape[0] * EXTRA_RATIO, init="k-means++", n_init="auto", random_state=np.random.RandomState(rng.bit_generator), algorithm="lloyd").fit(np.concatenate([rng.multivariate_normal(pt, var, EXTRA_RATIO ** 2, "raise", method="cholesky") for pt in kmeans_center.cluster_centers_]))
		extra_pts[...] = kmeans_extra.cluster_centers_
		draw_sample_pts(central_pts, extra_pts)
		num_plot += 1
		kmeans_center.fit(np.concatenate([central_pts, extra_pts]))
		var = np.diag(np.var(kmeans_center.cluster_centers_, 0))
		extra_pts = kmeans_extra.fit(np.concatenate([extra_pts, kmeans_extra.fit(np.concatenate([rng.multivariate_normal(pt, var, EXTRA_RATIO ** 2, "raise", method="cholesky") for pt in kmeans_center.cluster_centers_])).cluster_centers_])).cluster_centers_
		draw_sample_pts(central_pts, extra_pts)
		num_plot += 1
		return central_pts, extra_pts

	x, extra_x = random_walk()
	y: npt.NDArray[np.double] = dist_func(x)
	extra_y: npt.NDArray[np.double] = dist_func(extra_x)
	x_all: npt.NDArray[np.double] = np.concatenate((x, extra_x))
	y_all: npt.NDArray[np.double] = np.concatenate((y, extra_y))
	x_t: torch.Tensor = torch.from_numpy(x)
	y_t: torch.Tensor = torch.from_numpy(y)
	extra_x_t: torch.Tensor = torch.from_numpy(extra_x)
	extra_y_t: torch.Tensor = torch.from_numpy(extra_y)
	x_all_t: torch.Tensor = torch.from_numpy(x_all)
	y_all_t: torch.Tensor = torch.from_numpy(y_all)
	x_test_t: torch.Tensor = torch.from_numpy(x_test)
	y_test_t: torch.Tensor = torch.from_numpy(y_test)

	print(datetime.datetime.now(), flush=True)
	gc.collect()

	def plot(pred: npt.NDArray[np.double], name: str) -> None:
		diff: npt.NDArray[np.double] = pred - y_test
		mse: float = float(np.average(diff ** 2))
		mae: float = float(np.average(np.abs(diff)))
		max_e: float = np.max(np.abs(diff))
		print("Mean squared error =", mse)
		print("Mean absolute error =", mae)
		print("Maximum error =", max_e)
		if (mse_contribute := max_e ** 2 / (mse * y_test.size) ) > 0.01:
			print("Maximum error point contribute {:.2f}% of MSE".format(mse_contribute * 100))
		if (mae_contribute := max_e / (mae * y_test.size)) > 0.01:
			print("Maximum error point contribute {:.2f}% of MAE".format(mae_contribute * 100))
		print("", end="", flush=True)
		max_ind: tuple = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
		fig: matplotlib.figure.Figure = plt.figure(figsize=(FIGSIZE[0] * 3, FIGSIZE[1] * 4))
		axs: np.ndarray[collections.abc.Sequence[collections.abc.Sequence[matplotlib.axes.Axes]], np.dtype[np.object_]] = fig.subplots(4, 3)
		titles: tuple = ("Exact Density", "Density Fit")
		data: tuple = (y_test, pred)
		for i in range(2):
			axs[0, i].set_title(titles[i])
			axs[0, i].contourf(limited_region(x1), limited_region(x2), limited_region(data[i]), CTR_LEVELS, cmap=CMAP, norm=CTR_NORM)
			axs[1, i].contourf(x1, x2, data[i], CTR_LEVELS, cmap=CMAP, norm=CTR_NORM)
			axs[2, i].contourf(x1, x2, data[i], CTR_LEVELS, cmap=CMAP, norm=CTR_NORM)
			axs[2, i].scatter(extra_x[:, 0], extra_x[:, 1], 1, 'green')
			axs[2, i].scatter(x[:, 0], x[:, 1], 10, 'black')
			axs[3, i].contourf(x1, x2, data[i], LOG_LEVELS, cmap=CMAP, norm=LOG_NORM)
		fig.colorbar(matplotlib.cm.ScalarMappable(CTR_NORM, CMAP), ax=axs[:3, :2].ravel().tolist(), ticks=CTR_TICKS)
		fig.colorbar(matplotlib.cm.ScalarMappable(LOG_NORM, CMAP), ax=axs[3, :2].tolist(), ticks=LOG_TICKS, format="%.1e")

		ctr_norm, ctr_levels, ctr_ticks = get_centered_norm_and_levels(np.max(np.abs(diff)) * COLOR_RATIO, CMAP)
		pn_log_norm, pn_log_levels, pn_log_ticks = get_posneg_log_norm_and_levels(np.min(np.abs(diff[diff != 0])), np.max(np.abs(diff)), CMAP)
		axs[0, 2].set_title("Error\nMaximum at ({:.2f}, {:.2f})".format(x1[max_ind], x2[max_ind]))
		axs[0, 2].contourf(limited_region(x1), limited_region(x2), limited_region(diff), ctr_levels, cmap=CMAP, norm=ctr_norm)
		axs[0, 2].scatter(np.clip(x1[max_ind], MIN / 4, MAX / 4), np.clip(x2[max_ind], MIN / 4, MAX / 4), 10, 'black')
		axs[1, 2].contourf(x1, x2, diff, ctr_levels, cmap=CMAP, norm=ctr_norm)
		axs[1, 2].scatter(x1[max_ind], x2[max_ind], 10, 'black')
		axs[2, 2].contourf(x1, x2, diff, ctr_levels, cmap=CMAP, norm=ctr_norm)
		axs[2, 2].scatter(extra_x[:, 0], extra_x[:, 1], 1, 'green')
		axs[2, 2].scatter(x[:, 0], x[:, 1], 10, 'black')
		axs[3, 2].contourf(x1, x2, diff, pn_log_levels, cmap=CMAP, norm=pn_log_norm)
		fig.colorbar(matplotlib.cm.ScalarMappable(ctr_norm, CMAP), ax=axs[:3, 2].tolist(), ticks=ctr_ticks)
		fig.colorbar(matplotlib.cm.ScalarMappable(pn_log_norm, CMAP), ax=axs[3, 2], ticks=pn_log_ticks, format="%.1e")
		fig.savefig(name + ".png")
		plt.close(fig)
		del fig

	def gpytorch_gpr(model: MyGPR, x_test: torch.Tensor = x_test_t, predictor: PredType | None = None, **kwargs) -> npt.NDArray[np.double]:
		try:
			result: torch.Tensor = predictor(model, x_test, **kwargs) if predictor is not None else default_predict(model, x_test)
			return result.detach().numpy().reshape(N_GRIDS, N_GRIDS)
		except Exception as e:
			traceback.print_exception(e)
			return np.zeros((N_GRIDS, N_GRIDS), np.double)

	def se_trials() -> None:
		for erf, pred in zip([loocv_se, approx_se], [None, pinv_pred]):
			print("gpytorch GPR using RBF kernel and {}".format(erf.__name__.replace('_', ' ')), flush=True)
			if (model := gpytorch_train(x_t, y_t, gpytorch.kernels.RBFKernel(DIM), erf, initial_values=[torch.from_numpy(WIDTH)], x_all=x_all_t, y_all=y_all_t, pred=pred)) is not None:
				plot(gpytorch_gpr(model, predictor=pred, x_all=x_all_t, y_all=y_all_t), erf.__name__)

	se_trials()
	print(datetime.datetime.now(), flush=True)
	gc.collect()

	def plot_lengthscale_error(
		name: str,
		min: float | npt.NDArray[np.double] = 0.25,
		max: float | npt.NDArray[np.double] = 1.25,
		n_grids_to_plot: int = 101,
		predictor: PredType | None = None,
		erf: LossFuncType = approx_se,
		**kwargs
	) -> MyGPR:
		# NOISE = gpytorch.settings.min_fixed_noise.value(torch.double)
		# assert isinstance(NOISE, float)
		# if not isinstance(min, np.ndarray):
		# 	min = np.full(2, min, np.double)
		# if not isinstance(max, np.ndarray):
		# 	max = np.full(2, max, np.double)
		# xv: npt.NDArray[np.double]
		# pv: npt.NDArray[np.double]
		# xv, pv = np.meshgrid(np.linspace(min[0], max[0], n_grids_to_plot), np.linspace(min[1], max[1], n_grids_to_plot))
		# lengths: npt.NDArray[np.double] = np.concatenate((xv[..., np.newaxis], pv[..., np.newaxis]), axis=-1)
		# values: npt.NDArray[np.double] = np.zeros((3, n_grids_to_plot, n_grids_to_plot))
		# for i in range(n_grids_to_plot):
		# 	for j in range(n_grids_to_plot):
		# 		diff: npt.NDArray[np.double] = gpytorch_gpr(
		# 			MyGPR(
		# 				x_t,
		# 				y_t,
		# 				gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.full(x.shape[:-1], NOISE)),
		# 				gpytorch.kernels.RBFKernel(DIM, lengthscale_constraint=NoConstraint(torch.from_numpy(lengths[i, j])))
		# 			).eval(),
		# 			predictor=predictor,
		# 			**kwargs
		# 		) - y_test
		# 		values[0, i, j] = np.average(diff ** 2)
		# 		values[1, i, j] = np.average(np.abs(diff))
		# 		values[2, i, j] = np.max(np.abs(diff))
		# fig, axs = plt.subplots(1, 3, figsize=(FIGSIZE[0] * 3, FIGSIZE[1]))
		# titles: list[str] = ['Mean Squared Error', 'Mean Absolute Error', 'Maximum Error']
		# for i in range(3):
		# 	ax: matplotlib.axes.Axes = axs[i]
		# 	lg_lb: float = np.floor(np.log10(np.min(values[i])))
		# 	lg_ub: float = np.ceil(np.log10(np.max(values[i])))
		# 	norm: matplotlib.colors.LogNorm = matplotlib.colors.LogNorm(np.power(10.0, lg_lb), np.power(10.0, lg_ub), True)
		# 	levels: npt.NDArray[np.double] = np.power(10.0, np.linspace(lg_lb, lg_ub, matplotlib.colormaps[CMAP].N, True))
		# 	ax.contourf(xv, pv, values[i], levels, cmap=CMAP, norm=norm)
		# 	argmin_ind = np.unravel_index(np.argmin(values[i]), values[i].shape)
		# 	ax.set_title(titles[i] + '\nMinimum at {}'.format(lengths[argmin_ind]) + '\nMinimum is {}'.format(values[i][argmin_ind]))
		# 	ax.scatter(lengths[argmin_ind][0], lengths[argmin_ind][1], 10, "black")
		# 	ax.set_xlabel('Length of x')
		# 	ax.set_ylabel('Length of p')
		# 	fig.colorbar(matplotlib.cm.ScalarMappable(cmap=CMAP, norm=norm), ax=ax, ticks=np.power(10.0, np.linspace(lg_lb, lg_ub, int(lg_ub - lg_lb) + 1, True)), format="%.1e")
		# fig.savefig("length_{}.png".format(name))
		# plt.close(fig)
		# del fig
		model = gpytorch_train(
			x_t,
			y_t,
			gpytorch.kernels.RBFKernel(DIM),
			erf,
			# initial_values=[torch.from_numpy(lengths[np.unravel_index(np.argmin(values[0]), values[0].shape)])],
			initial_values=[torch.from_numpy(WIDTH)],
			# initial_value_search=False,
			pred=predictor,
			**kwargs
		)
		assert model is not None
		plot(gpytorch_gpr(model, predictor=predictor, **kwargs), "opt_{}".format(name))
		return model

	pinv_model: MyGPR = plot_lengthscale_error("pinv", predictor=pinv_pred, x_all=x_all_t, y_all=y_all_t)
	print(datetime.datetime.now(), flush=True)
	gc.collect()

	# loocv_model: MyGPR = plot_lengthscale_error("loocv", erf=loocv_se, x_all=x_all_t, y_all=y_all_t)
	# print(datetime.datetime.now(), flush=True)
	# gc.collect()

	# def pinv_local_pred_v1(
	# 	model: gpytorch.models.ExactGP,
	# 	x_test: torch.Tensor,
	# 	x_all: torch.Tensor,
	# 	y_all: torch.Tensor,
	# 	variance_cutoff_factor: float,
	# 	num_neighbor: int,
	# 	nn: sklearn.neighbors.NearestNeighbors | None = None,
	# 	neighbor_ind: torch.Tensor | None = None,
	# 	**kwagrs
	# ) -> torch.Tensor:
	# 	assert model.train_inputs is not None
	# 	x: torch.Tensor = model.train_inputs[0]
	# 	kxm: torch.Tensor = get_covariance_matrix(model, x_test, x)
	# 	k_inv_y: torch.Tensor = preconditioned_lstsq_solve(get_covariance_matrix(model, x_all, x), y_all)
	# 	pred: torch.Tensor = kxm @ k_inv_y
	# 	var: torch.Tensor = get_covariance_matrix(model, x_test, diag=True) - torch.einsum("ij,jk,ik->i", kxm, square_solver(get_covariance_matrix(model, x), torch.eye(x.shape[-2])), kxm) # diagonal only
	# 	sigma_f2: float = (k_inv_y.reshape(1, -1) @ get_covariance_matrix(model, x) @ k_inv_y.reshape(-1, 1)).item() / y_all.numel()
	# 	result: torch.Tensor = pred.clone()
	# 	need_local_gp: torch.Tensor = pred ** 2 < variance_cutoff_factor * sigma_f2 * var
	# 	num_need_local_gp: int = int(torch.count_nonzero(need_local_gp).item())
	# 	print("{} inputs, where {} need local GP.".format(x_test.numel() // DIM, num_need_local_gp))
	# 	if num_need_local_gp > 0:
	# 		if neighbor_ind is not None and neighbor_ind.shape == x_test.shape[:-1] + (num_neighbor,):
	# 			neighbor_ind_used: torch.Tensor = neighbor_ind[need_local_gp]
	# 		elif nn is not None and nn.get_params()["n_neighbors"] == num_neighbor:
	# 			neighbor_ind_used = torch.from_numpy(nn.kneighbors(x_test[need_local_gp].detach().numpy(), return_distance=False))
	# 		else:
	# 			nn = sklearn.neighbors.NearestNeighbors(n_neighbors=num_neighbor, n_jobs=-1).fit(x_all.detach().numpy())
	# 			neighbor_ind_used = torch.from_numpy(nn.kneighbors(x_test[need_local_gp].detach().numpy(), return_distance=False))
	# 		result[need_local_gp] = (get_covariance_matrix(model, x_test[need_local_gp, None], x_all[neighbor_ind_used]) @ square_solver(get_covariance_matrix(model, x_all[neighbor_ind_used]), y_all[neighbor_ind_used, None])).reshape(-1)
	# 	return result

	# def pinv_local_pred_trial_v1() -> None:
	# 	print("gpytorch GPR using RBF kernel with pseudo inverse and local GP, pseudo inverse for parameters only")
	# 	print("Check how center local GP should go\n")
	# 	num_neighbor_trial: int = 32
	# 	nn: sklearn.neighbors.NearestNeighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=num_neighbor_trial, n_jobs=-1).fit(x_all)
	# 	for factor in ["1e0", "1e1", "1e2", "1e3", "1e4", "1e5", "1e6", "1e7", "1e8", math.inf]:
	# 		print("Cutoff = {}".format(factor), flush=True)
	# 		plot(gpytorch_gpr(pinv_model, predictor=pinv_local_pred_v1, x_all=x_all_t, y_all=y_all_t, variance_cutoff_factor=float(factor), num_neighbor=num_neighbor_trial, nn=nn), "pinv_local_v1_{}".format(factor))

	# pinv_local_pred_trial_v1()
	# print(datetime.datetime.now(), flush=True)
	# gc.collect()

	# def pinv_local_pred_v2_1(
	# 	model: gpytorch.models.ExactGP,
	# 	x_test: torch.Tensor,
	# 	x_all: torch.Tensor,
	# 	y_all: torch.Tensor,
	# 	dist_ratio: float,
	# 	num_neighbor: int,
	# 	nn: sklearn.neighbors.NearestNeighbors | None = None,
	# 	neighbor_ind: torch.Tensor | None = None,
	# 	**kwagrs
	# ) -> torch.Tensor:
	# 	result: torch.Tensor = pinv_pred(model, x_test, x_all, y_all).clone()
	# 	if neighbor_ind is not None and neighbor_ind.shape == x_test.shape[:-1] + (num_neighbor,):
	# 		pass
	# 	elif nn is not None and nn.get_params()["n_neighbors"] == num_neighbor:
	# 		neighbor_ind = torch.from_numpy(nn.kneighbors(x_test.detach().numpy().reshape(-1, DIM), return_distance=False)).reshape(x_test.shape[:-1] + (num_neighbor,))
	# 	else:
	# 		nn = sklearn.neighbors.NearestNeighbors(n_neighbors=num_neighbor, n_jobs=-1).fit(x_all.detach().numpy().reshape(-1, DIM))
	# 		neighbor_ind = torch.from_numpy(nn.kneighbors(x_test.detach().numpy().reshape(-1, DIM), return_distance=False)).reshape(x_test.shape[:-1] + (num_neighbor,))
	# 	neighbors: torch.Tensor = x_all.reshape(-1, DIM)[neighbor_ind]
	# 	need_local_gp: torch.Tensor = torch.all(torch.abs(x_test - torch.mean(neighbors, -2)) < dist_ratio * torch.std(neighbors, -2), -1).reshape(x_test.shape[:-1])
	# 	num_need_local_gp: int = int(torch.count_nonzero(need_local_gp).item())
	# 	print("{} inputs, where {} need local GP.".format(x_test.numel() // DIM, num_need_local_gp))
	# 	if num_need_local_gp > 0:
	# 		neighbor_used: torch.Tensor = neighbors[need_local_gp]
	# 		result[need_local_gp] = (get_covariance_matrix(model, x_test[need_local_gp, None], neighbor_used) @ square_solver(get_covariance_matrix(model, neighbor_used), y_all[neighbor_ind[need_local_gp], None])).reshape(-1)
	# 	return result

	# def pinv_local_pred_trial_v2_1() -> None:
	# 	print("gpytorch GPR using RBF kernel with pseudo inverse and local GP, pseudo inverse for parameters only")
	# 	print("Check how far local GP should go\n")
	# 	num_neighbor_trial: int = 32
	# 	nn: sklearn.neighbors.NearestNeighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=num_neighbor_trial, n_jobs=-1).fit(x_all)
	# 	for factor in ["0.{}".format(i) for i in range(1, 10)] + [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
	# 		print("Cutoff = {}".format(factor), flush=True)
	# 		plot(gpytorch_gpr(pinv_model, predictor=pinv_local_pred_v2_1, x_all=x_all_t, y_all=y_all_t, dist_ratio=float(factor), num_neighbor=num_neighbor_trial, nn=nn), "pinv_local_v2.1_{}".format(factor))

	# pinv_local_pred_trial_v2_1()
	# print(datetime.datetime.now(), flush=True)
	# gc.collect()

	# def pinv_local_pred_v2_2(
	# 	model: gpytorch.models.ExactGP,
	# 	x_test: torch.Tensor,
	# 	x_all: torch.Tensor,
	# 	y_all: torch.Tensor,
	# 	importance_cutoff_factor: float,
	# 	num_neighbor: int,
	# 	nn: sklearn.neighbors.NearestNeighbors | None = None,
	# 	neighbor_ind: torch.Tensor | None = None,
	# 	**kwagrs
	# ) -> torch.Tensor:
	# 	result: torch.Tensor = pinv_pred(model, x_test, x_all, y_all).clone()
	# 	need_local_gp: torch.Tensor = torch.abs(result) >= importance_cutoff_factor * torch.max(torch.abs(result))
	# 	num_need_local_gp: int = int(torch.count_nonzero(need_local_gp).item())
	# 	print("{} inputs, where {} need local GP.".format(x_test.numel() // DIM, num_need_local_gp))
	# 	if num_need_local_gp > 0:
	# 		if neighbor_ind is not None and neighbor_ind.shape == x_test.shape[:-1] + (num_neighbor,):
	# 			neighbor_ind_used: torch.Tensor = neighbor_ind[need_local_gp]
	# 		elif nn is not None and nn.get_params()["n_neighbors"] == num_neighbor:
	# 			neighbor_ind_used = torch.from_numpy(nn.kneighbors(x_test[need_local_gp].detach().numpy(), return_distance=False))
	# 		else:
	# 			nn = sklearn.neighbors.NearestNeighbors(n_neighbors=num_neighbor, n_jobs=-1).fit(x_all.detach().numpy())
	# 			neighbor_ind_used = torch.from_numpy(nn.kneighbors(x_test[need_local_gp].detach().numpy(), return_distance=False))
	# 		result[need_local_gp] = (get_covariance_matrix(model, x_test[need_local_gp, None], x_all[neighbor_ind_used]) @ square_solver(get_covariance_matrix(model, x_all[neighbor_ind_used]), y_all[neighbor_ind_used, None])).reshape(-1)
	# 	return result

	# def pinv_local_pred_trial_v2_2() -> None:
	# 	print("gpytorch GPR using RBF kernel with pseudo inverse and local GP, pseudo inverse for parameters only")
	# 	print("Check how far local GP should go\n")
	# 	num_neighbor_trial: int = 32
	# 	nn: sklearn.neighbors.NearestNeighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=num_neighbor_trial, n_jobs=-1).fit(x_all)
	# 	for factor in ["1e-{}".format(i) for i in range(17)] + [0.0]:
	# 		print("Cutoff = {}".format(factor), flush=True)
	# 		plot(gpytorch_gpr(pinv_model, predictor=pinv_local_pred_v2_2, x_all=x_all_t, y_all=y_all_t, importance_cutoff_factor=float(factor), num_neighbor=num_neighbor_trial, nn=nn), "pinv_local_v2.2_{}".format(factor))

	# pinv_local_pred_trial_v2_2()
	# print(datetime.datetime.now(), flush=True)
	# gc.collect()

	def pinv_local_pred_v3(
		model: gpytorch.models.ExactGP,
		x_test: torch.Tensor,
		x_all: torch.Tensor,
		y_all: torch.Tensor,
		num_neighbor: int,
		nn: sklearn.neighbors.NearestNeighbors | None = None,
		neighbor_ind: torch.Tensor | None = None,
		**kwagrs
	) -> torch.Tensor:
		if neighbor_ind is not None and neighbor_ind.shape == x_test.shape[:-1] + (num_neighbor,):
			pass
		elif nn is not None and nn.get_params()["n_neighbors"] == num_neighbor:
			neighbor_ind = torch.from_numpy(nn.kneighbors(x_test.detach().numpy().reshape(-1, DIM), return_distance=False)).reshape(x_test.shape[:-1] + (num_neighbor,)).detach()
		else:
			nn = sklearn.neighbors.NearestNeighbors(n_neighbors=num_neighbor, n_jobs=-1).fit(x_all.detach().numpy().reshape(-1, DIM))
			neighbor_ind = torch.from_numpy(nn.kneighbors(x_test.detach().numpy().reshape(-1, DIM), return_distance=False)).reshape(x_test.shape[:-1] + (num_neighbor,)).detach()
		neighbors: torch.Tensor = x_all.reshape(-1, DIM)[neighbor_ind]
		return (get_covariance_matrix(model, x_test[..., None, :], neighbors) @ square_solver(get_covariance_matrix(model, neighbors), y_all[neighbor_ind, None])).reshape(x_test.shape[:-1])

	def pinv_local_pred_trial_v3() -> tuple[int, sklearn.neighbors.NearestNeighbors, torch.Tensor]:
		print("gpytorch GPR using RBF kernel with pseudo inverse and local GP, pseudo inverse for parameters only")
		print("Check how large scale the local GP should have\n")
		smallest_error: float = math.inf
		best_nn: int = -1
		its_nn: sklearn.neighbors.NearestNeighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=32)
		its_central_nn: torch.Tensor = torch.Tensor()
		for n_neighbor in [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]:
			nn = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbor, n_jobs=-1).fit(extra_x)
			central_neighbor_ind = nn.kneighbors(extra_x, n_neighbor + 1, False) # N_MEMBER * N_NEIGHBOR
			assert isinstance(central_neighbor_ind, np.ndarray)
			central_neighbor_ind = central_neighbor_ind[central_neighbor_ind != np.arange(extra_x.shape[0]).reshape(-1, 1)].reshape(extra_x.shape[0], n_neighbor)
			try:
				model = gpytorch_train(
					x_t,
					y_t,
					gpytorch.kernels.RBFKernel(DIM),
					approx_se,
					initial_values=[torch.from_numpy(WIDTH)],
					pred=pinv_local_pred_v3,
					x_all=extra_x_t,
					y_all=extra_y_t,
					num_neighbor=n_neighbor,
					nn=nn,
					neighbor_ind=torch.from_numpy(central_neighbor_ind).detach()
				)
				if model is None:
					model = pinv_model
				err: float = approx_se(model, pinv_local_pred_v3, x_all=extra_x_t, y_all=extra_y_t, num_neighbor=n_neighbor, nn=nn, neighbor_ind=torch.from_numpy(central_neighbor_ind).detach()).item()
				print("#Neighbor = {}, err = {}".format(n_neighbor, err), flush=True)
				if err < smallest_error:
					best_nn = n_neighbor
					its_nn = nn
					its_central_nn = torch.from_numpy(central_neighbor_ind)
				plot(gpytorch_gpr(model, predictor=pinv_local_pred_v3, x_all=extra_x_t, y_all=extra_y_t, num_neighbor=n_neighbor, nn=nn), "pinv_local_v3_{}".format(n_neighbor))
			except Exception as e:
				traceback.print_exception(e)
		return best_nn, its_nn, its_central_nn

	NUM_NEIGHBOR, nn, central_neighbor_ind = pinv_local_pred_trial_v3()
	print(datetime.datetime.now(), flush=True)
	gc.collect()

	local_model: MyGPR = plot_lengthscale_error("local", predictor=pinv_local_pred_v3, x_all=extra_x_t, y_all=extra_y_t, num_neighbor=NUM_NEIGHBOR, nn=nn, neighbor_ind=central_neighbor_ind)
	print(datetime.datetime.now(), flush=True)
	gc.collect()

	# def pinv_local_pinv_pred(
	# 	model: gpytorch.models.ExactGP,
	# 	x_test: torch.Tensor,
	# 	x_all: torch.Tensor,
	# 	y_all: torch.Tensor,
	# 	nn: sklearn.neighbors.NearestNeighbors | None = None,
	# 	neighbor_ind: torch.Tensor | None = None,
	# 	extra_neighbor: int = NUM_NEIGHBOR,
	# 	extra_nn: sklearn.neighbors.NearestNeighbors | None = None,
	# 	extra_neighbor_ind: torch.Tensor | None = None,
	# 	**kwagrs
	# ) -> torch.Tensor:
	# 	if neighbor_ind is not None and neighbor_ind.shape == x_test.shape[:-1] + (NUM_NEIGHBOR,):
	# 		pass
	# 	elif nn is not None and nn.get_params()["n_neighbors"] == NUM_NEIGHBOR:
	# 		neighbor_ind = torch.from_numpy(nn.kneighbors(x_test.detach().numpy().reshape(-1, DIM), return_distance=False)).reshape(x_test.shape[:-1] + (NUM_NEIGHBOR,))
	# 	else:
	# 		nn = sklearn.neighbors.NearestNeighbors(n_neighbors=NUM_NEIGHBOR, n_jobs=-1).fit(x_all.detach().numpy().reshape(-1, DIM))
	# 		neighbor_ind = torch.from_numpy(nn.kneighbors(x_test.detach().numpy().reshape(-1, DIM), return_distance=False)).reshape(x_test.shape[:-1] + (NUM_NEIGHBOR,))
	# 	neighbors: torch.Tensor = x_all.reshape(-1, DIM)[neighbor_ind]
	# 	if extra_neighbor <= NUM_NEIGHBOR: # less than the given neighbor, using square inverse
	# 		return (get_covariance_matrix(model, x_test[..., None, :], neighbors) @ square_solver(get_covariance_matrix(model, neighbors), y_all[neighbor_ind, None])[0]).reshape(x_test.shape[:-1])
	# 	elif extra_neighbor == y_all.numel(): # all points, special case, remove nn
	# 		return (get_covariance_matrix(model, x_test[..., None, :], neighbors) @ preconditioned_lstsq_solve(get_covariance_matrix(model, x_all.reshape((x_test.ndim - 1) * (1,) + x_all.shape), neighbors), y_all.reshape((x_test.ndim - 1) * (1,) + y_all.shape + (1,)))).reshape(x_test.shape[:-1])
	# 	else: # extra nn needed
	# 		if extra_neighbor_ind is not None and extra_neighbor_ind.shape == x_test.shape[:-1] + (extra_neighbor,):
	# 			pass
	# 		elif extra_nn is not None and extra_nn.get_params()["n_neighbors"] == extra_neighbor:
	# 			extra_neighbor_ind = torch.from_numpy(extra_nn.kneighbors(x_test.detach().numpy(), return_distance=False)).reshape(x_test.shape[:-1] + (extra_neighbor,))
	# 		else:
	# 			extra_nn = sklearn.neighbors.NearestNeighbors(n_neighbors=extra_neighbor, n_jobs=-1).fit(x_all.detach().numpy())
	# 			extra_neighbor_ind = torch.from_numpy(extra_nn.kneighbors(x_test.detach().numpy(), return_distance=False)).reshape(x_test.shape[:-1] + (extra_neighbor,))
	# 		return (get_covariance_matrix(model, x_test[..., None, :], neighbors) @ preconditioned_lstsq_solve(get_covariance_matrix(model, x_all.reshape(-1, DIM)[extra_neighbor_ind], neighbors), y_all[extra_neighbor_ind, None])).reshape(x_test.shape[:-1])

	# def pinv_local_pinv_pred_trial() -> None:
	# 	print("gpytorch GPR using RBF kernel with pseudo inverse and local GP, pseudo inverse for parameters only")
	# 	print("Check if local GP should use approximate method\n")
	# 	for extra_neighbor in [NUM_NEIGHBOR, 2 * NUM_NEIGHBOR, 4 * NUM_NEIGHBOR, extra_y.size]:
	# 		print("# Extra Neighbor = {}".format(extra_neighbor))
	# 		try:
	# 			plot(gpytorch_gpr(local_model, predictor=pinv_local_pinv_pred, x_all=extra_x_t, y_all=extra_y_t, nn=nn, extra_neighbor=extra_neighbor), "pinv_local_pinv_{}".format(extra_neighbor))
	# 		except Exception as e:
	# 			traceback.print_exception(e)

	# pinv_local_pinv_pred_trial()
	# print(datetime.datetime.now(), flush=True)
	# gc.collect()

	# def pinv_local_se(
	# 	model: gpytorch.models.ExactGP,
	# 	x_all: torch.Tensor,
	# 	y_all: torch.Tensor,
	# 	global_predictor: ApproxPredType | None,
	# 	neighbor_ind: torch.Tensor | None = None,
	# 	**kwargs
	# ) -> torch.Tensor:
	# 	result_global: torch.Tensor = global_predictor(model, x_all, x_all, y_all, **kwargs) if global_predictor is not None else default_predict(model, x_all)
	# 	result_local: torch.Tensor = pinv_local_pred_v3(model, x_all, x_all, y_all, NUM_NEIGHBOR, neighbor_ind=neighbor_ind, **kwargs)
	# 	return torch.minimum(torch.abs(y_all - result_global), torch.abs(y_all - result_local)).square().sum()

	# def pinv_local_pred_v4(
	# 	model: gpytorch.models.ExactGP,
	# 	x_test: torch.Tensor,
	# 	x_all: torch.Tensor,
	# 	y_all: torch.Tensor,
	# 	global_predictor: ApproxPredType | None = None,
	# 	nn: sklearn.neighbors.NearestNeighbors | None = None,
	# 	neighbor_ind: torch.Tensor | None = None,
	# 	use_local: torch.Tensor | None = None,
	# 	**kwargs
	# ) -> torch.Tensor:
	# 	if use_local is None:
	# 		return pinv_local_pred_v3(model, x_test, x_all, y_all, NUM_NEIGHBOR, nn, neighbor_ind, **kwargs)
	# 	else:
	# 		# same as v3
	# 		if neighbor_ind is not None and neighbor_ind.shape == x_test.shape[:-1] + (NUM_NEIGHBOR,):
	# 			pass
	# 		elif nn is not None and nn.get_params()["n_neighbors"] == NUM_NEIGHBOR:
	# 			neighbor_ind = torch.from_numpy(nn.kneighbors(x_test.detach().numpy().reshape(-1, DIM), return_distance=False)).reshape(x_test.shape[:-1] + (NUM_NEIGHBOR,))
	# 		else:
	# 			nn = sklearn.neighbors.NearestNeighbors(n_neighbors=NUM_NEIGHBOR, n_jobs=-1).fit(x_all.detach().numpy().reshape(-1, DIM))
	# 			neighbor_ind = torch.from_numpy(nn.kneighbors(x_test.detach().numpy().reshape(-1, DIM), return_distance=False)).reshape(x_test.shape[:-1] + (NUM_NEIGHBOR,))
	# 		neighbors: torch.Tensor = x_all.reshape(-1, DIM)[neighbor_ind]
	# 		result_local: torch.Tensor = (get_covariance_matrix(model, x_test[..., None, :], neighbors) @ square_solver(get_covariance_matrix(model, neighbors), y_all[neighbor_ind, None])).reshape(x_test.shape[:-1])
	# 		result_global: torch.Tensor = global_predictor(model, x_test, x_all, y_all, **kwargs) if global_predictor is not None else default_predict(model, x_test)
	# 		# predict based on votes of the locals
	# 		use_local_weight: torch.Tensor = use_local[neighbor_ind].to(torch.double).sum(dim=-1)
	# 		return use_local_weight / NUM_NEIGHBOR * result_local + (1.0 - use_local_weight / NUM_NEIGHBOR) * result_global

	# def pinv_local_se_trial() -> None:
	# 	print("gpytorch GPR using RBF kernel with pseudo inverse and local GP, both for parameters")
	# 	print("Use local GP to train parameter as well\n")
	# 	for pred, name in zip([None, pinv_pred], ["loocv", "pinv"]):
	# 		if (model := gpytorch_train(x_t, y_t, gpytorch.kernels.RBFKernel(DIM), pinv_local_se, initial_values=[p.data for p in local_model.parameters()], initial_value_search=False, x_all=x_all_t, y_all=y_all_t, global_predictor=pred, neighbor_ind=torch.from_numpy(central_neighbor_ind))):
	# 			result_global: torch.Tensor = pred(model, x_all_t, x_all_t, y_all_t) if pred is not None else default_predict(model, x_all_t)
	# 			result_local: torch.Tensor = pinv_local_pred_v4(model, x_all_t, x_all_t, y_all_t, neighbor_ind=torch.from_numpy(central_neighbor_ind))
	# 			use_local: torch.Tensor = torch.abs(y_all_t - result_local) < torch.abs(y_all_t - result_global)
	# 			plot(gpytorch_gpr(model, predictor=pinv_local_pred_v4, x_all=extra_x_t, y_all=extra_y_t, nn=nn, use_local=use_local), "local_weighted_{}".format(name))

	# pinv_local_se_trial()
	# print(datetime.datetime.now(), flush=True)
	# gc.collect()

if __name__ == "__main__":
	main()