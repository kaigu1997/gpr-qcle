import collections.abc
import math
import os
import sys
import traceback

import gpytorch
import gpytorch.constraints
import numpy as np
import numpy.typing as npt
import torch

sys.path.append(os.path.dirname(__file__))

import utility


def square_solver(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
	"""
	Solve `Ax=b` when A is real-symmetric

	This is done by eigenvalue decomposition for numerical stability

	Parameters
	----------
	A : torch.Tensor, of shape (*, n, n)
		The lhs real-symmetric matrix
	B : torch.Tensor, of shape (n,) or (*, n, k)
		The rhs tensor

	Returns
	-------
	torch.Tensor
		Solution by EVD
	"""
	val: torch.Tensor
	vec: torch.Tensor
	val, vec = torch.linalg.eigh(A) # A = vec @ val.diag() @ vec.T
	return vec @ (1.0 / val).diag_embed() @ vec.mH @ B


def preconditioned_lstsq_solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
	"""
	Solve `min||Ax-b||` with SVD-based precondioner

	Parameters
	----------
	A : torch.Tensor, of shape (*, m, n)
		The lhs tensor where m>n 
	B : torch.Tensor, of shape (*, n, k)
		The rhs tensor

	Returns
	-------
	torch.Tensor
		Least square solution done by preconditioning
	"""
	s: torch.Tensor
	vh: torch.Tensor
	_, s, vh = torch.linalg.svd(A) # A: m-n, _: m-m, s: n, vh: n-n
	q: torch.Tensor = vh.mH @ (1.0 / s.abs()).diag_embed()
	return q @ torch.linalg.lstsq(A @ q, B, rcond=0.0, driver="gels").solution


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
	"""
	To print all stuffs needed

	Parameters
	----------
	model : gpytorch.models.ExactGP
		Gaussian process model, containing mean and covariances and their parameters
	loss : float
		Loss by error function
	learning_rate : float
		The learning rate
	indent : int, optional
		The number of "\\t" in front of each line, by default 0
	print_grad : bool, optional
		Whether to print gradient in the model or not, by default False
	hessian : torch.Tensor, optional
		Second order derivatives, only used when `print_grad=True`, by default None
	start_str : str, optional
		An extra string added at the front, by default ""
	end_str : str, optional
		An extra string added at the end, by default ""
	flush : bool, optional
		Whether to forcibly flush the stream, by default False
	"""
	def print_model(model_print_grad: bool = False) -> None:
		"""
		To print the parameters of the model

		Parameters
		----------
		model_print_grad : bool, optional
			Whether to print the gradient or not, by default False
		"""
		param_name_fmt_str: str = f"{"\t" * indent}Parameter name: {{}}"
		param_name: str
		param: torch.nn.Parameter
		constraint: gpytorch.constraints.Interval | None
		for param_name, param, constraint in model.named_parameters_and_constraints():
			if model_print_grad and param.grad is not None:
				print(
					param_name_fmt_str.format(param_name),
					utility.format_array("value", param),
					utility.format_array("grad", param.grad)
				)
			else:
				print(
					param_name_fmt_str.format("".join(param_name.split("raw_"))),
					utility.format_array("value", constraint.transform(param) if isinstance(constraint, gpytorch.constraints.Interval) else param)
				)

	print(f"{"\t" * indent}{start_str + " " if start_str != "" else ""}loss = {loss:.15e}, lr = {learning_rate}")
	print_model()
	if print_grad:
		print_model(True)
		if hessian is not None:
			print("\t" * indent, utility.format_array("hessian", hessian.reshape(-1)), sep="")
			print(f"{"\t" * indent}Cond(hessian): {torch.linalg.cond(hessian).item()}")
	print(("\t" * indent + end_str + "\n") if end_str != "" else "", end="", flush=flush)


LossFuncType = collections.abc.Callable[[gpytorch.kernels.Kernel], torch.Tensor]


class GradientOptimization:
	"""
	Do one step of gradient-based optimization

	Parameters
	----------
	model : gpytorch.models.ExactGP
		Model containing all parameters
	param_indices : list[int]
		The starting index of each parameters, staring from 0 and end at N+1
	loss_func : LossFuncType
		Loss function
	print_log : bool
		Whether to print the log to console
	cov_name : str, optional
		The name of the covariance part of GPR, by default "cov"
	ftol : float | None, optional
		Tolerance of function values, by default None
	xtol : float | None, optional
		Tolerance of gradient / parameter values, by default None

	Attributes
	----------
	__FTOL : float
		Absolute and relative tolerance of function in optimization
	__GTOL : float
		Tolerance of gradient norm in optimization
	"""
	__FTOL: float = 2.2204460492503131e-09
	__GTOL: float = 1e-5
	__slots__ = ("__model", "__param_indices", "__loss_func", "__print_log", "__cov", "__ftol", "__xtol")
	def __init__(
		self,
		model: gpytorch.models.ExactGP,
		param_indices: list[int],
		loss_func: LossFuncType,
		print_log: bool,
		cov_name: str = "cov",
		ftol: float | None = None,
		xtol: float | None = None
	):
		self.__model: gpytorch.models.ExactGP = model
		self.__param_indices: list[int] = param_indices
		self.__loss_func: LossFuncType = loss_func
		self.__print_log: bool = print_log
		assert hasattr(model, cov_name)
		self.__cov = getattr(model, cov_name)
		assert isinstance(self.__cov, gpytorch.kernels.Kernel)
		self.__ftol: float = ftol or __class__.__FTOL
		self.__xtol: float = xtol or __class__.__GTOL

	def __call__(self, learning_rate: float, loss: torch.Tensor) -> tuple[float, torch.Tensor, bool]:
		"""
		Do one step of gradient-based optimization

		Parameters
		----------
		learning_rate : float
			Starting learning rate
		loss : torch.Tensor
			Loss tensor, could backpropagate for gradients

		Returns
		-------
		tuple[float, torch.Tensor, float, bool]
			Learning rate, loss, and to stop iteration or not

			Learning rate of NAN means to turn to derivative-free optimization
		"""
		def optimize_with_learning_rate(
			combined_parameters: torch.Tensor,
			change: torch.Tensor,
			last_loss: float,
			method_name: str,
			initial_learning_rate: float = 1.0
		) -> tuple[torch.Tensor, float]:
			"""
			To change the parameter with adjustable learning rate

			Parameters
			----------
			combined_parameters : torch.Tensor, shape of (N,)
				Initial values of parameters
			change : torch.Tensor, shape of (N,)
				The changing quantity of the parameters, e.g., combined gradient of parameters
			last_loss : float
				Loss from last iteration, as a comparison
			method_name : str
				The name of the method, only used when there is no stepping forward
			initial_learning_rate : float, optional
				Trial learning rate at the beginning, by default 1.0

			Returns
			-------
			tuple[torch.Tensor, float]
				Final loss and learning rate
			"""
			def change_param(target: torch.Tensor) -> None:
				"""
				To change the parameter of the model to `target`

				Parameters
				----------
				target : torch.Tensor
					Combined values that the parameters should be
				"""
				with torch.no_grad():
					for iParam, param in enumerate(self.__model.parameters()):
						param[...] = target[self.__param_indices[iParam]:self.__param_indices[iParam + 1]].reshape_as(param).detach()

			def return_back() -> tuple[torch.Tensor, float]:
				"""
				To turn back to the status when the function is called

				Returns
				-------
				tuple[torch.Tensor, float]
					`last_loss` in `torch.Tensor` form, and `initial_learning_rate`
				"""
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
					loss = self.__loss_func(self.__cov)
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
					loss = self.__loss_func(self.__cov)
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
		hessian: torch.Tensor = torch.eye(self.__param_indices[-1])
		try:
			for iParam, param in enumerate(self.__model.parameters()):
				param.grad = torch.autograd.grad(loss, param, None, True, True, True, True, False, True)[0] # same shape of param
				if torch.any(torch.isnan(param.grad)).item() or torch.any(torch.isinf(param.grad)).item():
					return math.nan, loss, False
				for iGrad, grad_elm in enumerate(param.grad.reshape(-1)):
					for jParam, param_for_grad in enumerate(self.__model.parameters()):
						hessian[self.__param_indices[iParam] + iGrad, self.__param_indices[jParam]:self.__param_indices[jParam + 1]] = torch.autograd.grad(grad_elm, param_for_grad, None, True, False, True, True, False, True)[0].reshape(-1)
		except Exception as e:
			traceback.print_exception(e)
			return math.nan, loss, False
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
		newton_change: torch.Tensor = square_solver(hessian, grad_combined)
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
		for change, method in zip([newton_change, grad_combined], ["Newton Method", "Gradient Descent"]):
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
	"""
	Do one step of derivative-free optimization by using Nelder-Mead simplex algorithm

	Parameters
	----------
	model : gpytorch.models.ExactGP
		Model containing all parameters
	param_indices : list[int]
		The starting index of each parameters, staring from 0 and end at N+1
	loss_func : LossFuncType
		Loss function, may
	print_log : bool
		Whether to print the log to console
	cov_name : str, optional
		The name of the covariance part of GPR, by default "cov"
	ftol : float | None, optional
		Tolerance of function values, by default None
	xtol : float | None, optional
		Tolerance of gradient / parameter values, by default None

	Attributes
	----------
	__FTOL : float
		Absolute and relative tolerance of function in optimization
	__XTOL : float
		Absolute tolerance of parameters
	__RHO : float
	__CHI : float
	__PSI : float
	__SIGMA : float
	__NONZDELT : float
	__ZDELT : float
		Magic numbers in simplex algorithm

	Methods
	-------
	__assign_value_to_model(x)
		To assign the volunteer value to model parameters
	__func(x)
		To give the loss function result
	converged()
		To judge if simplex converges
	"""
	__FTOL = 1e-4
	__XTOL = 1e-4
	__RHO = 1.0
	__CHI = 2.0
	__PSI = 0.5
	__SIGMA = 0.5
	__NONZDELT = 0.05
	__ZDELT = 0.00025
	__slots__ = ("__model", "__loss_func", "__print_log", "__ftol", "__xtol", "__cov", "__N", "__sim", "__one2np1", "__fsim")

	def __assign_value_to_model(self, x: npt.NDArray[np.double]) -> None:
		"""
		To assign the volunteer value to model parameters

		Parameters
		----------
		x : npt.NDArray[np.double]
			Vector containing all parameter values
		"""
		idx: int = 0
		for param in self.__model.parameters():
			param.ravel()[...] = torch.from_numpy(x[idx:idx + param.numel()])
			idx += param.numel()

	def __func(self, x: npt.NDArray[np.double]) -> float:
		"""
		To give the loss function result

		Parameters
		----------
		x : npt.NDArray[np.double]
			Values of all parameters

		Returns
		-------
		float
			Corresponding loss function value
		"""
		self.__assign_value_to_model(x)
		return self.__loss_func(self.__cov).item()

	def __init__(
		self,
		model: gpytorch.models.ExactGP,
		loss_func: LossFuncType,
		print_log: bool,
		cov_name: str = "cov",
		ftol: float | None = None,
		xtol: float | None = None
	) -> None:
		self.__model: gpytorch.models.ExactGP = model
		self.__loss_func: LossFuncType = loss_func
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
			y: npt.NDArray[np.double] = np.copy(x0)
			if y[k] != 0:
				y[k] = (1 + __class__.__NONZDELT) * y[k]
			else:
				y[k] = __class__.__ZDELT
			self.__sim[k + 1] = y
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
		"""
		To judge if simplex is converged

		Returns
		-------
		bool
			Whether converged
		"""
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
		"""
		Do one step of derivative-free optimization by using Nelder-Mead simplex algorithm
		"""
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