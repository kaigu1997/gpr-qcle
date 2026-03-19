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
import numpy as np
import torch
import torch_kmeans

import constant
import evolve
import pes
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
		extra_start_str: str = "\t\t\t",
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
			An extra string added at the front, by default "\t"
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
		Optimizer.print_stuff(loss.item(), param, lr, grad, None, "\tInit ")
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
					print("\t\t\tloss < last_value")
			else:
				if print_log:
					print("\t\t\tloss > last_value or loss is NaN")
				while loss >= last_value or loss.isnan().item():
					last_loop_value: float = loss.item()
					lr /= 2.0
					loss = loss_func((param - lr * grad).detach())
					if print_log:
						Optimizer.print_stuff(loss.item(), param - lr * grad, lr, grad, None, "\t\t\t\t")
					if last_loop_value == loss.item():
						# no stepping forward, but still larger than last, meaning last is the best
						return Optimizer.Result(param, grad, None, lr, loss_func(param).item(), i, Optimizer.ResultMessage.STUCK)
				param = (param - lr * grad).detach().requires_grad_()
			if i % (Optimizer.MAX_ITER // 1000) == 0 or print_log:
				Optimizer.print_stuff(loss.item(), param, lr, grad, None, f"\tIter {i} - last = {last_value} - ", "\t\t")
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
		Optimizer.print_stuff(loss.item(), param, mu, grad, hessian, "\tInit ")
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
					print("\t\t\tloss < last_value")
			else:
				if print_log:
					print("\t\t\tloss > last_value or loss is NaN")
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
						Optimizer.print_stuff(loss.item(), param - update, mu, grad, hessian + mu * torch.eye(param.numel()), "\t\t\t\t")
					if last_loop_value == loss.item():
						# no stepping forward, but still larger than last, meaning last is the best
						return Optimizer.Result(param, grad, hessian, mu, loss_func(param).item(), i, Optimizer.ResultMessage.STUCK)
				# once loss is less than last, change the parameter
				param = (param - update).detach().requires_grad_()
				rho: float = (last_value - loss.item()) / (grad @ update - 0.5 * update @ hessian @ update + 1e-12).item()
				mu = mu * max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
				v = 2.0
			if i % (Optimizer.MAX_ITER // 1000) == 0 or print_log:
				Optimizer.print_stuff(loss.item(), param, mu, grad, hessian, f"\t\tIter {i} - last = {last_value} - ", "\t\t")
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


@typing.final
class SinglePredictor:
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
	MAX_ITER : typing.Literal[50000]
		Maximum iteration of optimization
	FTOL : float
		Absolute and relative tolerance of function in optimization
	GTOL : float
		Tolerance of gradient in optimization
	x_ind : torch.Tensor, of shape (N_IND, PHASEDIM)
		Coordinates of inducing points
	x_all : torch.Tensor, of shape (N_ALL, PHASEDIM)
		All training inputs
	y_all : torch.Tensor, of shape (N_ALL)
		All training targets

	Methods
	-------
	model()
		The exact GPR model
	x_all()
		All features for approximate method
	get_training_features()
		To get the training features of the core subset
	update_weights()
		To update the weights, :math:`K^{-1}y`
	get_weights()
		To get the weights, :math:`K^{-1}y`
	predict(x_test)
		To predict the average
	error()
		To get the error by comparing label with prediction
	train()
		To train the parameters
	"""
	__slots__: typing.Final[tuple] = ("x_all", "x_ind", "y_all", "scale", "__raw_to_real", "__raw_lengthscale", "__lr", "__k_inv_y", "__weights_updated")
	x_all: torch.Tensor
	x_ind: torch.Tensor
	y_all: torch.Tensor
	__raw_to_real: typing.Final[collections.abc.Callable[[torch.Tensor], torch.Tensor]]
	__raw_lengthscale: torch.Tensor
	scale: float
	__lr: float | None
	__k_inv_y: torch.Tensor
	__weights_updated: bool

	def __init__(
		self,
		x_ind: torch.Tensor,
		x_all: torch.Tensor,
		y_all: torch.Tensor,
		scale: float,
		lengthscale_initial_value: torch.Tensor
	) -> None:
		self.x_ind = x_ind.detach().clone()
		self.x_all = x_all.detach().clone()
		self.y_all = y_all.detach().clone()
		self.__raw_to_real = torch.nn.Softplus()
		self.__raw_lengthscale = torch.where(
			lengthscale_initial_value * self.__raw_to_real.beta > self.__raw_to_real.threshold,
			lengthscale_initial_value,
			lengthscale_initial_value + torch.log(-torch.expm1(-lengthscale_initial_value)) # num stable inv softplus
		).detach()
		self.scale = scale
		self.__lr = 1.0 if self.__raw_lengthscale.numel() >= 10 else None
		self.train()
		self.__k_inv_y = (linear_operator.utils.stable_pinverse(rbf(self.__raw_to_real(self.__raw_lengthscale), self.x_all, self.x_ind)) @ self.y_all).detach()
		self.__weights_updated = True

	@property
	def lengthscale(self) -> torch.Tensor:
		r"""To access the real lengthscale

		Returns
		-------
		torch.Tensor
			Lengthscale in kernel function
		"""
		return self.__raw_to_real(self.__raw_lengthscale)

	def __update_weights(self) -> None:
		r"""To update the weights, :math:`K^{-1}y`
		"""
		if not self.__weights_updated:
			self.__k_inv_y = (linear_operator.utils.stable_pinverse(rbf(self.__raw_to_real(self.__raw_lengthscale), self.x_all, self.x_ind)) @ self.y_all).detach()
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
		return (rbf(self.lengthscale, x_test, self.x_ind) @ self.k_inv_y).detach()

	def error(self) -> torch.Tensor:
		r"""Error function of subset of regressor (SR) / projected process (PP)

		This function gives the sum of squared error

		Returns
		-------
		torch.Tensor
			The sum of squared prediction error
		"""
		return torch.sum((self.y_all - self.predict(self.x_all)) ** 2) * (self.scale ** 2)

	def update(
		self,
		x_ind: torch.Tensor,
		x_all: torch.Tensor,
		y_all: torch.Tensor,
		scale: float,
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
		scale : float
			The scaling factor
		"""
		self.x_ind = x_ind.reshape(-1, x_ind.shape[-1]).detach().clone()
		self.x_all = x_all.reshape(-1, x_all.shape[-1]).detach().clone()
		self.y_all = y_all.reshape(-1).detach().clone()
		self.scale = scale
		self.__weights_updated = False

	def train(self, print_log: bool = constant.DEBUG_MODE) -> None:
		r"""To train the parameters

		Parameters
		----------
		print_log : bool, optional
			Whether to print the log to console, by default `constant.DEBUG_MODE`
		"""
		def loss_func(length: torch.Tensor) -> torch.Tensor:
			kmn: torch.Tensor = rbf(self.__raw_to_real(length), self.x_all, self.x_ind)
			return torch.sum((self.y_all - (kmn @ (linear_operator.utils.stable_pinverse(kmn) @ self.y_all)).to_dense()) ** 2) * (self.scale ** 2)

		self.__weights_updated = False
		# train model
		with torch.no_grad():
			self.__raw_lengthscale.requires_grad = True
		print(f"\tscale = {self.scale}\n\t", end="")
		Optimizer.print_model(self.lengthscale)
		if self.__lr is None:
			loss: float = math.inf
			param: torch.Tensor = self.__raw_lengthscale
			while True:
				result = NewtonMethod(param, loss_func, print_log)
				print(f"\tIter = {result.num_iter} - {result.message}")
				Optimizer.print_stuff(result.func_value, self.__raw_to_real(result.param), None, extra_start_str="\t")
				if result.message in (Optimizer.ResultMessage.FVAL, Optimizer.ResultMessage.GRAD) or result.func_value >= loss:
					break
				else:
					loss = result.func_value
					param = result.param
				result = GradientDescend(param, loss_func, print_log=print_log)
				print(f"\tIter = {result.num_iter} - {result.message}")
				Optimizer.print_stuff(result.func_value, self.__raw_to_real(result.param), result.lr, extra_start_str="\t")
				if result.message in (Optimizer.ResultMessage.FVAL, Optimizer.ResultMessage.GRAD) or result.func_value >= loss:
					break
				else:
					loss = result.func_value
					param = result.param
		else:
			result = GradientDescend(self.__raw_lengthscale, loss_func, self.__lr, print_log)
			print(f"\tIter = {result.num_iter} - {result.message}")
			Optimizer.print_stuff(result.func_value, self.__raw_to_real(result.param), result.lr, extra_start_str="\t")
		with torch.no_grad():
			self.__raw_lengthscale.requires_grad = False
			self.__raw_lengthscale = result.param.detach().clone()
			if self.__lr is not None:
				self.__lr = result.lr

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
		phasedim: typing.Final[int] = self.x_all.shape[-1]
		prefactor: typing.Final[float] = math.sqrt((2.0 * torch.pi) ** (phasedim - len(dimensions))) * self.lengthscale[[i for i in range(phasedim) if i not in dimensions]].prod().item()
		return prefactor * rbf(self.lengthscale[dimensions], x_test, self.x_ind[:, dimensions]).to_dense() @ linear_operator.utils.stable_pinverse(rbf(self.lengthscale[dimensions], self.x_all[:, dimensions], self.x_ind[:, dimensions])) @ self.y_all


@typing.final
class GPRPredictors:
	r"""Combination of single predictors

	Parameters
	----------
	config : pes.ModelConfig
		Configuration of the model
	init_dist : pes.InitialDistribution
		Initial distribution to generate points, density, and weights
	x_all : list[torch.Tensor], len of NUM_TRIG, each of shape (num_points * NUM_XTR_RATIO, PHASEDIM)
		All training inputs
	y_all : list[torch.Tensor], len of NUM_TRIG, each of shape (num_points * NUM_XTR_RATIO,)
		All training targets
	num_ind : int
		The number of inducing points
	kernel_initial_value : torch.Tensor
		The initial value of lengthscale for all predictors
	model : pes.Potential
		Quantities derived from potential
	mass : torch.Tensor, shape of (DIM,)
		Mass of classical degree of freedom
	dt : float
		Time interval

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
		return not torch.all(predictor.y_all == 0).item()

	drc: typing.Final = evolve.Direction.BACKWARD
	__JUDGE_INCLUDE_THRESHOLD: typing.Final = 0.1
	__JUDGE_REGISTER_THRESHOLD: typing.Final = 0.1
	__slots__: typing.Final[tuple] = ("__config", "__predictors", "__kmeans", "__x_inds", "__lengths", "__k_inv_y", "__num_dt", "__model", "__mass", "__dt")
	__config: typing.Final[pes.ModelConfig]
	__predictors: typing.Final[list[SinglePredictor]]
	__kmeans: typing.Final[torch_kmeans.KMeans]
	__x_inds: typing.Final[tuple[list[torch.Tensor], ...]]
	__lengths: typing.Final[tuple[list[torch.Tensor], ...]]
	__k_inv_y: typing.Final[tuple[list[torch.Tensor], ...]]
	__num_dt: typing.Final[tuple[list[int], ...]]
	__model: typing.Final[pes.Potential]
	__mass: typing.Final[torch.Tensor]
	__dt: typing.Final[float]

	def __init__(
		self,
		config: pes.ModelConfig,
		x_all: list[torch.Tensor],
		y_all: list[torch.Tensor],
		num_ind: int,
		kernel_initial_value: torch.Tensor,
		model: pes.Potential,
		mass: torch.Tensor,
		dt: float,
	):
		KMEANS_THRESHOLD: typing.Final = 0.1
		self.__config = config
		self.__predictors = []
		self.__kmeans = torch_kmeans.KMeans(init_method="k-means++", n_clusters=num_ind, seed=constant.SEED, verbose=constant.DEBUG_MODE)
		ind_pts: typing.Final[torch.Tensor] = torch.cat([self.__kmeans(x[torch.newaxis, y.abs() > KMEANS_THRESHOLD * y.abs().max()]).centers for x, y in zip(x_all, y_all)])
		self.__x_inds = tuple([] for _ in config.TRIG_RANGE) # same for real and imag, since they have the same inducing points
		self.__num_dt = tuple([] for _ in config.TRIG_RANGE) # same for real and imag
		for iElement in config.ELEMENT_RANGE:
			print("Initial Training " + plot.get_RI_label(iElement, config.NUM_PES))
			TrilIndex: int = self.__config.FLATTEN_TRIL_INDEX[iElement]
			y: torch.Tensor = y_all[TrilIndex].real if iElement // config.NUM_PES <= iElement % config.NUM_PES else y_all[TrilIndex].imag
			self.__predictors.append(SinglePredictor(
				ind_pts[TrilIndex],
				x_all[TrilIndex],
				y,
				1.0 / y.abs().max().item(),
				kernel_initial_value
			)) # this includes training of initial distribution
		self.__lengths = tuple([] for _ in config.ELEMENT_RANGE)
		self.__k_inv_y = tuple([] for _ in config.ELEMENT_RANGE)
		self.train(y_all, 0) # to save the initial fitting, and take the residue to re-fit
		self.__model = model
		self.__mass = mass
		self.__dt = dt

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
		assert 0 <= ElementIndex < self.__config.NUM_ELM
		return self.__predictors[ElementIndex]

	@property
	def inducing_points(self) -> torch.Tensor:
		r"""To get all inducing points

		Returns
		-------
		torch.Tensor, shape of (NUM_TRIG, num_ind, PHASEDIM)
			Inducing points of independent, lower triangular elements
		"""
		return torch.stack([self.__predictors[iTrig].x_ind for iTrig in self.__config.TRIL_ELEMENT_INDICES], 0)

	@property
	def scale(self) -> torch.Tensor:
		r"""The rescaling factor of each predictor

		Returns
		-------
		torch.Tensor, shape of (NUM_ELM,)
			The rescaling factor
		"""
		return torch.tensor([pred.scale for pred in self.__predictors])

	def __combine_to_complex[**P](
		self,
		x_input: torch.Tensor,
		RowIndex: int,
		ColIndex: int,
		call_single_predictor: collections.abc.Callable[typing.Concatenate[SinglePredictor, torch.Tensor, P], collections.abc.Iterable[torch.Tensor]],
		*args: P.args,
		**kwargs: P.kwargs
	) -> tuple[torch.Tensor, ...]:
		r"""To combine results from single predictor into complex arrays

		Parameters
		----------
		x_input : torch.Tensor, shape of (..., PHASEDIM)
			Test inputs
		RowIndex : int
			Index of row of the element in density matrix
		ColIndex : int
			Index of column of the element in density matrix
		call_single_predictor : collections.abc.Callable[typing.Concatenate[SinglePredictor, torch.Tensor, P], collections.abc.Iterable[torch.Tensor]]
			The function that takes the single predictor and generates some Tensor (prediction, derivatives, marginals, etc)

		Returns
		-------
		tuple[torch.Tensor, ...]
			Combined complex arrays from single predictor
		"""
		assert 0 <= RowIndex < self.__config.NUM_PES and 0 <= ColIndex < self.__config.NUM_PES
		x_test: typing.Final[torch.Tensor] = x_input.reshape(-1, x_input.shape[-1])
		if RowIndex == ColIndex:
			return tuple(item.reshape(x_input.shape[:-1] + item.shape[1:]) + 0.j for item in call_single_predictor(self.__predictors[RowIndex * self.__config.NUM_PES + ColIndex], x_test, *args, **kwargs))
		elif RowIndex > ColIndex:
			return tuple((real + 1.j * imag).reshape(x_input.shape[:-1] + real.shape[1:]) for real, imag in zip(call_single_predictor(self.__predictors[ColIndex * self.__config.NUM_PES + RowIndex], x_test, *args, **kwargs), call_single_predictor(self.__predictors[RowIndex * self.__config.NUM_PES + ColIndex], x_test, *args, **kwargs)))
		else: # RowIndex < ColIndex
			return tuple((real - 1.j * imag).reshape(x_input.shape[:-1] + real.shape[1:]) for real, imag in zip(call_single_predictor(self.__predictors[RowIndex * self.__config.NUM_PES + ColIndex], x_test, *args, **kwargs), call_single_predictor(self.__predictors[ColIndex * self.__config.NUM_PES + RowIndex], x_test, *args, **kwargs)))

	def __residue_predict(self, x_input: torch.Tensor, ElementIndex: int) -> torch.Tensor:
		r"""To predict test targets residue based on input and corresponding density matrix element

		Parameters
		----------
		x_input : torch.Tensor, shape of (..., PHASEDIM)
			Test inputs
		ElementIndex : int
			Index of the element

		Returns
		-------
		torch.Tensor, shape of (...)
			Density of the element of all test inputs
		"""
		return self.__combine_to_complex(
			x_input,
			ElementIndex // self.__config.NUM_PES,
			ElementIndex % self.__config.NUM_PES,
			lambda pred, x_test: (pred.predict(x_test),) if GPRPredictors.__check_predictor(pred) else (torch.zeros(x_test.shape[0]),)
		)[0]

	def __non_residue_predict(
		self,
		x_input: torch.Tensor,
		ElementIndex: int,
		num_dt: int
	) -> torch.Tensor:
		r"""To predict main part by back propagation to each saved predictors

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
		TrilIndex: typing.Final[int] = self.__config.FLATTEN_TRIL_INDEX[ElementIndex]
		result: torch.Tensor = torch.zeros(x_input.shape[:-1], dtype=torch.cdouble)
		next_idx: int = len(self.__num_dt[TrilIndex]) - 1
		if next_idx >= 0: # only back-propagate when there are saved predictors
			RowIndex: typing.Final[int] = ElementIndex // self.__config.NUM_PES
			ColIndex: typing.Final[int] = ElementIndex % self.__config.NUM_PES
			RealElmIdx: typing.Final[int] = min(RowIndex, ColIndex) * self.__config.NUM_PES + max(RowIndex, ColIndex)
			ImagElmIdx: typing.Final[int] = max(RowIndex, ColIndex) * self.__config.NUM_PES + min(RowIndex, ColIndex)
			phase_factor: torch.Tensor = torch.ones_like(result)
			r0: torch.Tensor = x_input[..., :self.__config.DIM] # M * D
			p0: torch.Tensor = x_input[..., self.__config.DIM:] # M * D
			if self.__num_dt[TrilIndex][next_idx] == num_dt: # meet last one first
				# GP predict
				if RowIndex == ColIndex:
					result += rbf(self.__lengths[ElementIndex][next_idx], x_input, self.__x_inds[TrilIndex][next_idx]) @ self.__k_inv_y[ElementIndex][next_idx]
				else:
					result += torch.complex(rbf(self.__lengths[RealElmIdx][next_idx], x_input, self.__x_inds[TrilIndex][next_idx]) @ self.__k_inv_y[RealElmIdx][next_idx], rbf(self.__lengths[ImagElmIdx][next_idx], x_input, self.__x_inds[TrilIndex][next_idx]) @ self.__k_inv_y[ImagElmIdx][next_idx])
				next_idx -= 1
				if next_idx < 0: # no more saved predictor, break to save time
					return result
			for iTick in range(num_dt - 1, -1, -1): # n-1, n-2, ..., 0
				# evolve back
				r2, p1 = evolve.evolve_coordinates_adiabatically(self.__model, r0, p0, self.__mass, self.__dt / 2.0, GPRPredictors.drc, RowIndex, ColIndex)
				r4, p2 = evolve.evolve_coordinates_adiabatically(self.__model, r2, p1, self.__mass, self.__dt / 2.0, GPRPredictors.drc, RowIndex, ColIndex)
				if RowIndex != ColIndex:
					# accumulate phase factor
					evolve.evolve_density_adiabatically(self.__model, phase_factor, r4, r2, r0, evolve.Direction.FORWARD, self.__dt, RowIndex, ColIndex)
				r0 = r4
				p0 = p2
				if next_idx >= 0 and self.__num_dt[TrilIndex][next_idx] == iTick: # meet last one first
					# GP predict
					x0: torch.Tensor = torch.cat((r0, p0), -1)
					if RowIndex == ColIndex:
						result += rbf(self.__lengths[ElementIndex][next_idx], x0, self.__x_inds[TrilIndex][next_idx]) @ self.__k_inv_y[ElementIndex][next_idx]
					else:
						result += torch.complex(rbf(self.__lengths[RealElmIdx][next_idx], x0, self.__x_inds[TrilIndex][next_idx]) @ self.__k_inv_y[RealElmIdx][next_idx], rbf(self.__lengths[ImagElmIdx][next_idx], x0, self.__x_inds[TrilIndex][next_idx]) @ self.__k_inv_y[ImagElmIdx][next_idx]) * phase_factor
					next_idx -= 1
					if next_idx < 0: # no more saved predictor, break to save time
						return result
		return result

	def predict(
		self,
		x_input: torch.Tensor,
		ElementIndex: int,
		num_dt: int
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
		return self.__residue_predict(x_input, ElementIndex) + self.__non_residue_predict(x_input, ElementIndex, num_dt)

	def get_marginal(
		self,
		dimensions: int | collections.abc.Iterable[int],
		x_input: torch.Tensor,
		ElementIndex: int,
		num_dt: int
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
		def call_single_predictor(pred: SinglePredictor, x_test: torch.Tensor, dims: collections.abc.Sequence[int]) -> tuple[torch.Tensor]:
			r"""To do prediction of a single predictor

			Parameters
			----------
			pred : SinglePredictor
				The predictor
			x_test : torch.Tensor, shape of (N, len(dimensions))
				Test inputs
			dims : collections.abc.Sequence[int]
				The dims to be kept

			Returns
			-------
			torch.Tensor, shape of (N,)
				Test targets by the predictor
			"""
			if GPRPredictors.__check_predictor(pred):
				return (pred.get_marginal(dims, x_test),)
			else:
				return (torch.zeros(x_test.shape[0]),)

		if isinstance(dimensions, int):
			dimensions = [dimensions]
		else:
			dimensions = tuple(set(dimensions)) # remove duplicate
		assert all(0 <= dim <= self.__config.PHASEDIM for dim in dimensions)
		assert x_input.shape[-1] == len(dimensions)
		# RowIndex: typing.Final[int] = ElementIndex // self.__config.NUM_PES
		# ColIndex: typing.Final[int] = ElementIndex % self.__config.NUM_PES
		# TrilIndex: typing.Final[int] = self.__config.FLATTEN_TRIL_INDEX[ElementIndex]
		# result: torch.Tensor = torch.zeros(x_input.shape[:-1], dtype=torch.cdouble)
		# next_idx: int = len(self.__num_dt[TrilIndex]) - 1
		# r0: torch.Tensor = x_input[..., :self.__config.DIM] # M * D
		# p0: torch.Tensor = x_input[..., self.__config.DIM:] # M * D
		# for iTick in range(num_dt):
		# 	# evolve back
		# 	r2, p1 = evolve.evolve_coordinates_adiabatically(self.__model, r0, p0, self.__mass, self.__dt / 2.0, GPRPredictors.drc, RowIndex, ColIndex)
		# 	r0, p0 = evolve.evolve_coordinates_adiabatically(self.__model, r2, p1, self.__mass, self.__dt / 2.0, GPRPredictors.drc, RowIndex, ColIndex)
		# 	if next_idx >= 0 and self.__num_dt[TrilIndex][next_idx] == num_dt - 1 - iTick: # meet last one first
		# 		# GP predict
		# 		x0: torch.Tensor = torch.cat((r0, p0), -1)
		# 		if RowIndex == ColIndex:
		# 			result += rbf(self.__lengths[ElementIndex][next_idx][dimensions], x0[..., dimensions], self.__x_inds[TrilIndex][next_idx][..., dimensions]) @ self.__k_inv_y[ElementIndex][next_idx]
		# 		else:
		# 			RealElmIdx: int = min(RowIndex, ColIndex) * self.__config.NUM_PES + max(RowIndex, ColIndex)
		# 			ImagElmIdx: int = max(RowIndex, ColIndex) * self.__config.NUM_PES + min(RowIndex, ColIndex)
		# 			result += (rbf(self.__lengths[RealElmIdx][next_idx][dimensions], x0[..., dimensions], self.__x_inds[TrilIndex][next_idx][..., dimensions]) @ self.__k_inv_y[RealElmIdx][next_idx]) + 1.j * (rbf(self.__lengths[ImagElmIdx][next_idx][dimensions], x0[..., dimensions], self.__x_inds[TrilIndex][next_idx][..., dimensions]) @ self.__k_inv_y[ImagElmIdx][next_idx])
		# 		next_idx -= 1
		# # back to origin, using initial distribution
		# return result + self.__combine_to_complex(
		return self.__combine_to_complex(
			x_input,
			ElementIndex // self.__config.NUM_PES,
			ElementIndex % self.__config.NUM_PES,
			call_single_predictor,
			dimensions
		)[0]

	def update(
		self,
		x_ind: list[torch.Tensor],
		x_all: list[torch.Tensor],
		y_all: list[torch.Tensor],
		num_dt: int
	) -> None:
		r"""To update the training inputs and targets, as well as the rescale factor, and to judge whether to register the predictor with current num_dt

		Parameters
		----------
		x_ind : list[torch.Tensor], len of NUM_TRIG, each of shape (num_inducing, PHASEDIM)
			All inducing points
		x_all : list[torch.Tensor], len of NUM_TRIG, each of shape (num_points, PHASEDIM)
			All training inputs
		y_all : list[torch.Tensor], len of NUM_TRIG, each of shape (num_points,)
			All training targets
		num_dt : int
			The number of time steps since epoch
		"""
		# update the predictor
		for iPES, jPES, iElement, int_pt, x, y in zip(self.__config.TRIL_ROW_INDICES, self.__config.TRIL_COL_INDICES, self.__config.TRIL_ELEMENT_INDICES, x_ind, x_all, y_all):
			# predict sample points, and choose the inducing points
			y_res: torch.Tensor = y - self.__non_residue_predict(x, iElement, num_dt) # residue of all training points
			if iPES == jPES:
				self.__predictors[iElement].update(int_pt, x, y_res.real, 1.0 / y_res.real.abs().max().item())
			else: # if iPES != jPES
				SymElmIdx: int = jPES * self.__config.NUM_PES + iPES
				self.__predictors[iElement].update(int_pt, x, y_res.imag, 1.0 / y_res.imag.abs().max().item())
				self.__predictors[SymElmIdx].update(int_pt, x, y_res.real, 1.0 / y_res.real.abs().max().item())

	def train(
		self,
		y_all: list[torch.Tensor],
		num_dt: int,
		print_log: bool = constant.DEBUG_MODE
	) -> None:
		r"""To train each predictor, and to extract old predictor if necessary

		Parameters
		----------
		y_all : list[torch.Tensor], len of NUM_TRIG, each of shape (num_points,)
			All training targets
		num_dt : int
			The number of time steps since epoch
		print_log : bool, optional
			Whether to print the log to console, by default `constant.DEBUG_MODE`
		"""
		for iTrig, iPES, jPES, iElement, y in zip(self.__config.TRIG_RANGE, self.__config.TRIL_ROW_INDICES, self.__config.TRIL_COL_INDICES, self.__config.TRIL_ELEMENT_INDICES, y_all):
			SymElmIdx: int = jPES * self.__config.NUM_PES + iPES
			x_all: torch.Tensor = self.__predictors[iElement].x_all
			# two cases: if no predictors is saved, residue is exact - predict; if there is predictor, residue is y_all in predictors
			y_res: torch.Tensor # residue of all training points
			if len(self.__num_dt[iTrig]) == 0:
				y_res = y - self.__residue_predict(x_all, iElement) # use predict instead of y_all since y_all is exact
			else:
				y_res = (self.__predictors[iElement].y_all + 0.0j) if iPES == jPES else (self.__predictors[SymElmIdx].y_all + 1.j * self.__predictors[iElement].y_all)
			pts_include: torch.Tensor = y.abs() > y.abs().max() * GPRPredictors.__JUDGE_INCLUDE_THRESHOLD
			if torch.any(y_res[pts_include].abs() > GPRPredictors.__JUDGE_REGISTER_THRESHOLD * y[pts_include].abs()).item():
				y_res_res: torch.Tensor = y - y_res # the new residue to fit
				print(f"Register the residue predictor for {plot.get_element_label(iPES, jPES)} at {num_dt} dt; now maximum residue is {y_res_res.abs().max()} at {plot.format_array(x_all[y_res_res.abs().argmax()])}")
				# register predictor
				self.__x_inds[iTrig].append(self.__predictors[iElement].x_ind.detach().clone())
				self.__lengths[iElement].append(self.__predictors[iElement].lengthscale.detach().clone())
				self.__k_inv_y[iElement].append(self.__predictors[iElement].k_inv_y.detach().clone())
				self.__num_dt[iTrig].append(num_dt)
				# then choose the inducing points, and clear the predictor
				x_ind: torch.Tensor = self.__kmeans(x_all[torch.newaxis]).centers[0]
				if iPES == jPES:
					self.__predictors[iElement].update(x_ind, x_all, y_res_res.real, 1.0 / y_res_res.real.abs().max().item())
				else: # if iPES < jPES:
					self.__lengths[SymElmIdx].append(self.__predictors[SymElmIdx].lengthscale.detach().clone())
					self.__k_inv_y[SymElmIdx].append(self.__predictors[SymElmIdx].k_inv_y.detach().clone())
					self.__predictors[SymElmIdx].update(x_ind, x_all, y_res_res.real, 1.0 / y_res_res.real.abs().max().item())
					self.__predictors[iElement].update(x_ind, x_all, y_res_res.imag, 1.0 / y_res_res.imag.abs().max().item())
		for iElement, pred in zip(self.__config.ELEMENT_RANGE, self.__predictors):
			if __class__.__check_predictor(pred):
				print("Training " + plot.get_RI_label(iElement, self.__config.NUM_PES))
				pred.train(print_log)

	def print(self, f: typing.IO) -> None:
		r"""To print the parameters to file

		Parameters
		----------
		f : io.TextIOWrapper
			The file to save the parameters
		"""
		for predictor in self.__predictors:
			np.savetxt(f, predictor.lengthscale.detach().cpu().numpy().reshape(1, -1))
		print("\n", file=f)
