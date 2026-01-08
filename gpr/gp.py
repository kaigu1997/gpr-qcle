r"""gp
==
Implementation for gaussian process (gp) regression.
"""
import collections.abc
import copy
import math
import typing

import gpytorch
import gpytorch.constraints
import linear_operator
import numpy as np
import torch

import constant
import pes
import plot
import point

torch.set_default_dtype(constant.DTYPE)
torch.set_default_device(constant.DEVICE)
torch.manual_seed(point.SEED)
DEBUG_MODE: bool = False


@typing.final
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

	Methods
	----------
	cov()
		The kernel function
	forward(x)
		The implementation of GPR
	"""
	__slots__: typing.Final[tuple] = ("__mean", "__cov")
	__mean: typing.Final[gpytorch.means.Mean]
	__cov: typing.Final[gpytorch.kernels.Kernel]

	def __init__(
		self,
		x: torch.Tensor,
		y: torch.Tensor,
		likelihood: gpytorch.likelihoods.GaussianLikelihood | gpytorch.likelihoods.FixedNoiseGaussianLikelihood,
		kernel: gpytorch.kernels.Kernel
	):
		super().__init__(x, y, likelihood)
		self.__mean = gpytorch.means.ZeroMean()
		self.__cov = kernel

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


@typing.final
class _NoConstraint(gpytorch.constraints.Interval):
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


@typing.final
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
	MAX_ITER: typing.Final = 15000
	FTOL: typing.Final = 2.2204460492503131e-09
	GTOL: typing.Final = 1e-5
	NOISE: typing.Final = float(gpytorch.settings.min_fixed_noise.value(torch.double) or 1e-8)
	__slots__: typing.Final[tuple] = ("__kernel", "__x_all", "__y_all", "__scale", "__model", "__model_param", "__k_inv_y", "__weights_updated", "__optimizer")
	__kernel: typing.Final[gpytorch.kernels.Kernel]
	__x_all: torch.Tensor
	__y_all: torch.Tensor
	__scale: float
	__model: typing.Final[GP]
	__model_param: dict[str, torch.Tensor]
	__k_inv_y: torch.Tensor
	__weights_updated: bool
	__optimizer: typing.Final[torch.optim.Optimizer]

	def __init__(self, kernel: gpytorch.kernels.Kernel, phasedim: int):
		self.__kernel = copy.deepcopy(kernel)
		likelihood: typing.Final = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.full((2,), SinglePredictor.NOISE))
		self.__x_all = torch.Tensor()
		self.__y_all = torch.Tensor()
		self.__scale: float = 1.0
		self.__model = GP(torch.zeros((2, phasedim)), torch.zeros((2,)), likelihood, self.__kernel)
		self.__model_param: dict[str, torch.Tensor] = copy.deepcopy(self.__model.state_dict())
		self.__k_inv_y = torch.Tensor()
		self.__weights_updated: bool = False
		self.__optimizer = torch.optim.Rprop(self.__model.parameters(), lr=1.0)

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
			self.__k_inv_y = (linear_operator.utils.stable_pinverse(self.__model.cov(self.__x_all, self.get_training_features()).to_dense()) @ self.__y_all).to_dense().detach()
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
		return (self.__model.cov(x_test, self.get_training_features()) @ self.k_inv_y).to_dense().detach()

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
		self.__x_all = x_all.reshape(-1, x_all.shape[-1]).clone().detach()
		self.__y_all = y_all.reshape(-1).clone().detach()
		self.__scale = scale
		self.__model.set_train_data(self.__x_all[:num_points].detach(), self.__y_all[:num_points].detach(), False)
		self.__weights_updated = False

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
			for param_name, param, constraint in model.named_parameters_and_constraints():
				if print_grad and param.grad is not None:
					print(f"Parameter name: {param_name:42} value = {plot.format_array(param)} grad = {plot.format_array(param.grad)}")
				else:
					print(f"Parameter name: {"".join(param_name.split("raw_")):42} value = {plot.format_array(constraint.transform(param) if isinstance(constraint, gpytorch.constraints.Interval) else param)}")

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
		
		def set_lr(optimizer: torch.optim.Optimizer, new_lr: float) -> None:
			r"""To set the learning rate of the optimizer

			Parameters
			----------
			optimizer : torch.optim.Optimizer
				The optimizer, which contains learning rate
			new_lr : float
				The learning rate to set to the optimizer
			"""
			for param in optimizer.param_groups:
				param["lr"] = new_lr

		def print_stuff(
			loss: torch.Tensor,
			optimizer: torch.optim.Optimizer,
			model: gpytorch.models.ExactGP,
			print_grad: bool = False,
			extra_str: str = "\t"
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
			print(f"{extra_str}loss = {loss.item():.15e}, lr = {get_lr(optimizer)}")
			print_model(model, print_grad)

		self.__weights_updated = False
		assert isinstance(self.__model.train_targets, torch.Tensor)
		# train model
		self.__model.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.full((self.get_training_features().shape[0],), SinglePredictor.NOISE))
		self.__model.train()
		self.__model.likelihood.train()
		self.__model.load_state_dict(self.__model_param)
		if print_log:
			print_model(self.__model)
		finish_early: bool = False
		self.__optimizer.zero_grad()
		loss: torch.Tensor = self.error(False)
		loss.backward()
		last_value: float = loss.item()
		print_stuff(loss, self.__optimizer, self.__model, True, "Init")
		for i in range(1, SinglePredictor.MAX_ITER + 1):
			if math.sqrt(sum(torch.sum(param.grad ** 2).item() if param.grad is not None else 0.0 for param in self.__model.parameters())) < SinglePredictor.GTOL:
				finish_early = True
				print("Convergence: |Gradient| <= GTOL")
				i -= 1
				break
			# adjust lr
			old_prm: dict[str, torch.Tensor] = copy.deepcopy(self.__model.state_dict())
			self.__optimizer.step()
			loss = self.error(False)
			if print_log:
				print_stuff(loss, self.__optimizer, self.__model, True)
			if loss < last_value:
				set_lr(self.__optimizer, get_lr(self.__optimizer) * 2.0)
				if print_log:
					print("loss < last_value")
					print_stuff(loss, self.__optimizer, self.__model, True)
			else:
				if print_log:
					print("loss > last_value or loss is NaN")
				while loss >= last_value or loss.isnan().item():
					last_loop_value: float = loss.item()
					self.__model.load_state_dict(old_prm)
					set_lr(self.__optimizer, get_lr(self.__optimizer) / 2.0)
					self.__optimizer.step()
					loss = self.error(False)
					if print_log:
						print_stuff(loss, self.__optimizer, self.__model, True)
					if last_loop_value == loss.item():
						print("No stepping forward")
						# no stepping forward, but still larger than last, meaning last is the best
						self.__model.load_state_dict(old_prm)
						loss = self.error(False)
						break
			if i % (SinglePredictor.MAX_ITER // 100) == 0:
				print(f"Iter {i} - Loss: {loss.item():.15e} - lr: {get_lr(self.__optimizer)}")
				print_model(self.__model, True)
				print_model(self.__model)
			# stopping criteria
			if (last_value - loss.item()) / max(abs(last_value), abs(loss.item()), 1.0) < SinglePredictor.FTOL:
				finish_early = True
				print("Convergence: |f_i - f_{i+1}| <= FTOL")
				break
			if math.sqrt(sum(torch.sum(param.grad ** 2).item() if param.grad is not None else 0.0 for param in self.__model.parameters())) < SinglePredictor.GTOL:
				finish_early = True
				print("Convergence: |Gradient| <= GTOL")
				break
			set_lr(self.__optimizer, get_lr(self.__optimizer))
			self.__optimizer.zero_grad()
			last_value = loss.item()
			loss.backward()
			if print_log:
				print_stuff(loss, self.__optimizer, self.__model, True, f"\tlast = {last_value}, ")
		if not finish_early:
			print("Stop: Total No. iterations reached limit.")
		print(f"Iter {i} - Loss: {loss.item():.15e} - lr: {get_lr(self.__optimizer)}")
		print_model(self.__model)
		print("", flush=True)
		self.__model.eval()
		self.__model.likelihood.eval()
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
		phasedim: typing.Final[int] = self.__x_all.shape[-1]
		marginal_kernel: typing.Final[gpytorch.kernels.RBFKernel] = gpytorch.kernels.RBFKernel(len(dimensions), lengthscale_constraint=_NoConstraint(self.__model.cov.lengthscale.reshape(1, phasedim)[:, dimensions]))
		prefactor: typing.Final[float] = math.sqrt((2.0 * torch.pi) ** (phasedim - len(dimensions))) * self.__model.cov.lengthscale[:, [i for i in range(phasedim) if i not in dimensions]].prod().item()
		return prefactor * marginal_kernel(x_test, self.get_training_features()[:, dimensions]).to_dense() @ self.k_inv_y


@typing.final
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

	__slots__: typing.Final[tuple] = ("__config", "__predictors",)
	__config: typing.Final[pes.ModelConfig]
	__predictors: typing.Final[tuple[SinglePredictor, ...]]

	def __init__(self, config: pes.ModelConfig, kernel_initial_value: torch.Tensor | None):
		self.__config = config
		self.__predictors = tuple(SinglePredictor(gpytorch.kernels.RBFKernel(config.PHASEDIM, lengthscale_constraint=_NoConstraint(kernel_initial_value)), config.PHASEDIM) for _ in config.ELEMENT_RANGE)

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

	def update(
		self,
		x_all: list[torch.Tensor],
		y_all: list[torch.Tensor],
		num_points: int | torch.Tensor,
		scale: torch.Tensor
	) -> None:
		r"""To update the training inputs and targets, as well as the rescale factor

		Parameters
		----------
		x_all : list[torch.Tensor], len of NUM_TRIG, each of shape (num_points * (1 + NUM_XTR_RATIO), PHASEDIM)
			All training inputs
		y_all : list[torch.Tensor], len of NUM_TRIG, each of shape (num_points * (1 + NUM_XTR_RATIO))
			All training targets
		num_points : int | torch.Tensor, shape of (NUM_TRIG,)
			The number of points located at the front of all points that is used as the subset
		scale : torch.Tensor, shape of (NUM_ELM,)
			The rescale factor
		"""
		if isinstance(num_points, int):
			num_points = torch.full((self.__config.NUM_TRIG,), num_points)
		for iElement, pred in enumerate(self.__predictors):
			RowIndex: int = iElement // self.__config.NUM_PES
			ColIndex: int = iElement % self.__config.NUM_PES
			TrilIndex: int = self.__config.FLATTEN_TRIL_INDEX[iElement]
			pred.update(
				x_all[TrilIndex],
				y_all[TrilIndex].real if RowIndex <= ColIndex else y_all[TrilIndex].imag,
				scale[iElement].item(),
				int(num_points[TrilIndex].item())
			)

	def train(self, print_log: bool = DEBUG_MODE) -> None:
		r"""To train each predictor

		Parameters
		----------
		print_log : bool, optional
			Whether to print the log to console, by default `DEBUG_MODE`
		"""
		for iElement in self.__config.ELEMENT_RANGE:
			if __class__.__check_predictor(self.__predictors[iElement]):
				print("Training " + plot.get_RI_label(iElement, self.__config.NUM_PES))
				self.__predictors[iElement].train(print_log)

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

	def predict(self, x_input: torch.Tensor, ElementIndex: int) -> torch.Tensor:
		r"""To predict test targets based on input and corresponding density matrix element

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

	def get_marginal(
		self,
		dimensions: int | collections.abc.Iterable[int],
		x_input: torch.Tensor,
		ElementIndex: int
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

		Returns
		-------
		torch.Tensor, shape of (N,)
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
		return self.__combine_to_complex(
			x_input,
			ElementIndex // self.__config.NUM_PES,
			ElementIndex % self.__config.NUM_PES,
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
