"""
gp
==
This module provides support for gaussian process (gp) regression.
"""
import copy
import math
import io
import os
import sys
import typing

import gpytorch
import gpytorch.constraints
import linear_operator
import numpy as np
import numpy.typing as npt
import torch

sys.path.append(os.path.dirname(__file__) + "/..")

import pes
import utility

torch.set_default_dtype(torch.double)
torch.manual_seed(0)
DEBUG_MODE: bool = True


class GP(gpytorch.models.ExactGP):
	"""
	Gaussian process regression

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
	forward(x)
		The implementation of GPR
	"""
	__slots__: tuple = ("mean", "cov")

	def __init__(self, x: torch.Tensor, y: torch.Tensor, likelihood: gpytorch.likelihoods.Likelihood, kernel: gpytorch.kernels.Kernel):
		super().__init__(x, y, likelihood)
		self.mean: gpytorch.means.Mean = gpytorch.means.ZeroMean()
		self.cov: gpytorch.kernels.Kernel = kernel

	def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
		"""
		The implementation of GPR

		Parameters
		----------
		x : torch.Tensor
			Training inputs

		Returns
		-------
		gpytorch.distributions.MultivariateNormal
			A gaussian process with certain mean and covariance
		"""
		Mean: torch.Tensor | torch.distributions.Distribution | linear_operator.LinearOperator = self.mean(x)
		assert isinstance(Mean, torch.Tensor)
		return gpytorch.distributions.MultivariateNormal(Mean, self.cov(x))


class SinglePredictor:
	"""
	The instantiation of gaussian process predictor

	Parameters
	----------
	kernel : gpytorch.kernels.Kernel
		The kernel to use

	Attributes
	----------
	__MAX_ITER : typing.Literal[15000]
		Maximum iteration of optimization
	__FTOL : float
		Absolute and relative tolerance of function in optimization
	__GTOL : float
		Tolerance of gradient in optimization
	__NOISE: float
		Extra noise term added for numerical stability in matrix inversion

	Methods
	-------
	__update_weights()
		To update the weights, :math:`K^{-1}y`
	get_training_features()
		To get the training features of the core subset
	get_weights()
		To get the weights, :math:`K^{-1}y`
	predict(x_test)
		To predict the average estimation
	error()
		To get the error by comparing label with prediction
	train()
		To train the parameters
	update(x_all, y_all, num_points)
		To update features and labels
	get_marginal(dimensions)
		To get the marginal distribution over given dimensions
	"""
	__MAX_ITER: typing.Literal[15000] = 15000
	__FTOL: float = 2.2204460492503131e-09
	__GTOL: float = 1e-5 ** 2
	__NOISE: float = 1e-4
	__slots__: tuple = ("__kernel", "__x_all", "__y_all", "__scale", "model", "__k_inv_y", "__weights_updated")

	def __init__(self, kernel: gpytorch.kernels.Kernel):
		self.__kernel: gpytorch.kernels.Kernel = copy.deepcopy(kernel)
		likelihood: gpytorch.likelihoods.FixedNoiseGaussianLikelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.full((2,), __class__.__NOISE))
		self.__x_all: torch.Tensor = torch.Tensor()
		self.__y_all: torch.Tensor = torch.Tensor()
		self.__scale: float = 1.0
		self.model: GP = GP(torch.zeros((2, pes.PHASEDIM)), torch.zeros((2,)), likelihood, self.__kernel)
		self.__k_inv_y: torch.Tensor = torch.Tensor()
		self.__weights_updated: bool = False

	def __update_weights(self) -> None:
		"""
		To update the weights, :math:`K^{-1}y`
		"""
		self.__k_inv_y = linear_operator.utils.stable_pinverse(self.model.cov(self.__x_all, self.get_training_features()).to_dense()) @ self.__y_all
		self.__weights_updated = True

	def get_training_features(self) -> torch.Tensor:
		"""
		To get the training features of the core subset

		Returns
		-------
		torch.Tensor
			The training features
		"""
		assert self.model.train_inputs is not None
		return self.model.train_inputs[0].detach()

	def get_weights(self) -> torch.Tensor:
		"""
		To get the weights, :math:`K^{-1}y`

		Returns
		-------
		torch.Tensor
			Weights, :math:`K^{-1}y`
		"""
		if not self.__weights_updated:
			self.__update_weights()
		return self.__k_inv_y

	def predict(self, x_test: torch.Tensor) -> torch.Tensor:
		"""
		To predict the average estimation

		Parameters
		----------
		x_test : torch.Tensor, shape of (N, PHASEDIM)
			Validation/Test inputs

		Returns
		-------
		torch.Tensor, shape of (N,)
			Corresponding validation/test targets based on noise-free SR/PP mean.

		Notes
		-----
		Instance of prediction of projected process (PP)
		"""
		return (self.model.cov(x_test, self.get_training_features()) @ self.get_weights()).to_dense()

	def error(self, use_weight: bool = True) -> torch.Tensor:
		"""
		Error function of projected process (PP)

		This function gives the sum of squared error

		Returns
		-------
		torch.Tensor
			The sum of squared prediction error
		"""
		if use_weight:
			return torch.sum((self.__y_all - self.predict(self.__x_all)) ** 2) * (self.__scale ** 2)
		else:
			kmn: torch.Tensor = self.model.cov(self.__x_all, self.get_training_features()).to_dense()
			return torch.sum((self.__y_all - kmn @ (linear_operator.utils.stable_pinverse(kmn) @ self.__y_all)) ** 2) * (self.__scale ** 2)

	def train(self) -> None:
		"""
		To train the parameters
		"""
		def print_stuff(
			loss: torch.Tensor,
			learning_rate: float,
			model: gpytorch.models.ExactGP,
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
			loss : torch.Tensor
				Loss by error function
			learning_rate : float
				The learning rate
			model : gpytorch.models.ExactGP
				Gaussian process model, containing mean and covariances and their parameters
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
				param_name_fmt_str: str = "{}Parameter name: {{}}".format("\t" * indent)
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

			print("{}{}loss = {:.15e}, lr = {}".format("\t" * indent, start_str + " " if start_str != "" else "", loss.item(), learning_rate))
			print_model()
			if print_grad:
				print_model(True)
				if hessian is not None:
					print("\t" * indent, utility.format_array("hessian", hessian.reshape(-1)), sep="")
					print("{}Cond(hessian): {}".format("\t" * indent, torch.linalg.cond(hessian).item()))
			print("\t" * indent + end_str + "\n" if end_str != "" else "", end="", flush=flush)

		def optimize_with_learning_rate(
			combined_parameters: torch.Tensor,
			change: torch.Tensor,
			param_indices: list[int],
			last_loss: float,
			method_name: str,
			initial_learning_rate: float = 1.0
		) -> tuple[torch.Tensor, float]:
			learning_rate: float = initial_learning_rate
			with torch.no_grad():
				for iParam, param in enumerate(self.model.parameters()):
					param[...] = (combined_parameters - learning_rate * change)[param_indices[iParam]:param_indices[iParam + 1]].reshape_as(param).detach()
			loss: torch.Tensor = self.error(False)
			while loss.item() >= last_loss:
				learning_rate /= 2.0
				with torch.no_grad():
					for iParam, param in enumerate(self.model.parameters()):
						param[...] = (combined_parameters - learning_rate * change)[param_indices[iParam]:param_indices[iParam + 1]].reshape_as(param).detach()
				loss = self.error(False)
				if DEBUG_MODE:
					print_stuff(
						loss,
						learning_rate,
						self.model,
						2,
						start_str="last = {:.15e}, ".format(last_value),
						flush=True
					)
				if all((param.reshape(-1) == param_combined[param_indices[iParam]:param_indices[iParam + 1]]).all().item() for iParam, param in enumerate(self.model.parameters())):
					# based on current learning rate, all parameters being the same
					print("No stepping Forward for {}.".format(method_name))
					break
			return loss, learning_rate

		self.__weights_updated = False
		assert isinstance(self.model.train_targets, torch.Tensor)
		# train model
		self.model.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.full((self.get_training_features().shape[0],), __class__.__NOISE))
		self.model.train()
		self.model.likelihood.train()
		param_indices: list[int] = [0]
		for param in self.model.parameters():
			param.requires_grad = True
			param_indices.append(param_indices[-1] + param.numel())
		finish_early: bool = False
		loss: torch.Tensor = self.error(False)
		last_value: float = torch.inf
		learning_rate: float = 1.0
		hessian: torch.Tensor = torch.zeros(param_indices[-1], param_indices[-1], dtype=torch.double)
		num_iter: int = 0
		for i in range(1, __class__.__MAX_ITER + 1):
			# calculate gradient and hessian
			for iParam, param in enumerate(self.model.parameters()):
				param.grad = torch.autograd.grad(loss, param, None, True, True, True, True, False, True)[0] # same shape of param
				for iGrad, grad_elm in enumerate(param.grad.reshape(-1)):
					for jParam, param_for_grad in enumerate(self.model.parameters()):
						hessian[param_indices[iParam] + iGrad, param_indices[jParam]:param_indices[jParam + 1]] = torch.autograd.grad(grad_elm, param_for_grad, None, True, False, True, True, False, True)[0]
			hessian = (hessian + hessian.T) / 2.0 # symmetrize
			cond_mat: torch.Tensor = (1.0 / torch.norm(hessian, torch.inf, 0).sqrt()).diag() # Q
			hessian_precond: torch.Tensor = cond_mat @ hessian @ cond_mat # (QAQ)(Q^{-1}x)=Qb -> x = Q @ solve(QAQ, Qb)
			hessian_precond = (hessian_precond + hessian_precond.T) / 2.0 # symmetrize
			# log
			if DEBUG_MODE:
				print_stuff(
					loss,
					learning_rate,
					self.model,
					1,
					True,
					hessian,
					"Iter {} - last = {:.15e}, ".format(i, last_value),
					"Cond(preconditioned hessian) = {}".format(torch.linalg.cond(hessian_precond).item()),
					True
				)
			else:
				if i % (__class__.__MAX_ITER // 100) == 0:
					print_stuff(
						loss,
						learning_rate,
						self.model,
						1,
						True,
						hessian,
						"Iter {} - last = {:.15e}, ".format(i, last_value),
						"Cond(preconditioned hessian) = {}".format(torch.linalg.cond(hessian_precond).item())
					)
			# stopping criteria
			grad_sqnm: float = sum(torch.sum(param.grad ** 2).item() if param.grad is not None else 0.0 for param in self.model.parameters())
			if grad_sqnm < __class__.__GTOL:
				finish_early = True
				num_iter = i
				print("Convergence: |Gradient| = {} <= GTOL = {}".format(math.sqrt(grad_sqnm), math.sqrt(__class__.__GTOL)))
				break
			if abs(last_value - loss.item()) / max(abs(last_value), abs(loss.item()), 1.0) < __class__.__FTOL:
				finish_early = True
				num_iter = i
				print("Convergence: |f_i - f_{{i+1}}| = {} / {} <= FTOL = {}".format(abs(last_value - loss.item()), max(abs(last_value), abs(loss.item()), 1.0), __class__.__FTOL))
				break
			# change parameter by Newton
			last_value = loss.item()
			ld: torch.Tensor
			pivot: torch.Tensor
			ld, pivot, _ = torch.linalg.ldl_factor_ex(hessian_precond)
			grad_combined: torch.Tensor = torch.cat([(param.grad if param.grad is not None else torch.zeros_like(param)).reshape(-1) for param in self.model.parameters()])
			change: torch.Tensor = (cond_mat @ torch.linalg.ldl_solve(ld, pivot, cond_mat @ grad_combined.reshape(-1, 1))).reshape(-1)
			param_combined: torch.Tensor = torch.cat([param.reshape(-1) for param in self.model.parameters()])
			if learning_rate < 1.0:
				learning_rate *= 2.0
			loss, learning_rate = optimize_with_learning_rate(
				param_combined,
				change,
				param_indices,
				last_value,
				"Newton Method",
				learning_rate
			)
			if loss.item() >= last_value:
				loss, learning_rate = optimize_with_learning_rate(
					param_combined,
					grad_combined,
					param_indices,
					last_value,
					"Gradient Descent"
				)
			if loss.item() >= last_value:
				finish_early = True
				num_iter = i
				break
		if not finish_early:
			print("Stop: Total No. iterations reached limit.")
			num_iter = __class__.__MAX_ITER
		print_stuff(
			loss,
			learning_rate,
			self.model,
			0,
			True,
			hessian,
			"Iter {} - last = {:.15e}, ".format(num_iter, last_value),
			"Cond(preconditioned hessian) = {}".format(torch.linalg.cond(hessian_precond).item()),
			DEBUG_MODE
		)

	def update(
		self,
		x_center: torch.Tensor,
		y_center: torch.Tensor,
		x_all: torch.Tensor,
		y_all: torch.Tensor,
		scale: float
	) -> None:
		"""
		To update the training features and labels of the model

		Parameters
		----------
		x_center : torch.Tensor, shape of (num_points, PHASEDIM)
			Subset training inputs
		y_center : torch.Tensor, shape of (num_points,)
			Subset training targets
		x_all : torch.Tensor, shape of (num_points * (1 + NUM_XTR_RATIO), PHASEDIM)
			All training inputs
		y_all : torch.Tensor, shape of (num_points * (1 + NUM_XTR_RATIO))
			All training targets
		scale : float
			The scaling factor to increase
		"""
		assert x_center.shape[-1] == pes.PHASEDIM and x_center.numel() == y_center.numel() * pes.PHASEDIM
		assert x_all.shape[-1] == pes.PHASEDIM and x_all.numel() == y_all.numel() * pes.PHASEDIM
		self.__x_all = copy.deepcopy(x_all.reshape(-1, pes.PHASEDIM))
		self.__y_all = copy.deepcopy(y_all.reshape(-1))
		self.__scale = scale
		self.model.set_train_data(x_center.reshape(-1, pes.PHASEDIM).detach(), y_center.reshape(-1).detach(), False)
		self.__weights_updated = False

	def get_marginal(self, dimensions: list[int], x_test: torch.Tensor) -> torch.Tensor:
		"""
		To get the marginal distribution of current gaussian process regression

		Parameters
		----------
		dimensions : list[int]
			The dimensions to be kept, must not have any repeat
		x_test : torch.Tensor, shape of (N, len(dimensions))
			Validation/Test inputs

		Returns
		-------
		torch.Tensor, shape of (N,)
			Corresponding validation/test targets based on noise-free SR/PP mean.
		"""
		if not self.__weights_updated:
			self.__update_weights()
		marginal_kernel: gpytorch.kernels.Kernel = gpytorch.kernels.RBFKernel(len(dimensions))
		marginal_kernel.lengthscale = self.model.cov.lengthscale.reshape(1, pes.PHASEDIM)[:, dimensions]
		prefactor: float = np.sqrt((2.0 * torch.pi) ** (pes.PHASEDIM - len(dimensions))) * self.model.cov.lengthscale[:, [i for i in range(pes.PHASEDIM) if i not in dimensions]].prod().item()
		return prefactor * (marginal_kernel(x_test, self.get_training_features()[:, dimensions]) @ self.__k_inv_y).to_dense()


class GPRPredictors:
	"""
	Combination of single predictors

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
	get_marginal(dimensions, x_input, ElementIndex)
		To get the marginal distribution over given dimensions
	print(f)
		To print parameters to file
	"""
	@staticmethod
	def __check_predictor(predictor: SinglePredictor) -> bool:
		"""
		To check if the predictor could be used for training / predicting

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
		return isinstance(predictor.model.train_targets, torch.Tensor) and not torch.all(predictor.model.train_targets == 0)

	__slots__: tuple = ("__predictors", "__scale")

	def __init__(self, parameter_initial_values: npt.NDArray[np.double] | None = None):
		kernel: gpytorch.kernels.Kernel
		if parameter_initial_values is not None:
			assert parameter_initial_values.size == pes.PHASEDIM
			kernel = gpytorch.kernels.RBFKernel(pes.PHASEDIM)
			kernel.lengthscale = torch.from_numpy(parameter_initial_values)
		else:
			kernel = gpytorch.kernels.RBFKernel(pes.PHASEDIM)
		self.__predictors: list[SinglePredictor] = [SinglePredictor(kernel) for _ in range(pes.NUM_ELM)]
		self.__scale: npt.NDArray[np.double] = np.ones(pes.NUM_ELM, np.double)

	def __getitem__(self, ElementIndex: int) -> SinglePredictor:
		"""
		To get corresponding predictor

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

	def update(
		self,
		x_center: npt.NDArray[np.double],
		y_center: npt.NDArray[np.cdouble],
		x_all: npt.NDArray[np.double],
		y_all: npt.NDArray[np.cdouble],
		scale: npt.NDArray[np.double]
	) -> None:
		"""
		To update the training inputs and targets, as well as the rescale factor

		Parameters
		----------
		x_center : npt.NDArray[np.double], shape of (NUM_TRIG, num_points, PHASEDIM)
			Subset training inputs
		y_center : npt.NDArray[np.cdouble], shape of (NUM_TRIG, num_points)
			Subset training targets
		x_all : npt.NDArray[np.double], shape of (NUM_TRIG, num_points * (1 + NUM_XTR_RATIO), PHASEDIM)
			All training inputs
		y_all : npt.NDArray[np.cdouble], shape of (NUM_TRIG, num_points * (1 + NUM_XTR_RATIO))
			All training targets
		scale : npt.NDArray[np.double], shape of (NUM_ELM,)
			The rescale factor
		"""
		self.__scale[...] = scale
		for iElement, pred in enumerate(self.__predictors):
			RowIndex: int = iElement // pes.NUM_PES
			ColIndex: int = iElement % pes.NUM_PES
			TrilIndex: int = pes.flatten_tril_index[RowIndex, ColIndex]
			pred.update(
				torch.from_numpy(x_center[TrilIndex]),
				torch.from_numpy(y_center[TrilIndex].real if RowIndex <= ColIndex else y_center[TrilIndex].imag),
				torch.from_numpy(x_all[TrilIndex]),
				torch.from_numpy(y_all[TrilIndex].real if RowIndex <= ColIndex else y_all[TrilIndex].imag),
				self.__scale[iElement]
			)

	def train(self) -> None:
		"""
		To train each predictor
		"""
		for iElement in range(pes.NUM_ELM):
			if __class__.__check_predictor(self.__predictors[iElement]):
				print("Training " + utility.get_RI_label(iElement))
				self.__predictors[iElement].train()

	def predict(self, x_input: npt.NDArray[np.double], ElementIndex: int) -> npt.NDArray[np.cdouble]:
		"""
		To predict test targets based on input and corresponding density matrix element

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
		def call_single_predictor(pred: SinglePredictor, x_test: torch.Tensor) -> npt.NDArray[np.double]:
			"""
			To do prediction of a single predictor

			Parameters
			----------
			pred : SinglePredictor
				The predictor
			x_test : torch.Tensor, shape of (N, PHASEDIM)
				Test inputs

			Returns
			-------
			npt.NDArray[np.double], shape of (N,)
				Test targets by the predictor
			"""
			if __class__.__check_predictor(pred):
				return pred.predict(x_test).detach().numpy()
			else:
				return np.zeros(x_test.shape[0], np.double)

		assert x_input.shape[-1] == pes.PHASEDIM and 0 <= ElementIndex < pes.NUM_ELM
		x_test: torch.Tensor = torch.from_numpy(x_input.reshape(-1, pes.PHASEDIM))
		RowIndex: int = ElementIndex // pes.NUM_PES
		ColIndex: int = ElementIndex % pes.NUM_PES
		result: npt.NDArray[np.cdouble] = np.empty(x_test.shape[0], np.cdouble)
		if RowIndex == ColIndex:
			result.real = call_single_predictor(self.__predictors[ElementIndex], x_test)
			result.imag = 0
		elif RowIndex > ColIndex:
			result.real = call_single_predictor(self.__predictors[ColIndex * pes.NUM_PES + RowIndex], x_test)
			result.imag = call_single_predictor(self.__predictors[ElementIndex], x_test)
		else: # RowIndex < ColIndex
			result.real = call_single_predictor(self.__predictors[ElementIndex], x_test)
			result.imag = -call_single_predictor(self.__predictors[ColIndex * pes.NUM_PES + RowIndex], x_test)
		return result.reshape(x_input.shape[:-1])

	def get_marginal(
		self,
		dimensions: int | typing.Iterable[int],
		x_input: npt.NDArray[np.double],
		ElementIndex: int
	) -> npt.NDArray[np.cdouble]:
		"""
		To get the marginal distribution of current gaussian process regressions

		Parameters
		----------
		dimensions : int | typing.Iterable[int]
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
		def call_single_predictor(pred: SinglePredictor, dims: list[int], x_test: torch.Tensor) -> npt.NDArray[np.double]:
			"""
			To do prediction of a single predictor

			Parameters
			----------
			pred : SinglePredictor
				The predictor
			dims : list[int]
				The dims to be kept
			x_input : torch.Tensor, shape of (N, len(dimensions))
				Test inputs

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
		assert x_input.shape[-1] == len(dimensions) and 0 <= ElementIndex < pes.NUM_ELM
		x_test: torch.Tensor = torch.from_numpy(x_input.reshape(-1, len(dimensions)))
		RowIndex: int = ElementIndex // pes.NUM_PES
		ColIndex: int = ElementIndex % pes.NUM_PES
		result: npt.NDArray[np.cdouble] = np.empty(x_test.shape[0], np.cdouble)
		if RowIndex == ColIndex:
			result.real = call_single_predictor(self.__predictors[ElementIndex], dimensions, x_test)
			result.imag = 0
		elif RowIndex > ColIndex:
			result.real = call_single_predictor(self.__predictors[ColIndex * pes.NUM_PES + RowIndex], dimensions, x_test)
			result.imag = call_single_predictor(self.__predictors[ElementIndex], dimensions, x_test)
		else: # RowIndex < ColIndex
			result.real = call_single_predictor(self.__predictors[ElementIndex], dimensions, x_test)
			result.imag = -call_single_predictor(self.__predictors[ColIndex * pes.NUM_PES + RowIndex], dimensions, x_test)
		return result.reshape(x_input.shape[:-1])

	def print(self, f: io.TextIOWrapper) -> None:
		"""
		To print the parameters to file

		Parameters
		----------
		f : io.TextIOWrapper
			The file to save the parameters
		"""
		for predictor in self.__predictors:
			np.savetxt(f, predictor.model.cov.lengthscale.detach().numpy().reshape(1, -1))
		print("\n", file=f)
