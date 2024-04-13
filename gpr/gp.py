"""
gp
==
This module provides support for gaussian process (gp) regression.
"""
import copy
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
	get_training_features()
		To get the training features of the core subset
	update_weights()
		To update the weights, :math:`K^{-1}y`
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
	MAX_ITER: typing.Literal[50000] = 50000
	FTOL: float = 2.2204460492503131e-09
	GTOL: float = 1e-5
	NOISE: float = 1e-4

	def __init__(self, kernel: gpytorch.kernels.Kernel):
		self.kernel: gpytorch.kernels.Kernel = copy.deepcopy(kernel)
		likelihood: gpytorch.likelihoods.FixedNoiseGaussianLikelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.full((2,), __class__.NOISE))
		self.x_all: torch.Tensor = torch.Tensor()
		self.y_all: torch.Tensor = torch.Tensor()
		self.scale: float = 1.0
		self.model: GP = GP(torch.zeros((2, pes.PHASEDIM)), torch.zeros((2,)), likelihood, self.kernel)
		self.model_param: dict[str, torch.Tensor] = copy.deepcopy(self.model.state_dict())
		self.k_inv_y: torch.Tensor = torch.Tensor()
		self.weights_updated: bool = False

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

	def update_weights(self) -> None:
		"""
		To update the weights, :math:`K^{-1}y`
		"""
		self.k_inv_y = (linear_operator.utils.stable_pinverse(self.model.cov(self.x_all, self.get_training_features()).to_dense()) @ self.y_all).to_dense()
		self.weights_updated = True

	def get_weights(self) -> torch.Tensor:
		"""
		To get the weights, :math:`K^{-1}y`

		Returns
		-------
		torch.Tensor
			Weights, :math:`K^{-1}y`
		"""
		if not self.weights_updated:
			self.update_weights()
		return self.k_inv_y

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
		Instance of prediction of subset of regressor (SR) / projected process (PP)
		"""
		if not self.weights_updated:
			self.update_weights()
		return (self.model.cov(x_test, self.get_training_features()) @ self.k_inv_y).to_dense()

	def error(self, use_weight: bool = True) -> torch.Tensor:
		"""
		Error function of subset of regressor (SR) / projected process (PP)

		This function gives the sum of squared error

		Returns
		-------
		torch.Tensor
			The sum of squared prediction error
		"""
		if use_weight:
			return torch.sum((self.y_all - self.predict(self.x_all)) ** 2) * (self.scale ** 2)
		else:
			kmn: torch.Tensor = self.model.cov(self.x_all, self.get_training_features()).to_dense()
			return torch.sum((self.y_all - (kmn @ (linear_operator.utils.stable_pinverse(kmn) @ self.y_all)).to_dense()) ** 2) * (self.scale ** 2)

	def train(self) -> None:
		"""
		To train the parameters
		"""
		def print_model(model: gpytorch.models.ExactGP, print_grad: bool = False) -> None:
			"""
			To print the parameters of the model

			Parameters
			----------
			model : gpytorch.models.ExactGP
				Gaussian process model, containing mean and covariances and their parameters
			print_grad : bool, optional
				Whether to print the gradient or not, by default False
			"""
			for param_name, param, constraint in model.named_parameters_and_constraints():
				if print_grad and param.grad is not None:
					print(
						"Parameter name: {}".format(param_name),
						utility.format_array("value", param),
						utility.format_array("grad", param.grad)
					)
				else:
					print(
						"Parameter name: {}".format("".join(param_name.split("raw_"))),
						utility.format_array("value", constraint.transform(param) if isinstance(constraint, gpytorch.constraints.Interval) else param)
					)

		def get_lr(optimizer: torch.optim.Optimizer) -> float:
			"""
			To get the learning rate of the optimizer

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

		def print_stuff(loss: torch.Tensor, optimizer: torch.optim.Optimizer, model: gpytorch.models.ExactGP, print_grad: bool = False, extra_str="\t") -> None:
			"""
			To print all stuffs needed

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
			print(extra_str + "loss = {:.15e}, lr = {}".format(loss.item(), get_lr(optimizer)))
			print_model(model, print_grad)

		self.weights_updated = False
		assert isinstance(self.model.train_targets, torch.Tensor)
		# train model
		self.model.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.full((self.get_training_features().shape[0],), __class__.NOISE))
		self.model.train()
		self.model.likelihood.train()
		self.model.load_state_dict(self.model_param)
		# print_model(self.model)
		finish_early: bool = False
		optimizer: torch.optim.Optimizer = torch.optim.Rprop(self.model.parameters(), lr=1.0)
		optimizer.zero_grad()
		loss: torch.Tensor = self.error(False)
		loss.backward()
		last_value: float = loss.item()
		print_stuff(loss, optimizer, self.model, True, "Init")
		for i in range(1, __class__.MAX_ITER + 1):
			# adjust lr
			old_prm: dict[str, torch.Tensor] = copy.deepcopy(self.model.state_dict())
			optimizer.step()
			loss = self.error(False)
			# print_stuff(loss, optimizer, self.model, True)
			if loss < last_value:
				optimizer = optimizer.__class__(self.model.parameters(), lr=get_lr(optimizer) * 2.0)
				# print("loss < last_value")
				# print_stuff(loss, optimizer, self.model, True)
			else:
				# print("loss > last_value")
				while loss >= last_value:
					last_loop_value: float = loss.item()
					self.model.load_state_dict(old_prm)
					optimizer = optimizer.__class__(self.model.parameters(), lr=get_lr(optimizer) / 2.0)
					optimizer.step()
					loss = self.error(False)
					# print_stuff(loss, optimizer, self.model, True)
					if last_loop_value == loss.item():
						print("No stepping forward")
						# no stepping forward, but still larger than last, meaning last is the best
						self.model.load_state_dict(old_prm)
						loss = self.error(False)
						break
			if i % (__class__.MAX_ITER // 100) == 0:
				print("Iter {} - Loss: {:.15e} - lr: {}".format(i, loss.item(), get_lr(optimizer)))
				print_model(self.model, True)
				print_model(self.model)
			# stopping criteria
			if (last_value - loss.item()) / max(abs(last_value), abs(loss.item()), 1.0) < __class__.FTOL:
				finish_early = True
				print("Convergence: |f_i - f_{i+1}| <= FTOL")
				print("Iter {} - Loss: {:.15e} - lr: {}".format(i, loss.item(), get_lr(optimizer)))
				print_model(self.model, True)
				print_model(self.model)
				break
			if np.sqrt(sum(torch.sum(param.grad ** 2).item() if param.grad is not None else 0.0 for param in self.model.parameters())) < __class__.GTOL:
				finish_early = True
				print("Convergence: |Gradient| <= GTOL")
				print("Iter {} - Loss: {:.15e} - lr: {}".format(i, loss.item(), get_lr(optimizer)))
				print_model(self.model, True)
				print_model(self.model)
				break
			optimizer = optimizer.__class__(self.model.parameters(), lr=get_lr(optimizer))
			optimizer.zero_grad()
			last_value = loss.item()
			loss.backward()
			# print_stuff(loss, optimizer, self.model, True, "\tlast = {}, ".format(last_value))
		if not finish_early:
			print("Stop: Total No. iterations reached limit.")
			print("Iter {} - Loss: {:.15e} - lr: {}".format(__class__.MAX_ITER, last_value, [param["lr"] for param in optimizer.param_groups]))
			print_model(self.model, True)
			print_model(self.model)
		print_model(self.model)
		print("", flush=True)
		self.model_param = copy.deepcopy(self.model.state_dict())

	def update(
		self,
		x_all: torch.Tensor,
		y_all: torch.Tensor,
		scale: float,
		num_points: int
	) -> None:
		"""
		To update the training features and labels of the model

		Parameters
		----------
		x_all : torch.Tensor, of shape (num_points * (1 + NUM_XTR_RATIO), PHASEDIM)
			All training inputs
		y_all : torch.Tensor, of shape (num_points * (1 + NUM_XTR_RATIO))
			All training targets
		scale : float
			The scaling factor to increase
		num_points : int
			The number of points located at the front of all points that is used as the subset
		"""
		assert x_all.shape[-1] == pes.PHASEDIM and x_all.numel() == y_all.numel() * pes.PHASEDIM
		self.x_all = copy.deepcopy(x_all.reshape(-1, pes.PHASEDIM))
		self.y_all = copy.deepcopy(y_all.reshape(-1))
		self.scale = scale
		self.model.set_train_data(self.x_all[:num_points].detach(), self.y_all[:num_points].detach(), False)
		self.weights_updated = False

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
		if not self.weights_updated:
			self.update_weights()
		marginal_kernel: gpytorch.kernels.Kernel = gpytorch.kernels.RBFKernel(len(dimensions))
		marginal_kernel.lengthscale = self.model.cov.lengthscale.reshape(1, pes.PHASEDIM)[:, dimensions]
		prefactor: float = np.sqrt((2.0 * torch.pi) ** (pes.PHASEDIM - len(dimensions))) * self.model.cov.lengthscale[:, [i for i in range(pes.PHASEDIM) if i not in dimensions]].prod().item()
		return prefactor * (marginal_kernel(x_test, self.get_training_features()[:, dimensions]) @ self.k_inv_y).to_dense()


class GPRPredictors:
	"""
	Combination of single predictors

	Parameters
	----------
	kernel : gpytorch.kernels.Kernel, optional
		The kernel of predictors, by default gpytorch.kernels.RBFKernel(pes.PHASEDIM)

	Methods
	-------
	check_predictor(predictor)
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
	def check_predictor(predictor: SinglePredictor) -> bool:
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

	def __init__(self, parameter_initial_values: npt.NDArray[np.double] | None = None):
		kernel: gpytorch.kernels.Kernel
		if parameter_initial_values is not None:
			assert parameter_initial_values.size == pes.PHASEDIM
			kernel = gpytorch.kernels.RBFKernel(pes.PHASEDIM)
			kernel.lengthscale = torch.from_numpy(parameter_initial_values)
		else:
			kernel = gpytorch.kernels.RBFKernel(pes.PHASEDIM)
		self.predictors: list[SinglePredictor] = [SinglePredictor(kernel) for _ in range(pes.NUM_ELM)]
		self.scale: npt.NDArray[np.double] = np.ones(pes.NUM_ELM, np.double)

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
		return self.predictors[ElementIndex]

	def update(
		self,
		x_all: list[npt.NDArray[np.double]],
		y_all: list[npt.NDArray[np.cdouble]],
		num_points: int | npt.NDArray[np.int_],
		scale: npt.NDArray[np.double]
	) -> None:
		"""
		To update the training inputs and targets, as well as the rescale factor

		Parameters
		----------
		x_all : list[npt.NDArray[np.double]], len of NUM_TRIG, each of shape (num_points * (1 + NUM_XTR_RATIO), PHASEDIM)
			All training inputs
		y_all : list[npt.NDArray[np.cdouble]], len of NUM_TRIG, each of shape (num_points * (1 + NUM_XTR_RATIO))
			All training targets
		num_points : int | npt.NDArray[np.int_], shape of (NUM_TRIG,)
			The number of points located at the front of all points that is used as the subset
		scale : npt.NDArray[np.double], shape of (NUM_ELM,)
			The rescale factor
		"""
		if isinstance(num_points, int):
			num_points = np.full(pes.NUM_TRIG, num_points, np.int_)
		self.scale[...] = scale
		for iElement, pred in enumerate(self.predictors):
			RowIndex: int = iElement // pes.NUM_PES
			ColIndex: int = iElement % pes.NUM_PES
			TrilIndex: int = pes.flatten_tril_index[RowIndex, ColIndex]
			pred.update(
				torch.from_numpy(x_all[TrilIndex]),
				torch.from_numpy(y_all[TrilIndex].real if RowIndex <= ColIndex else y_all[TrilIndex].imag),
				self.scale[iElement],
				num_points[TrilIndex]
			)

	def train(self) -> None:
		"""
		To train each predictor
		"""
		for iElement in range(pes.NUM_ELM):
			if __class__.check_predictor(self.predictors[iElement]):
				print("Training " + utility.get_RI_label(iElement))
				self.predictors[iElement].train()

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
			if __class__.check_predictor(pred):
				return pred.predict(x_test).detach().numpy()
			else:
				return np.zeros(x_test.shape[0], np.double)

		assert x_input.shape[-1] == pes.PHASEDIM and 0 <= ElementIndex < pes.NUM_ELM
		x_test: torch.Tensor = torch.from_numpy(x_input.reshape(-1, pes.PHASEDIM))
		RowIndex: int = ElementIndex // pes.NUM_PES
		ColIndex: int = ElementIndex % pes.NUM_PES
		result: npt.NDArray[np.cdouble] = np.empty(x_test.shape[0], np.cdouble)
		if RowIndex == ColIndex:
			result.real = call_single_predictor(self.predictors[ElementIndex], x_test)
			result.imag = 0
		elif RowIndex > ColIndex:
			result.real = call_single_predictor(self.predictors[ColIndex * pes.NUM_PES + RowIndex], x_test)
			result.imag = call_single_predictor(self.predictors[ElementIndex], x_test)
		else: # RowIndex < ColIndex
			result.real = call_single_predictor(self.predictors[ElementIndex], x_test)
			result.imag = -call_single_predictor(self.predictors[ColIndex * pes.NUM_PES + RowIndex], x_test)
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
			if __class__.check_predictor(pred):
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
			result.real = call_single_predictor(self.predictors[ElementIndex], dimensions, x_test)
			result.imag = 0
		elif RowIndex > ColIndex:
			result.real = call_single_predictor(self.predictors[ColIndex * pes.NUM_PES + RowIndex], dimensions, x_test)
			result.imag = call_single_predictor(self.predictors[ElementIndex], dimensions, x_test)
		else: # RowIndex < ColIndex
			result.real = call_single_predictor(self.predictors[ElementIndex], dimensions, x_test)
			result.imag = -call_single_predictor(self.predictors[ColIndex * pes.NUM_PES + RowIndex], dimensions, x_test)
		return result.reshape(x_input.shape[:-1])

	def print(self, f: io.TextIOWrapper) -> None:
		"""
		To print the parameters to file

		Parameters
		----------
		f : io.TextIOWrapper
			The file to save the parameters
		"""
		for predictor in self.predictors:
			np.savetxt(f, predictor.model.cov.lengthscale.detach().numpy().reshape(1, -1))
		print("\n", file=f)
