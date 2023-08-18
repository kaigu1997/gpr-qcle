"""
gp
==

This module provides support for gaussian process (gp) regression.
"""

import copy
import io
import typing

import gpytorch
import gpytorch.constraints
import linear_operator
import numpy as np
import numpy.typing as npt
import torch

import pes

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


def get_RI_label(row: int, col: int) -> str:
	"""
	To have the real/imaginary part name of the given input row and column.

	Strictly-upper triangular is real, strictly-lower triangular is imaginary, and diagonal elements are as they are.

	Parameters
	----------
	row : int
		Index of row
	col : int
		Index of column

	Returns
	-------
	str
		Name of the real/imaginary part. 
	"""
	if row == col:
		return r'$\rho_{%d,%d}$' % (row, col)
	elif row < col:
		return r'$\Re\rho_{%d,%d}$' % (col, row)
	else:
		return r'$\Im\rho_{%d,%d}$' % (row, col)


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


def sr_pred(
	model: GP,
	likelihood: gpytorch.likelihoods.GaussianLikelihood,
	x: torch.Tensor,
	x_all: torch.Tensor,
	y_all: torch.Tensor,
	x_test: torch.Tensor
) -> torch.Tensor:
	"""
	Instance of prediction of subset of regressor (SR) / projected process (PP)

	Parameters
	----------
	model : GP
		Gaussian process model, including mean and covariance information
	likelihood : gpytorch.likelihoods.GaussianLikelihood
		Likelihood, containing noise information
	x : torch.Tensor
		Training inputs of the selected subset
	x_all : torch.Tensor
		All training inputs
	y_all : torch.Tensor
		All training targets
	x_test : torch.Tensor
		Validation/Test inputs

	Returns
	-------
	torch.Tensor
		Corresponding validation/test targets based on noise-free SR/PP mean.
	"""
	return (model.cov(x_test, x) @ (linear_operator.utils.stable_pinverse(model.cov(x_all, x).to_dense()) @ y_all)).to_dense()


def sr_se(
	x: torch.Tensor,
	y: torch.Tensor,
	model: GP,
	likelihood: gpytorch.likelihoods.GaussianLikelihood,
	x_all: torch.Tensor,
	y_all: torch.Tensor
) -> torch.Tensor:
	"""
	Error function of subset of regressor (SR) / projected process (PP)

	This function gives the sum of squared error

	Parameters
	----------
	x : torch.Tensor
		Training inputs of the selected subset
	y : torch.Tensor
		Training targets of the selected subset
	model : GP
		Gaussian process model, including mean and covariance information
	likelihood : gpytorch.likelihoods.GaussianLikelihood
		Likelihood, containing noise information
	x_all : torch.Tensor
		All training inputs
	y_all : torch.Tensor
		All training targets

	Returns
	-------
	torch.Tensor, shape of (1,)
		Sum of squared error, remaining in tensor form for autograd
	"""
	return ((y_all - sr_pred(model, likelihood, x, x_all, y_all, x_all)) ** 2).sum()


class SinglePredictor:
	"""
	The instantiation of gaussian process predictor

	Parameters
	----------
	kernel : gpytorch.kernels.Kernel
		The kernel to use
	loss_func : typing.Callable[[torch.Tensor, torch.Tensor, GP, gpytorch.likelihoods.GaussianLikelihood, torch.Tensor, torch.Tensor], torch.Tensor], optional
		The loss function for optimization, by default sr_se
	predictor : typing.Callable[[GP, gpytorch.likelihoods.GaussianLikelihood, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None, optional
		The way to do prediction, by default sr_pred

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
	train()
		To train the parameters
	error()
		To get the error from the loss function
	variance()
		To get the variance of the predictor
	"""
	MAX_ITER: typing.Literal[50000] = 50000
	FTOL: float = 2.2204460492503131e-09
	GTOL: float = 1e-5
	NOISE: float = 1e-4

	def __init__(
		self,
		kernel: gpytorch.kernels.Kernel,
		loss_func: typing.Callable[[torch.Tensor, torch.Tensor, GP, gpytorch.likelihoods.GaussianLikelihood, torch.Tensor, torch.Tensor], torch.Tensor] = sr_se,
		predictor: typing.Callable[[GP, gpytorch.likelihoods.GaussianLikelihood, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = sr_pred
	):
		self.kernel: gpytorch.kernels.Kernel = copy.deepcopy(kernel)
		self.loss_func: typing.Callable = loss_func
		self.predictor: typing.Callable | None = predictor
		likelihood: gpytorch.likelihoods.FixedNoiseGaussianLikelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.full((2,), SinglePredictor.NOISE))
		self.x_all: torch.Tensor = torch.Tensor()
		self.y_all: torch.Tensor = torch.Tensor()
		self.model: GP = GP(torch.zeros((2, pes.PHASEDIM)), torch.zeros((2,)), likelihood, self.kernel)
		self.model_param: dict[str, torch.Tensor] = copy.deepcopy(self.model.state_dict())

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
			def print_tensor(t: torch.Tensor) -> float | npt.NDArray[np.double]:
				"""
				To transform a torch Tensor to read-friendly form

				If the tensor contains only 1 element, return the element;
				otherwise, return the flattened numpy array

				Parameters
				----------
				t : torch.Tensor
					The tensor

				Returns
				-------
				float | npt.NDArray[np.double]
					Return the only element or the flattened array
				"""
				if t.dim() == 0 or np.prod(t.shape) == 1:
					return t.item()
				else:
					return t.detach().numpy().ravel()

			fmt: str = 'Parameter name: {0:42} value = {1}'
			fmt_grad: str = fmt + ' grad = {2}'
			for param_name, param, constraint in model.named_parameters_and_constraints():
				if print_grad and param.grad is not None:
					print(fmt_grad.format(param_name, print_tensor(param), print_tensor(param.grad)))
				else:
					print(fmt.format(''.join(param_name.split('raw_')), print_tensor(constraint.transform(param) if isinstance(constraint, gpytorch.constraints.Interval) else param)))

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
			return optimizer.param_groups[0]['lr']

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
			print('\n')

		assert self.model.train_inputs is not None and isinstance(self.model.train_targets, torch.Tensor)
		x: torch.Tensor = self.model.train_inputs[0]
		y: torch.Tensor = self.model.train_targets
		# train model
		likelihood: gpytorch.likelihoods.FixedNoiseGaussianLikelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.full((x.shape[0],), SinglePredictor.NOISE))
		self.model.likelihood = likelihood
		self.model.train()
		likelihood.train()
		self.model.load_state_dict(self.model_param)
		# print_model(model)
		finish_early: bool = False
		optimizer: torch.optim.Optimizer = torch.optim.Rprop(self.model.parameters(), lr=1.0)
		optimizer.zero_grad()
		loss: torch.Tensor = self.loss_func(x, y, self.model, likelihood, self.x_all, self.y_all)
		loss.backward()
		last_value: float = loss.item()
		print_stuff(loss, optimizer, self.model, True, "\n\nInit")
		for i in range(1, SinglePredictor.MAX_ITER + 1):
			# adjust lr
			old_prm: dict[str, torch.Tensor] = copy.deepcopy(self.model.state_dict())
			optimizer.step()
			loss = self.loss_func(x, y, self.model, likelihood, self.x_all, self.y_all)
			# print_stuff(loss, optimizer, self.model, True)
			if loss < last_value:
				optimizer = optimizer.__class__(self.model.parameters(), lr=get_lr(optimizer) * 2.0)
				# print('loss < last_value\n')
				# while loss < last_value:
				# 	second_last_value: float = last_value
				# 	last_value = loss.item()
				# 	self.model.load_state_dict(old_prm)
				# 	optimizer = optimizer.__class__(self.model.parameters(), lr=get_lr(optimizer) * 2.0)
				# 	optimizer.step()
				# 	loss = self.loss_func(x, y, self.model, likelihood, self.x_all, self.y_all)
				# 	# print_stuff(loss, optimizer, self.model, True)
				# 	if loss >= last_value:
				# 		# goes back, not only loss but also last value, for stop criteria judgment
				# 		last_value = second_last_value
				# 		break
				# # when exit, loss >= last value, so learning rate should be halved
				# self.model.load_state_dict(old_prm)
				# optimizer = optimizer.__class__(self.model.parameters(), lr=get_lr(optimizer) * 2.0)
				# optimizer.step()
				# loss = self.loss_func(x, y, self.model, likelihood, self.x_all, self.y_all)
				# print_stuff(loss, optimizer, self.model, True)
			else:
				# print('loss > last_value')
				while loss >= last_value:
					last_loop_value: float = loss.item()
					self.model.load_state_dict(old_prm)
					optimizer = optimizer.__class__(self.model.parameters(), lr=get_lr(optimizer) / 2.0)
					optimizer.step()
					loss = self.loss_func(x, y, self.model, likelihood, self.x_all, self.y_all)
					# print_stuff(loss, optimizer, self.model, True)
					if last_loop_value == loss.item():
						# no stepping forward, but still larger than last, meaning last is the best
						self.model.load_state_dict(old_prm)
						loss = self.loss_func(x, y, self.model, likelihood, self.x_all, self.y_all)
						break
			if i % (SinglePredictor.MAX_ITER // 100) == 0:
				print('\n')
				print('Iter {} - Loss: {:.15e} - lr: {}'.format(i, loss.item(), get_lr(optimizer)))
				print_model(self.model, True)
				print_model(self.model)
				print('\n')
			# stopping criteria
			if (last_value - loss.item()) / max(abs(last_value), abs(loss.item()), 1.0) < SinglePredictor.FTOL:
				finish_early = True
				print("Convergence: |f_i - f_{i+1}| <= FTOL")
				print('Iter {} - Loss: {:.15e} - lr: {}'.format(i, loss.item(), get_lr(optimizer)))
				break
			if np.sqrt(sum(torch.sum(param.grad ** 2).item() if param.grad is not None else 0.0 for param in self.model.parameters())) < SinglePredictor.GTOL:
				finish_early = True
				print("Convergence: |Gradient| <= GTOL")
				print('Iter {} - Loss: {:.15e} - lr: {}'.format(i, loss.item(), get_lr(optimizer)))
				break
			optimizer = optimizer.__class__(self.model.parameters(), lr=get_lr(optimizer))
			optimizer.zero_grad()
			last_value = loss.item()
			loss.backward()
			# print_stuff(loss, optimizer, self.model, True, "\n\n\tlast = {}, ".format(last_value))
		if not finish_early:
			print('Iter {} - Loss: {:.15e} - lr: {}'.format(SinglePredictor.MAX_ITER, last_value, [param['lr'] for param in optimizer.param_groups]))
			print("Stop: Total No. iterations reached limit.")
		print_model(self.model)
		print('\n')
		self.model_param = copy.deepcopy(self.model.state_dict())

	def error(self) -> float:
		"""
		To calculate the error from the loss function

		Returns
		-------
		float
			The error
		"""
		assert self.model.train_inputs is not None and isinstance(self.model.train_targets, torch.Tensor)
		x: torch.Tensor = self.model.train_inputs[0]
		y: torch.Tensor = self.model.train_targets
		return self.loss_func(x, y, self.model, gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.full((x.shape[0],), SinglePredictor.NOISE)), self.x_all, self.y_all).item()

	def variance(self) -> npt.NDArray[np.double]:
		"""
		To calculate the variance based on the squared density.

		The variance is half the square of characteristic lengths for each dimension.

		Returns
		-------
		npt.NDArray[np.double]
			Variance
		"""
		return self.model.cov.lengthscale.detach().numpy().reshape(-1) ** 2 / 2.0


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


def get_scale(scale: np.double) -> np.double:
	"""
	Return 1 if the scale is 0, else the value itself

	Parameters
	----------
	scale : np.double
		A non-negative value

	Returns
	-------
	np.double
		1 if the scale is 0, else the value itself
	"""
	if scale == 0.0:
		return np.double(1.0)
	else:
		return scale


class GPRPredictors:
	"""
	Combination of single predictors

	Parameters
	----------
	kernel : gpytorch.kernels.Kernel, optional
		The kernel of predictors, by default gpytorch.kernels.RBFKernel(pes.PHASEDIM)
	loss_func : typing.Callable[[torch.Tensor, torch.Tensor, GP, gpytorch.likelihoods.GaussianLikelihood, torch.Tensor, torch.Tensor], torch.Tensor], optional
		The loss function for optimization of predictors, by default sr_se
	predictor : typing.Callable[[GP, gpytorch.likelihoods.GaussianLikelihood, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None, optional
		The way to do prediction, by default sr_pred

	Methods
	-------
	update(x_all, y_all, num_pt, scale)
		To update the training inputs and targets, as well as the rescale factor
	train()
		To train each predictor
	predict(x_input, ElementIndex)
		To predict test targets based on input and corresponding density matrix element
	print(f)
		To print hyperparameters to file
	"""
	def __init__(
		self,
		kernel: gpytorch.kernels.Kernel = gpytorch.kernels.RBFKernel(pes.PHASEDIM),
		loss_func: typing.Callable[[torch.Tensor, torch.Tensor, GP, gpytorch.likelihoods.GaussianLikelihood, torch.Tensor, torch.Tensor], torch.Tensor] = sr_se,
		predictor: typing.Callable[[GP, gpytorch.likelihoods.GaussianLikelihood, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = sr_pred
	):
		self.predictors: npt.NDArray[np.object_] = np.array([SinglePredictor(kernel, loss_func, predictor) for i in range(pes.NUM_ELM)], np.object_)
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

	def update(self, x_all: npt.NDArray[np.double], y_all: npt.NDArray[np.cdouble], num_pts: int, scale: npt.NDArray[np.double]) -> None:
		"""
		To update the training inputs and targets, as well as the rescale factor

		Parameters
		----------
		x_all : npt.NDArray[np.double], shape of (NUM_ELM, NUM_PTS * (1 + NUM_XTR_RATIO), PHASEDIM)
			All training inputs
		y_all : npt.NDArray[np.cdouble], shape of (NUM_ELM, NUM_PTS * (1 + NUM_XTR_RATIO))
			All training targets
		num_pts : int
			The number of points located at the front of all points that is used as the subset
		scale : npt.NDArray[np.double]
			The rescale factor
		"""
		self.scale[...] = scale
		for iElement in range(pes.NUM_ELM):
			self.predictors[iElement].x_all = copy.deepcopy(torch.from_numpy(x_all[iElement]))
			if iElement // pes.NUM_PES <= iElement % pes.NUM_PES:
				self.predictors[iElement].y_all = copy.deepcopy(torch.from_numpy(y_all[iElement].real))
			else:
				self.predictors[iElement].y_all = copy.deepcopy(torch.from_numpy(y_all[iElement].imag))
			self.predictors[iElement].y_all *= self.scale[iElement]
			self.predictors[iElement].model.set_train_data(self.predictors[iElement].x_all[:num_pts], self.predictors[iElement].y_all[:num_pts], False)

	def train(self) -> None:
		"""
		To train each predictor
		"""
		for iElement in range(pes.NUM_ELM):
			if check_predictor(self.predictors[iElement]):
				self.predictors[iElement].train()

	def predict(self, x_input: npt.NDArray[np.double], ElementIndex: int) -> npt.NDArray[np.cdouble]:
		"""
		To predict test targets based on input and corresponding density matrix element

		Parameters
		----------
		x_input : npt.NDArray[np.double]
			Test inputs
		ElementIndex : int
			Index of the element

		Returns
		-------
		npt.NDArray[np.cdouble]
			Density of the element of all test inputs
		"""
		x_test: torch.Tensor = torch.from_numpy(x_input.reshape(-1, pes.PHASEDIM))

		def call_single_predictor(pred: SinglePredictor) -> npt.NDArray[np.double]:
			"""
			To do prediction of a single predictor

			Parameters
			----------
			pred : SinglePredictor
				The predictor

			Returns
			-------
			npt.NDArray[np.double]
				Test targets by the predictor
			"""
			if check_predictor(pred):
				assert pred.model.train_inputs is not None and isinstance(pred.model.train_inputs[0], torch.Tensor)
				likelihood: gpytorch.likelihoods.FixedNoiseGaussianLikelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.full((pred.model.train_inputs[0].shape[0],), SinglePredictor.NOISE))
				# predict
				pred.model.eval()
				likelihood.eval()
				if pred.predictor is None:
					return likelihood(pred.model(x_test), noise=torch.full((x_test.shape[0],), SinglePredictor.NOISE)).mean.detach().numpy()
				else:
					return pred.predictor(pred.model, likelihood, pred.model.train_inputs[0], pred.x_all, pred.y_all, x_test).detach().numpy()
			else:
				return np.zeros(x_test.shape[0], np.double)

		RowIndex: int = ElementIndex // pes.NUM_PES
		ColIndex: int = ElementIndex % pes.NUM_PES
		result: npt.NDArray[np.cdouble] = np.empty(x_test.shape[0], np.cdouble)
		if RowIndex == ColIndex:
			result.real = call_single_predictor(self.predictors[ElementIndex]) / get_scale(self.scale[ElementIndex])
		elif RowIndex > ColIndex:
			result.real = call_single_predictor(self.predictors[ColIndex * pes.NUM_PES + RowIndex]) / get_scale(self.scale[ColIndex * pes.NUM_PES + RowIndex])
			result.imag = call_single_predictor(self.predictors[ElementIndex]) / get_scale(self.scale[ElementIndex])
		else: # RowIndex < ColIndex
			result.real = call_single_predictor(self.predictors[ElementIndex]) / get_scale(self.scale[ElementIndex])
			result.imag = -call_single_predictor(self.predictors[ColIndex * pes.NUM_PES + RowIndex]) / get_scale(self.scale[ColIndex * pes.NUM_PES + RowIndex])
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
		print('\n', file=f)


def construct_kernel_matrix(predictor: SinglePredictor, x1: npt.NDArray[np.double], x2: npt.NDArray[np.double] | None = None) -> npt.NDArray[np.double]:
	"""
	To construct the kernel (covariance) matrix of the predictor

	Parameters
	----------
	predictor : SinglePredictor
		The predictor, containing information of constructing the covariance matrix
	x1 : npt.NDArray[np.double]
		Features
	x2 : npt.NDArray[np.double] | None, optional
		Another feature, by default None (meaning the same as x1)

	Returns
	-------
	npt.NDArray[np.double]
		Covariance matrix
	"""
	if x2 is None:
		x2 = x1
	if check_predictor(predictor):
		return predictor.model.cov(torch.from_numpy(x1), torch.from_numpy(x2)).detach().numpy()
	else:
		return np.eye(x1.shape[-2], x2.shape[-2], dtype=np.double)
