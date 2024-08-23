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
import numpy as np
import numpy.typing as npt
import sklearn.neighbors
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
	__slots__: tuple = ("__mean", "__cov")

	def __init__(self, x: torch.Tensor, y: torch.Tensor, likelihood: gpytorch.likelihoods.Likelihood, kernel: gpytorch.kernels.Kernel):
		super().__init__(x, y, likelihood)
		self.__mean: gpytorch.means.Mean = gpytorch.means.ZeroMean()
		self.__cov: gpytorch.kernels.Kernel = kernel

	@property
	def cov(self) -> gpytorch.kernels.Kernel:
		return self.__cov

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
		Mean = self.__mean(x)
		assert isinstance(Mean, torch.Tensor)
		return gpytorch.distributions.MultivariateNormal(Mean, self.__cov(x))


class NoConstraint(gpytorch.constraints.Interval):
	"""
	No constraint on the parameter, the value could be any float value

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
		"""
		The official string representation of an object

		Returns
		-------
		str
			The name of the class with an empty parenthesis
		"""
		return __class__.__name__ + "()"

	def transform(self, tensor: torch.Tensor) -> torch.Tensor:
		"""
		To transform the raw value to the actual value

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
		"""
		To transform the actual value to the raw value

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


class SinglePredictor:
	"""
	The instantiation of gaussian process predictor

	Parameters
	----------
	kernel : gpytorch.kernels.Kernel
		The kernel to use
	num_neighbors : int, optional
		The number of neighbors (`k`) searched by kNN for local GP, by default `__NUM_NEIGHBOR`

	Attributes
	----------
	__MAX_ITER : typing.Literal[15000]
		Maximum iteration of optimization
	__FTOL : float
		Absolute and relative tolerance of function in optimization
	__GTOL_SQRT : float
		Tolerance of gradient norm in optimization
	__GTOL : float
		Square of `__GTOL_SQRT`
	__NOISE : float
		Extra noise term added for numerical stability in matrix inversion
	__NUM_NEIGHBOR : typing.Literal[32]
		The default number of neighbors to construct local GP

	Methods
	-------
	__preconditioned_ldl_solve(A, B, single_entry)
		To solve linear system `AX=B` by LDL factorization with pre-conditioned matrix for numerical stability
	__print_stuff(model, loss, learning_rate, indent, print_grad, hessian, start_str, end_str, flush)
		To print all stuffs needed
	get_training_features()
		To get the training features of the core subset
	__update_weights()
		To update the weights, :math:`K^{-1}y`
	__update_neighbor()
		To construct the neighbors and to determine whether use local GP or not
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
	__MAX_ITER = 15000
	__FTOL: float = 2.2204460492503131e-09
	__GTOL_SQRT: float = 1e-5
	__GTOL: float = __GTOL_SQRT ** 2
	__NOISE: float = float(gpytorch.settings.min_fixed_noise.value(torch.double) or 1e-8)
	__NUM_NEIGHBOR = 32
	__slots__: tuple = ("__kernel", "__x_all", "__y_all", "__scale", "__model", "__param_indices", "__k_inv_y", "__weights_updated", "__nn", "__use_local", "__neighbor_updated")

	@staticmethod
	def __preconditioned_ldl_solve(A: torch.Tensor, B: torch.Tensor, single_entry: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
		"""
		To solve linear system `AX=B` by LDL factorization with pre-conditioned matrix for numerical stability

		Parameters
		----------
		A : torch.Tensor, shape of (..., n, n)
			Tensor consisting of symmetric or Hermitian matrices
		B : torch.Tensor, shape of (..., n, k) or (..., n)
			Right-hand side tensor
		single_entry : bool, optional
			Whether B has one dimension less than A or not, by default False

		Returns
		-------
		tuple[torch.Tensor, torch.Tensor], of shape (..., n, k) or (..., n), and (..., n, n)
			Solution and the pre-conditioned `A`
		"""
		cond_mat: torch.Tensor = torch.diag_embed(1.0 / torch.linalg.norm(A, torch.inf, -1).sqrt()) # Q
		A_precond: torch.Tensor = cond_mat @ A @ cond_mat # (QAQ)(Q^{-1}x)=Qb -> x = Q @ solve(QAQ, Qb)
		A_precond = (A_precond + A_precond.mT) / 2.0 # symmetrize
		ld: torch.Tensor
		pivot: torch.Tensor
		ld, pivot, _ = torch.linalg.ldl_factor_ex(A_precond)
		result: torch.Tensor
		if single_entry:
			result = (cond_mat @ torch.linalg.ldl_solve(ld, pivot, cond_mat @ B[..., None]))[..., 0]
		else:
			result = cond_mat @ torch.linalg.ldl_solve(ld, pivot, cond_mat @ B)
		return result, A_precond

	@staticmethod
	def __print_stuff(
		model: gpytorch.models.ExactGP,
		loss: torch.Tensor,
		learning_rate: float,
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
		loss : torch.Tensor
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
		print(("\t" * indent + end_str + "\n") if end_str != "" else "", end="", flush=flush)

	def __init__(
		self,
		kernel: gpytorch.kernels.Kernel,
		initial_values: list[torch.Tensor] = [],
		num_neighbors: int = __NUM_NEIGHBOR
	):
		self.__kernel: gpytorch.kernels.Kernel = copy.deepcopy(kernel)
		likelihood: gpytorch.likelihoods.FixedNoiseGaussianLikelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.full((2,), __class__.__NOISE))
		self.__x_all: torch.Tensor = torch.Tensor()
		self.__y_all: torch.Tensor = torch.Tensor()
		self.__scale: float = 1.0
		self.__model: GP = GP(torch.zeros((2, pes.PHASEDIM)), torch.zeros((2,)), likelihood, self.__kernel)
		self.__param_indices: list[int] = [0]
		with torch.no_grad():
			for iParam, (name, param, constraint) in enumerate(self.__model.named_parameters_and_constraints()):
				# set parameter initial value
				try:
					param[...] = initial_values[iParam].expand_as(param).clone()
				except (IndexError, RuntimeError): # unable to expand, or do not have the parameter
					if isinstance(constraint, gpytorch.constraints.Interval):
						if constraint.initial_value is not None:
							param[...] = constraint.initial_value.expand_as(param).clone()
					else:
						param.fill_(1.0)
				param.detach_().requires_grad = True
				# get indices for each parameter
				self.__param_indices.append(self.__param_indices[-1] + param.numel())
				# set constraint
				name_levels: list[str] = name.split(".")
				attr = self.__model
				for i in range(len(name_levels) - 1):
					attr = getattr(attr, name_levels[i])
				attr.register_constraint(name_levels[-1], NoConstraint())
		self.__k_inv_y: torch.Tensor = torch.Tensor()
		self.__weights_updated: bool = False
		self.__nn: sklearn.neighbors.NearestNeighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=num_neighbors, n_jobs=-1)
		self.__use_local: bool = False
		self.__neighbor_updated: bool = False

	@property
	def model(self) -> GP:
		return self.__model

	def get_training_features(self) -> torch.Tensor:
		"""
		To get the training features of the core subset

		Returns
		-------
		torch.Tensor
			The training features
		"""
		assert self.__model.train_inputs is not None
		return self.__model.train_inputs[0].detach()

	def __update_weights(self) -> None:
		"""
		To update the weights, :math:`K^{-1}y`
		"""
		self.__k_inv_y = torch.linalg.lstsq(self.__model.cov(self.__x_all, self.get_training_features()).to_dense(), self.__y_all).solution
		self.__weights_updated = True

	@property
	def k_inv_y(self) -> torch.Tensor:
		if not self.__weights_updated:
			self.__update_weights()
		return self.__k_inv_y

	def __update_neighbor(self) -> None:
		"""
		To construct the neighbors and to determine whether use local GP or not
		"""
		self.__nn.fit(self.__x_all.detach().numpy())
		direct_err: float = ((self.__model.cov(self.__x_all, self.get_training_features()).to_dense() @ self.k_inv_y - self.__y_all) ** 2).sum().item()
		neighbor_ind: torch.Tensor = torch.from_numpy(self.__nn.kneighbors(self.__x_all.detach().numpy().reshape(-1, pes.PHASEDIM), return_distance=False)).reshape(self.__x_all.shape[:-1] + (__class__.__NUM_NEIGHBOR,)) # ... * DIM -> ... * NEIGHBOR
		neighbors: torch.Tensor = self.__x_all[neighbor_ind] # x_all must be M * DIM, this gives ... * NEIGHBOR * DIM
		local_err: float = (((self.__model.cov(self.__x_all[..., None, :], neighbors).to_dense() @ __class__.__preconditioned_ldl_solve(self.__model.cov(neighbors).to_dense(), self.__y_all[neighbor_ind, None])[0]).reshape(self.__x_all.shape[:-1]) - self.__y_all) ** 2).sum().item()
		self.__use_local = local_err < direct_err
		self.__neighbor_updated = True

	@property
	def use_local(self) -> bool:
		if not self.__neighbor_updated:
			self.__update_neighbor()
		return self.__use_local

	def predict(
		self,
		x_test: torch.Tensor,
		use_local_GP: bool = True
	) -> torch.Tensor:
		"""
		To predict the average estimation

		Parameters
		----------
		x_test : torch.Tensor, shape of (N, PHASEDIM)
			Validation/Test inputs
		use_local_GP : bool, optional
			Whether use local full GP on selected points or not, by default True

		Returns
		-------
		torch.Tensor, shape of (N,)
			Corresponding validation/test targets based on noise-free SR/PP mean.

		Notes
		-----
		Instance of prediction of projected process (PP)
		"""
		# if use_local_GP and self.use_local:
		# 	neighbor_ind: torch.Tensor = torch.from_numpy(self.__nn.kneighbors(x_test.detach().numpy().reshape(-1, pes.PHASEDIM), return_distance=False)).reshape(x_test.shape[:-1] + (__class__.__NUM_NEIGHBOR,)) # ... * DIM -> ... * NEIGHBOR
		# 	neighbors: torch.Tensor = self.__x_all[neighbor_ind] # x_all must be M * DIM, this gives ... * NEIGHBOR * DIM
		# 	return (self.__model.cov(x_test[..., None, :], neighbors).to_dense() @ __class__.__preconditioned_ldl_solve(self.__model.cov(neighbors).to_dense(), self.__y_all[neighbor_ind, None])[0]).reshape(x_test.shape[:-1])
		# 	"""
		# 		... * 1 * DIM @ ... * NEIGHBOR * DIM -> ... * 1 * NEIGHBOR
		# 		... * NEIGHBOR * DIM -> ... * NEIGHBOR * NEIGHBOR
		# 		[... * NEIGHBOR, 1] -> ... * NEIGHBOR * 1
		# 		-> ... * 1 * 1 -> ...
		# 	"""
		# else:
		# 	return self.__model.cov(x_test, self.get_training_features()).to_dense() @ self.k_inv_y
		return self.__model.cov(x_test, self.get_training_features()).to_dense() @ self.k_inv_y

	def error(self, use_weight: bool = True) -> torch.Tensor:
		"""
		Error function of projected process (PP)

		This function gives the sum of squared error

		Parameters
		----------
		use_weight : bool, optional
			Whether to use precomputed :math:`K^{-1}y` (gradient unavailable) or not, by default True

		Returns
		-------
		torch.Tensor
			The sum of squared prediction error
		"""
		if use_weight:
			return torch.sum((self.__y_all - self.predict(self.__x_all)) ** 2) * (self.__scale ** 2)
		else:
			kmn: torch.Tensor = self.__model.cov(self.__x_all, self.get_training_features()).to_dense()
			return torch.sum((self.__y_all - kmn @ torch.linalg.lstsq(kmn, self.__y_all).solution) ** 2) * (self.__scale ** 2)

	def train(self, print_log: bool = DEBUG_MODE) -> None:
		"""
		To train the parameters

		Parameters
		----------
		print_log : bool, optional
			Whether to print the log to console, by default `DEBUG_MODE`
		"""
		def optimize_with_learning_rate(
			model: gpytorch.models.ExactGP,
			combined_parameters: torch.Tensor,
			change: torch.Tensor,
			param_indices: list[int],
			last_loss: float,
			method_name: str,
			initial_learning_rate: float = 1.0
		) -> tuple[torch.Tensor, float]:
			"""
			To change the parameter with adjustable learning rate

			Parameters
			----------
			model : gpytorch.models.ExactGP
				Gaussian process model, containing mean and covariances and their parameters
			combined_parameters : torch.Tensor, shape of (N,)
				Initial values of parameters
			change : torch.Tensor, shape of (N,)
				The changing quantity of the parameters, e.g., combined gradient of parameters
			param_indices : list[int], len of n+1
				The starting index of each parameters, staring from 0 and end at N+1
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
					for iParam, param in enumerate(model.parameters()):
						param[...] = target[param_indices[iParam]:param_indices[iParam + 1]].reshape_as(param).detach()

			def return_back() -> tuple[torch.Tensor, float]:
				"""
				To turn back to the status when the function is called

				Returns
				-------
				tuple[torch.Tensor, float]
					`last_loss` in `torch.Tensor` form, and `initial_learning_rate`
				"""
				if print_log:
					print("\tNo stepping Forward for {}.".format(method_name))
				with torch.no_grad():
					for iParam, param in enumerate(model.parameters()):
						param[...] = combined_parameters[param_indices[iParam]:param_indices[iParam + 1]].reshape_as(param).detach()
				return torch.tensor(last_loss), initial_learning_rate

			learning_rate: float = initial_learning_rate
			change_param(combined_parameters - learning_rate * change)
			learning_rate_threshold: float = sys.float_info.epsilon * (combined_parameters.norm() / change.norm()).item() # lr * |change| should > double precision * |param|
			loss: torch.Tensor
			while learning_rate >= learning_rate_threshold:
				try:
					loss = self.error(False)
					break
				except RuntimeError: # NANs
					learning_rate /= 2.0
			if learning_rate < learning_rate_threshold:
				# based on current learning rate, all parameters have changes smaller than double precision
				return return_back()
			if print_log:
				__class__.__print_stuff(model, loss, learning_rate, 2, start_str="last = {:.15e},".format(last_loss), flush=True)
			while loss.isnan().item() or loss.item() >= last_loss:
				learning_rate /= 2.0
				change_param(combined_parameters - learning_rate * change)
				try:
					loss = self.error(False)
				except RuntimeError: # NANs
					return return_back()
				if print_log:
					__class__.__print_stuff(model, loss, learning_rate, 2, start_str="last = {:.15e},".format(last_value), flush=True)
				if learning_rate < learning_rate_threshold:
					# based on current learning rate, all parameters being the same
					if loss.item() >= last_loss:
						return return_back()
					else:
						if print_log:
							print("\tNo stepping Forward for {}.".format(method_name))
						break
			return loss, learning_rate

		self.__weights_updated = False
		# train model
		self.__model.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.full(self.get_training_features().shape[:-1], __class__.__NOISE))
		self.__model.train()
		self.__model.likelihood.train()
		finish_early: bool = False
		loss: torch.Tensor = self.error(False)
		last_value: float = torch.inf
		learning_rate: float = -1.0
		hessian: torch.Tensor = torch.eye(self.__param_indices[-1], dtype=torch.double)
		hessian_precond: torch.Tensor
		num_iter: int = 0
		for i in range(1, __class__.__MAX_ITER + 1):
			# calculate gradient and hessian
			for iParam, param in enumerate(self.__model.parameters()):
				param.grad = torch.autograd.grad(loss, param, None, True, True, True, True, False, True)[0] # same shape of param
				for iGrad, grad_elm in enumerate(param.grad.reshape(-1)):
					for jParam, param_for_grad in enumerate(self.__model.parameters()):
						hessian[self.__param_indices[iParam] + iGrad, self.__param_indices[jParam]:self.__param_indices[jParam + 1]] = torch.autograd.grad(grad_elm, param_for_grad, None, True, False, True, True, False, True)[0]
			hessian = (hessian + hessian.T) / 2.0 # symmetrize
			grad_combined: torch.Tensor = torch.cat([(param.grad if param.grad is not None else torch.zeros_like(param)).reshape(-1) for param in self.__model.parameters()])
			param_combined: torch.Tensor = torch.cat([param.reshape(-1) for param in self.__model.parameters()])
			if learning_rate == -1.0:
				learning_rate = min((param_combined.norm() / grad_combined.norm()).item(), 1.0)
			# stopping criteria
			grad_sqnm: float = torch.sum(grad_combined ** 2).item()
			if grad_sqnm < __class__.__GTOL:
				finish_early = True
				num_iter = i
				print("Convergence: |Gradient| = {} <= GTOL = {}".format(math.sqrt(grad_sqnm), __class__.__GTOL_SQRT))
				break
			if abs(last_value - loss.item()) / max(abs(last_value), abs(loss.item()), 1.0) < __class__.__FTOL:
				finish_early = True
				num_iter = i
				print("Convergence: |f_i - f_{{i+1}}| = {} / {} <= FTOL = {}".format(abs(last_value - loss.item()), max(abs(last_value), abs(loss.item()), 1.0), __class__.__FTOL))
				break
			# change parameter
			last_value = loss.item()
			newton_change: torch.Tensor
			newton_change, hessian_precond = __class__.__preconditioned_ldl_solve(hessian, grad_combined, True)
			if learning_rate < 1.0:
				learning_rate *= 2.0
			# log
			if print_log:
				__class__.__print_stuff(
					self.__model,
					loss,
					learning_rate,
					1,
					True,
					hessian,
					"Iter {} - last = {:.15e},".format(i, last_value),
					"Cond(preconditioned hessian) = {}".format(torch.linalg.cond(hessian_precond).item()),
					True
				)
			else:
				if i % (__class__.__MAX_ITER // 100) == 0:
					__class__.__print_stuff(
						self.__model,
						loss,
						learning_rate,
						1,
						True,
						hessian,
						"Iter {} - last = {:.15e},".format(i, last_value),
						"Cond(preconditioned hessian) = {}".format(torch.linalg.cond(hessian_precond).item())
					)
			old_learning_rate: float = learning_rate
			for change, method in zip([newton_change, grad_combined], ["Newton Method", "Gradient Descent"]):
				loss, learning_rate = optimize_with_learning_rate(
					self.__model,
					param_combined,
					change,
					self.__param_indices,
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
		if not finish_early:
			print("Stop: Total No. iterations reached limit.")
			num_iter = __class__.__MAX_ITER
		__class__.__print_stuff(
			self.__model,
			loss,
			learning_rate,
			0,
			True,
			hessian,
			"Iter {} - last = {:.15e},".format(num_iter, last_value),
			"",
			print_log
		)
		# turn to predict mode
		self.__model.eval()
		self.__model.likelihood.eval()

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
		self.__x_all = x_all.reshape(-1, pes.PHASEDIM).clone().detach()
		self.__y_all = y_all.reshape(-1).clone().detach()
		self.__scale = scale
		self.__model.set_train_data(self.__x_all[:num_points].detach(), self.__y_all[:num_points].detach(), False)
		self.__weights_updated = False
		self.__neighbor_updated = False

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
		marginal_kernel: gpytorch.kernels.Kernel = gpytorch.kernels.RBFKernel(len(dimensions), lengthscale_constraint=NoConstraint(self.__model.cov.lengthscale.reshape(1, pes.PHASEDIM)[:, dimensions]))
		prefactor: float = np.sqrt((2.0 * torch.pi) ** (pes.PHASEDIM - len(dimensions))) * self.__model.cov.lengthscale[:, [i for i in range(pes.PHASEDIM) if i not in dimensions]].prod().item()
		return prefactor * marginal_kernel(x_test, self.get_training_features()[:, dimensions]).to_dense() @ self.k_inv_y


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
		initial_value: torch.Tensor | None = None
		if parameter_initial_values is not None and parameter_initial_values.size == pes.PHASEDIM:
			initial_value = torch.from_numpy(parameter_initial_values.reshape(-1))
		self.__predictors: list[SinglePredictor] = [SinglePredictor(gpytorch.kernels.RBFKernel(pes.PHASEDIM, lengthscale_constraint=NoConstraint(initial_value))) for _ in range(pes.NUM_ELM)]
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
		self.__scale[...] = scale
		for iElement, pred in enumerate(self.__predictors):
			RowIndex: int = iElement // pes.NUM_PES
			ColIndex: int = iElement % pes.NUM_PES
			TrilIndex: int = pes.flatten_tril_index[RowIndex, ColIndex]
			pred.update(
				torch.from_numpy(x_all[TrilIndex]),
				torch.from_numpy(y_all[TrilIndex].real if RowIndex <= ColIndex else y_all[TrilIndex].imag),
				self.__scale[iElement],
				num_points[TrilIndex]
			)

	def train(self, print_log: bool = DEBUG_MODE) -> None:
		"""
		To train each predictor

		Parameters
		----------
		print_log : bool, optional
			Whether to print the log to console, by default `DEBUG_MODE`
		"""
		for iElement in range(pes.NUM_ELM):
			if __class__.__check_predictor(self.__predictors[iElement]):
				if print_log:
					print("Training " + utility.get_RI_label(iElement))
				self.__predictors[iElement].train(print_log)

	def predict(self, x_input: npt.NDArray[np.double], ElementIndex: int, use_local_gp: bool = True) -> npt.NDArray[np.cdouble]:
		"""
		To predict test targets based on input and corresponding density matrix element

		Parameters
		----------
		x_input : npt.NDArray[np.double], shape of (..., PHASEDIM)
			Test inputs
		ElementIndex : int
			Index of the element
		use_local_GP : bool, optional
			Whether use local full GP on selected points or not, by default True

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
				return pred.predict(x_test, use_local_gp).detach().numpy()
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
