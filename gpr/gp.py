r"""gp
==
Implementation for gaussian process (gp) regression.
"""
import collections.abc
import math
import typing

import linear_operator
import numpy as np
import torch

import constant
import expectation
import pes

torch.set_default_dtype(constant.DTYPE)
torch.set_default_device(constant.DEVICE)


def _distance_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
	r"""To compute the pairwise distance matrix between two sets of vectors

	Parameters
	----------
	x : torch.Tensor
		The first set of vectors
	y : torch.Tensor
		The second set of vectors

	Returns
	-------
	torch.Tensor
		The pairwise distance matrix
	"""
	return torch.linalg.norm(x[:, torch.newaxis] - y[torch.newaxis], dim=-1)


@typing.final
class _KNNResult(typing.NamedTuple):
	indices: torch.Tensor
	distances: torch.Tensor
	points: torch.Tensor


def _knn(train_points: torch.Tensor, k: int, test_points: torch.Tensor | None = None) -> _KNNResult:
	r"""To find the k-nearest neighbors of a set of points

	Author: Josue N Rivera (github.com/wzjoriv)

	Date: 7/3/2021

	Description: Snippet of various clustering implementations only using PyTorch

	Full project repository: https://github.com/wzjoriv/Lign (A graph deep learning framework that works alongside PyTorch)

	Parameters
	----------
	train_points : torch.Tensor
		The training points
	k : int
		The number of neighbors to consider
	test_points : torch.Tensor | None, optional, shape of (N, D)
		The points to predict the labels for, by default None (and uses the training points)

	Returns
	-------
	_KNNResult
		The k-nearest neighbors result, including indices (shape of (N, k)),
		distances (shape of (N, k)), and points (shape of (N, k, D))
	"""
	assert train_points.ndim == 2
	if test_points is not None:
		assert test_points.ndim == 2
	else: # test_points is None
		test_points = train_points
	topk: typing.Final = _distance_matrix(test_points, train_points).topk(k, -1, False, False)
	return _KNNResult(indices=topk.indices, distances=topk.values, points=train_points[topk.indices])


def _cov(lengthscale: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
	r"""To calculate the covariance matrix using RBF kernel

	Parameters
	----------
	lengthscale : torch.Tensor
		The characteristic lengths
	x1 : torch.Tensor
		The first feature set
	x2 : torch.Tensor
		The second feature set

	Returns
	-------
	torch.Tensor
		The covariance matrix
	"""
	return torch.exp(-torch.square((x1[:, torch.newaxis] - x2) / lengthscale).sum(-1))


def _raw_lengthscale_to_value(raw_lengthscale: torch.Tensor) -> torch.Tensor:
	r"""To calculate the explicit lengthscale

	Parameters
	----------
	raw_lengthscale : torch.Tensor
		Raw lengthscale in log form

	Returns
	-------
	torch.Tensor
		Exponential of raw lengthscale
	"""
	return torch.exp(raw_lengthscale)


def _lengthscale_to_raw_value(lengthscale: torch.Tensor) -> torch.Tensor:
	r"""To calculate the underlying lengthscale

	Parameters
	----------
	lengthscale : torch.Tensor
		Explicit lengthscale

	Returns
	-------
	torch.Tensor
		Logarithm of lengthscale
	"""
	return torch.log(lengthscale)


@typing.final
class SinglePredictor:
	r"""The instantiation of gaussian process predictor

	Parameters
	----------
	x_all : torch.Tensor, of shape (N_ALL_PT, PHASEDIM)
		All training inputs
	y_all : torch.Tensor, of shape (N_ALL_PT)
		All training targets
	num_points : int
		The number of points located at the front of all points that is used as the subset
	kernel_initial_value : torch.Tensor | None
		Initial value of kernel

	Methods
	-------
	x_all()
		All features for approximate method
	get_training_features()
		To get the training features of the core subset
	k_inv_y()
		To get the weights, :math:`K^{-1}y`
	predict(x_test)
		To predict the average
	error()
		To get the error by comparing label with prediction
	"""
	@typing.final
	class InputDerivativeReturn(typing.NamedTuple):
		predict: torch.Tensor
		derivative: torch.Tensor

	@typing.final
	class InternalDerivativeReturn(typing.NamedTuple):
		feature_derivative: torch.Tensor
		label_derivative: torch.Tensor
		raw_param_derivative: torch.Tensor

	__NOISE: typing.Final[float] = 1e-6
	__K_NEIGHBORHOOD: typing.Final[int] = 10
	# __slots__: typing.Final[tuple] = ("__x_all", "__y_all", "__num_pts", "__raw_lengthscale", "__k_inv_y", "__weights_updated")
	__x_all: torch.Tensor
	__y_all: torch.Tensor
	__num_pts: int
	__raw_lengthscale: torch.Tensor
	__old_lengthscale: torch.Tensor
	__k_inv_y: torch.Tensor

	def __init__(
		self,
		x_all: torch.Tensor,
		y_all: torch.Tensor,
		num_points: int,
		kernel_initial_value: torch.Tensor
	):
		PHASEDIM: typing.Final[int] = x_all.shape[-1]
		self.__x_all = x_all.reshape(-1, PHASEDIM).clone().detach()
		self.__y_all = y_all.reshape(-1).clone().detach()
		self.__num_pts = num_points
		nearest_distances: typing.Final[torch.Tensor] = _knn(self.__x_all[:self.__num_pts], self.__K_NEIGHBORHOOD).distances.mean(-1)
		noise: typing.Final[torch.Tensor] = SinglePredictor.__NOISE * torch.eye(self.__num_pts) + self.__NOISE * torch.diag(torch.square(nearest_distances / nearest_distances.median()))
		self.__raw_lengthscale = torch.log(torch.repeat_interleave(kernel_initial_value.reshape(-1), PHASEDIM // kernel_initial_value.numel()).detach()) # true length = exp(raw_length), and raw range in (-inf, inf)
		self.__old_lengthscale = self.lengthscale
		self.__k_inv_y = linear_operator.utils.stable_pinverse(_cov(self.lengthscale, self.__x_all, self.__x_all[:self.__num_pts]) + torch.cat((noise, torch.zeros(self.__x_all.shape[0] - self.__num_pts, self.__num_pts)))) @ self.__y_all

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
		return self.__x_all[:self.__num_pts]

	@property
	def k_inv_y(self) -> torch.Tensor:
		r"""To get the weights, :math:`K^{-1}y`

		Returns
		-------
		torch.Tensor
			Weights, :math:`K^{-1}y`
		"""
		return self.__k_inv_y
	
	@property
	def lengthscale(self) -> torch.Tensor:
		r"""To get the lengthscale

		Returns
		-------
		torch.Tensor
			The lengthscale
		"""
		return _raw_lengthscale_to_value(self.__raw_lengthscale)

	@property
	def old_lengthscale(self) -> torch.Tensor:
		r"""To get the old lengthscale

		Returns
		-------
		torch.Tensor
			Old lengthscale
		"""
		return self.__old_lengthscale

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
		return _cov(self.lengthscale, x_test, self.__x_all[:self.__num_pts]) @ self.k_inv_y

	def error(self) -> float:
		r"""Error function of subset of regressor (SR) / projected process (PP)

		This function gives the sum of squared error

		Returns
		-------
		float
			The sum of squared prediction error
		"""
		return torch.sum((self.__y_all - self.predict(self.__x_all)) ** 2).item()

	def predict_derivative_over_input(self, x_test: torch.Tensor) -> InputDerivativeReturn:
		r"""To give the derivative of prediction over the input

		Parameters
		----------
		x_test : torch.Tensor, shape of (N, PHASEDIM)
			Validation/Test inputs

		Returns
		-------
		__InputDerivativeReturn
			Prediction (shape of (N,)), and its derivative over input (shape of (N, PHASEDIM))
		"""
		with torch.no_grad():
			x_test = x_test.reshape(-1, x_test.shape[-1]).detach().requires_grad_()
		predict: typing.Final[torch.Tensor] = self.predict(x_test)
		return SinglePredictor.InputDerivativeReturn(predict=predict.detach(), derivative=torch.autograd.grad(predict, x_test, torch.ones_like(predict), False, False, True, True, False, True)[0].detach())

	def predict_derivative_over_internal(self, x_test: torch.Tensor) -> InternalDerivativeReturn:
		r"""To calculate the derivative of prediction over all related quantities

		Parameters
		----------
		x_test : torch.Tensor, shape of (N, PHASEDIM)
			Validation/Test inputs

		Returns
		-------
		__AllDerivativeReturn
			Prediction (shape of (N,)),
			derivative over each `x_test` (shape of (N, PHASEDIM)),
			derivative over training feature (shape of (N, M, PHASEDIM)),
			derivative over training label (shape of (N, M)),
			and derivative over raw characteristic lengthscale (shape of (N, PHASEDIM))
		"""
		with torch.no_grad():
			x_all: typing.Final[torch.Tensor] = self.__x_all.detach().requires_grad_()
			y_all: typing.Final[torch.Tensor] = self.__y_all.detach().requires_grad_()
			raw_lengthscale: typing.Final[torch.Tensor] = self.__raw_lengthscale.detach().requires_grad_()
		nearest_distances: typing.Final[torch.Tensor] = _knn(self.__x_all[:self.__num_pts], self.__K_NEIGHBORHOOD).distances.mean(-1)
		noise: typing.Final[torch.Tensor] = SinglePredictor.__NOISE * torch.eye(self.__num_pts) + self.__NOISE * torch.diag(torch.square(nearest_distances / nearest_distances.median())) # independent of derivative
		predict: typing.Final[torch.Tensor] = _cov(_raw_lengthscale_to_value(raw_lengthscale), x_test, x_all[:self.__num_pts]) @ linear_operator.utils.stable_pinverse(_cov(_raw_lengthscale_to_value(raw_lengthscale), x_all, x_all[:self.__num_pts]) + torch.cat((noise, torch.zeros(self.__x_all.shape[0] - self.__num_pts, self.__num_pts)))) @ y_all
		return SinglePredictor.InternalDerivativeReturn(
			feature_derivative=torch.autograd.grad(predict, x_all, torch.eye(predict.numel()), False, False, True, True, True, True)[0].detach().reshape(predict.shape + x_all.shape),
			label_derivative=torch.autograd.grad(predict, y_all, torch.eye(predict.numel()), False, False, True, True, True, True)[0].detach().reshape(predict.shape + y_all.shape),
			raw_param_derivative=torch.autograd.grad(predict, raw_lengthscale, torch.eye(predict.numel()), False, False, True, True, True, True)[0].detach().reshape(predict.shape + raw_lengthscale.shape)
		)

	def update_param(self, new_lengthscale: torch.Tensor) -> None:
		r"""To update the parameters

		Parameters
		----------
		new_lengthscale : torch.Tensor
			The updated parameter
		"""
		self.__old_lengthscale = self.lengthscale
		self.__raw_lengthscale = _lengthscale_to_raw_value(new_lengthscale)

	def update(
		self,
		x_all: torch.Tensor,
		y_all: torch.Tensor,
		num_points: int
	) -> None:
		r"""To update the training features and labels of the model

		Parameters
		----------
		x_all : torch.Tensor, of shape (N_ALL_PT, PHASEDIM)
			All training inputs
		y_all : torch.Tensor, of shape (N_ALL_PT)
			All training targets
		num_points : int
			The number of points located at the front of all points that is used as the subset
		"""
		self.__x_all = x_all.reshape(-1, x_all.shape[-1]).clone().detach()
		nearest_distances: typing.Final[torch.Tensor] = _knn(self.__x_all[:self.__num_pts], self.__K_NEIGHBORHOOD).distances.mean(-1)
		noise: typing.Final[torch.Tensor] = self.__NOISE * torch.eye(self.__num_pts) + self.__NOISE * torch.diag(torch.square(nearest_distances / nearest_distances.median()))
		self.__y_all = y_all.reshape(-1).clone().detach()
		self.__num_pts = num_points
		self.__k_inv_y = linear_operator.utils.stable_pinverse(_cov(self.lengthscale, self.__x_all, self.__x_all[:num_points]) + torch.cat((noise, torch.zeros(self.__x_all.shape[0] - self.__num_pts, self.__num_pts)))) @ self.__y_all

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
		prefactor: typing.Final[float] = math.sqrt((2.0 * torch.pi) ** (phasedim - len(dimensions))) * self.lengthscale[:, [i for i in range(phasedim) if i not in dimensions]].prod().item()
		return prefactor * _cov(self.lengthscale[dimensions], x_test, self.__x_all[:self.__num_pts, dimensions]) @ self.k_inv_y


@typing.final
class GPRPredictors:
	r"""Combination of single predictors

	Parameters
	----------
	kernel : gpytorch.kernels.Kernel, optional
		The kernel of predictors, by default gpytorch.kernels.RBFKernel(pes.PHASEDIM)

	Methods
	-------
	update(x_all, y_all, num_pt, scale)
		To update the training inputs and targets, as well as the rescale factor
	predict(x_input, ElementIndex)
		To predict test targets based on input and corresponding density matrix element
	print(f)
		To print hyperparameters to file
	"""

	__slots__: typing.Final[tuple] = ("__config", "__predictors",)
	__config: typing.Final[pes.ModelConfig]
	__predictors: typing.Final[list[SinglePredictor]]

	def __init__(
		self,
		config: pes.ModelConfig,
		x_all: list[torch.Tensor],
		y_all: list[torch.Tensor],
		num_points: int | list[int],
		kernel_initial_value: torch.Tensor
	):
		self.__config = config
		self.__predictors = []
		if isinstance(num_points, int):
			num_points = [num_points] * self.__config.NUM_TRIG
		for iElement in self.__config.ELEMENT_RANGE:
			RowIndex: int = iElement // self.__config.NUM_PES
			ColIndex: int = iElement % self.__config.NUM_PES
			TrilIndex: int = self.__config.FLATTEN_TRIL_INDEX[iElement]
			self.__predictors.append(SinglePredictor(
				x_all[TrilIndex],
				y_all[TrilIndex].real if RowIndex <= ColIndex else y_all[TrilIndex].imag,
				num_points[TrilIndex],
				kernel_initial_value
			))

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
		num_points: int | list[int],
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
		"""
		if isinstance(num_points, int):
			num_points = [num_points] * self.__config.NUM_TRIG
		for iElement, pred in enumerate(self.__predictors):
			RowIndex: int = iElement // self.__config.NUM_PES
			ColIndex: int = iElement % self.__config.NUM_PES
			TrilIndex: int = self.__config.FLATTEN_TRIL_INDEX[iElement]
			pred.update(
				x_all[TrilIndex],
				y_all[TrilIndex].real if RowIndex <= ColIndex else y_all[TrilIndex].imag,
				num_points[TrilIndex]
			)

	def __combine_to_complex[**P, T: collections.abc.Iterable[torch.Tensor]](
		self,
		x_input: torch.Tensor,
		RowIndex: int,
		ColIndex: int,
		call_single_predictor: collections.abc.Callable[typing.Concatenate[SinglePredictor, torch.Tensor, P], T],
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
			lambda pred, x_test: (pred.predict(x_test),)
		)[0]

	def evolve_parameter(
		self,
		integral_pts: torch.Tensor,
		weights: torch.Tensor,
		model: pes.Potential,
		mass: torch.Tensor,
		dt: float
	) -> None:
		r"""To evolve the parameters of predictors of each element

		Parameters
		----------
		integral_pts : torch.Tensor, shape of (NUM_TRIG, N, PHASEDIM)
			The points for monte carlo integration
		weights : torch.Tensor, shape of (NUM_TRIG, N)
			The weights of the monte carlo points
		model : pes.Potential
			Quantities derived from potential
		mass : torch.Tensor, shape of (DIM,)
			Mass of classical degree of freedom
		dt : float
			Time interval
		"""
		def coord_and_time_derivative(x: torch.Tensor, RowIndex: int, ColIndex: int) -> tuple[torch.Tensor, torch.Tensor]:
			r"""To calculate dX/dt and drho_ij/dt

			Parameters
			----------
			x : torch.Tensor, shape of (..., PHASEDIM)
				Phase space coordinates
			RowIndex : int
				Index of row of the element in density matrix
			ColIndex : int
				Index of column of the element in density matrix

			Returns
			-------
			tuple[torch.Tensor, torch.Tensor]
				dX/dt, shape of (..., PHASEDIM) and drho_ij/dt, shape of (...)
			"""
			predict_derivative_over_input: typing.Final[collections.abc.Callable[[torch.Tensor, int, int], SinglePredictor.InputDerivativeReturn]] = lambda x_input, RowIndex, ColIndex: SinglePredictor.InputDerivativeReturn(*self.__combine_to_complex(
				x_input,
				RowIndex,
				ColIndex,
				lambda pred, x_test: pred.predict_derivative_over_input(x_test)
			))
			r: torch.Tensor = x[..., :self.__config.DIM] # position coordinates
			v: torch.Tensor = x[..., self.__config.DIM:] / mass # velocity
			E, D, F = model.adiabatic_potential_coupling_and_force(r)
			# dGPR/dt at x_test
			phase_deriv: torch.Tensor = torch.cat((v, (F[..., RowIndex, RowIndex] + F[..., ColIndex, ColIndex]) / 2.0), -1) # same shape as x
			pred, grad_input = predict_derivative_over_input(x, RowIndex, ColIndex)
			# drho/dt at x_test
			time_deriv: torch.Tensor = -(phase_deriv * grad_input).sum(-1)
			if RowIndex != ColIndex:
				time_deriv -= 1.0j / constant.HBAR * (E[..., RowIndex] - E[..., ColIndex]) * pred
			for iPES in self.__config.PES_RANGE:
				if iPES != RowIndex:
					pred_kj, grad_kj = predict_derivative_over_input(x, iPES, ColIndex)
					time_deriv -= (D[..., RowIndex, iPES] * (v * pred_kj[..., np.newaxis] + (E[..., RowIndex] - E[..., iPES])[..., np.newaxis] / 2.0 * grad_kj[..., self.__config.DIM:])).sum(-1)
				if iPES != ColIndex:
					pred_ik, grad_ik = predict_derivative_over_input(x, RowIndex, iPES)
					time_deriv += (D[..., iPES, ColIndex] * (v * pred_ik[..., np.newaxis] + (E[..., ColIndex] - E[..., iPES])[..., np.newaxis] / 2.0 * grad_ik[..., self.__config.DIM:])).sum(-1)
			return phase_deriv, time_deriv

		def evolve(
			weight: torch.Tensor,
			test_den_time_deriv: torch.Tensor,
			grad_feature: torch.Tensor,
			grad_label: torch.Tensor,
			grad_param: torch.Tensor,
			feature_time_deriv: torch.Tensor,
			label_time_deriv: torch.Tensor,
			param_old: torch.Tensor,
			param_now: torch.Tensor
		) -> torch.Tensor:
			r"""To evolve the parameter by the given information

			Parameters
			----------
			weight : torch.Tensor
				The monte carlo weights of the points
			test_den_time_deriv : torch.Tensor
				Time derivative of the density at the monte carlo points
			grad_feature : torch.Tensor
				Derivative of prediction at the monte carlo points over training features
			grad_label : torch.Tensor
				Derivative of prediction at the monte carlo points over training labels
			grad_param : torch.Tensor
				Derivative of prediction at the monte carlo points over hyperparameters
			feature_time_deriv : torch.Tensor
				Derivative of training feature over time
			label_time_deriv : torch.Tensor
				Derivative of training label over time
			param_old : torch.Tensor
				Hyperparameters from last time step
			param_now : torch.Tensor
				Hyperparameters from this time step

			Returns
			-------
			torch.Tensor
				Hyperparameters for the next time step
			"""
			param_ref_epsilon: typing.Final = 0.01 # param_ref(t)=(1-epsilon)*param(t-dt)+epsilon*param(t)
			damp_c: typing.Final = 1.0
			damp_epsilon: typing.Final = 1e-6 # gamma(t)=c*||d(param)/dt||/(||param(t)-param_ref(t)||+epsilon*||param(t)||)
			residual: typing.Final[torch.Tensor] = test_den_time_deriv - grad_feature.reshape(grad_feature.shape[0], -1) @ feature_time_deriv.reshape(-1) - grad_label @ label_time_deriv
			time_depend: typing.Final[torch.Tensor] = torch.linalg.ldl_solve(*torch.linalg.ldl_factor((grad_param.T * grad_param.T[:, torch.newaxis] / weight).mean(-1)), (grad_param.T * residual / weight).mean(-1)) # it times dt gives Euler
			param_ref: typing.Final[torch.Tensor] = param_old + param_ref_epsilon * (param_now - param_old)
			param_ref_diff: typing.Final[torch.Tensor] = param_now - param_ref
			damp_coe: typing.Final[float] = damp_c * time_depend.norm().item() / (param_ref_diff.norm().item() + damp_epsilon * param_now.norm().item())
			damp_coe_exp: typing.Final[float] = math.exp(-damp_coe * dt)
			return param_ref + param_ref_diff * damp_coe_exp + (1 - damp_coe_exp) / damp_coe * time_depend

		for iTrig, (iPES, jPES, iElement) in enumerate(zip(self.__config.TRIL_ROW_INDICES, self.__config.TRIL_COL_INDICES, self.__config.TRIL_ELEMENT_INDICES)):
			test_pts: torch.Tensor = integral_pts[iTrig]
			test_wt: torch.Tensor = weights[iTrig]
			feature_time_deriv, label_time_deriv = coord_and_time_derivative(self[iElement].x_all, iPES, jPES)
			test_den_time_deriv: torch.Tensor = coord_and_time_derivative(test_pts, iPES, jPES)[1]
			# deal with real and imag part separately
			real_pred: SinglePredictor
			imag_pred: SinglePredictor | None
			if iPES == jPES:
				real_pred = self[iElement]
			else:
				real_pred = self[jPES * self.__config.NUM_PES + iPES]
				self[iElement].update_param(evolve(test_wt, test_den_time_deriv.imag, *self[iElement].predict_derivative_over_internal(test_pts), feature_time_deriv, label_time_deriv.imag, self[iElement].old_lengthscale, self[iElement].lengthscale))
			real_pred.update_param(evolve(test_wt, test_den_time_deriv.real, *real_pred.predict_derivative_over_internal(test_pts), feature_time_deriv, label_time_deriv.real, real_pred.old_lengthscale, real_pred.lengthscale))

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
			return (pred.get_marginal(dims, x_test),)

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
			np.savetxt(f, predictor.lengthscale.detach().cpu().numpy().reshape(1, -1))
		print("\n", file=f)


@typing.final
class AnalyticalAverager(expectation.Averager):
	r"""Using analytical integral of GPR to estimate averages

	Parameters
	----------
	config : pes.ModelConfig
		Configuration of the model
	pred : GPRPredictors
		GPR predictors
	"""
	__slots__: tuple = ("__AVERAGE_CONSTANT", "__predictors",)
	__AVERAGE_CONSTANT: typing.Final[float]
	__predictors: typing.Final[GPRPredictors]

	def __init__(self, config: pes.ModelConfig, pred: GPRPredictors):
		super().__init__(config)
		self.__AVERAGE_CONSTANT = (2.0 * math.pi) ** self.config.DIM
		self.__predictors = pred

	def population(self) -> torch.Tensor:
		result: torch.Tensor = torch.empty(self.config.NUM_PES)
		for iPES in range(self.config.NUM_PES):
			ElementIndex: int = iPES * self.config.NUM_PES + iPES
			pred: SinglePredictor = self.__predictors[ElementIndex]
			result[iPES] = pred.lengthscale.prod().item() * pred.k_inv_y.sum().item()
		return result * self.__AVERAGE_CONSTANT

	def coordinates(self) -> torch.Tensor:
		result: torch.Tensor = torch.zeros(self.config.PHASEDIM)
		for iPES in range(self.config.NUM_PES):
			ElementIndex: int = iPES * self.config.NUM_PES + iPES
			pred: SinglePredictor = self.__predictors[ElementIndex]
			result += pred.lengthscale.prod().item() * (pred.k_inv_y[:, None] * pred.get_training_features()).sum(0)
		return result * self.__AVERAGE_CONSTANT

	def square_coordinates(self) -> torch.Tensor:
		result: torch.Tensor = torch.zeros((self.config.PHASEDIM, self.config.PHASEDIM))
		for iPES in range(self.config.NUM_PES):
			ElementIndex: int = iPES * self.config.NUM_PES + iPES
			pred: SinglePredictor = self.__predictors[ElementIndex]
			result += pred.lengthscale.prod().item() * (
				(pred.k_inv_y[:, None, None] * pred.get_training_features()[:, :, None] * pred.get_training_features()[:, None, :]).sum(0)
				+ pred.k_inv_y.sum() * torch.diagflat(pred.lengthscale ** 2))
		return result * self.__AVERAGE_CONSTANT

	def covariance(self) -> torch.Tensor:
		return super().covariance()

	def potential(self, model: pes.Potential) -> float:
		return math.nan

	def purity(self) -> torch.Tensor:
		result: torch.Tensor = torch.empty(self.config.NUM_PES, self.config.NUM_PES)
		for iPES in range(self.config.NUM_PES):
			for jPES in range(self.config.NUM_PES):
				pred: SinglePredictor = self.__predictors[iPES * self.config.NUM_PES + jPES]
				result[iPES, jPES] = (math.pi ** self.config.DIM) * pred.lengthscale.prod().item() * (pred.k_inv_y @ _cov(pred.lengthscale * math.sqrt(2.0), pred.get_training_features(), pred.get_training_features()).to_dense() @ pred.k_inv_y).item()
		return self.PURITY_FACTOR * (result + result.T - torch.diag(torch.diag(result)))
