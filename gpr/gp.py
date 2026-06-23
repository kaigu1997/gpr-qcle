r"""gp
==
Implementation for gaussian process (gp) regression.
"""
import collections.abc
import math
import typing
import linear_operator
import torch
# import torchsparsegradutils as tsgu
# from torchsparsegradutils.utils import bicgstab

import constant
import opt
import wendland

torch.set_default_dtype(constant.DTYPE)
torch.set_default_device(constant.DEVICE)
torch.manual_seed(constant.SEED)


class GaussianProcess:
	r"""Base class of kernel predictor

	Parameters
	----------
	x_ind : torch.Tensor, of shape (N_IND_PT, PHASEDIM)
		Inducing Points
	y_ind : torch.Tensor, of shape (N_IND_PT)
		Targets of inducing points
	x_all : torch.Tensor, of shape (N_ALL, PHASEDIM)
		All training inputs
	y_all : torch.Tensor, of shape (N_ALL)
		All training targets
	scale : float
		The scaling factor
	lengthscale_initial_value : torch.Tensor
		Initial value of lengthscale
	indent : int
		The indent for printing log
	train_rc : bool
		Whether to train the cutoff

	Attributes
	----------
	x_all : torch.Tensor, of shape (N_ALL, PHASEDIM)
		All training inputs
	y_all : torch.Tensor, of shape (N_ALL)
		All training targets

	Methods
	-------
	lengthscale()
		To access the real lengthscale
	predict(x_test)
		To predict the average
	predict_derivative_over_input(x_test)
		To give the derivative of prediction over the input
	predict_derivative_over_internal(x_test)
		To calculate the derivative of prediction over all related quantities
	error()
		To get the error by comparing label with prediction
	"""
	@typing.final
	class InputDerivativeReturn(typing.NamedTuple):
		predict: torch.Tensor
		derivative: torch.Tensor

	@typing.final
	class InternalDerivativeReturn(typing.NamedTuple):
		inducing_derivative: torch.Tensor
		feature_derivative: torch.Tensor
		label_derivative: torch.Tensor
		raw_param_derivative: torch.Tensor

	NOISE: typing.Final[float] = torch.finfo(torch.float32).eps
	length_raw_to_real: collections.abc.Callable[[torch.Tensor], torch.Tensor] = staticmethod(torch.nn.Softplus())
	length_real_to_raw: collections.abc.Callable[[torch.Tensor], torch.Tensor] = staticmethod(lambda x: x + torch.log(-torch.expm1(-x)))
	rc_raw_to_real: collections.abc.Callable[[torch.Tensor], torch.Tensor] = staticmethod(lambda x: torch.sigmoid(x) * wendland.wendland_rbf.max_support)
	rc_real_to_raw: collections.abc.Callable[[torch.Tensor], torch.Tensor] = staticmethod(lambda x: torch.log(x / wendland.wendland_rbf.max_support) - torch.log(1.0 - x / wendland.wendland_rbf.max_support))
	raw_to_real: collections.abc.Callable[[torch.Tensor], torch.Tensor] = staticmethod(lambda x: torch.cat([GaussianProcess.length_raw_to_real(x[:-1]), GaussianProcess.rc_raw_to_real(x[-1]).reshape(1)]))
	real_to_raw: collections.abc.Callable[[torch.Tensor], torch.Tensor] = staticmethod(lambda x: torch.cat([GaussianProcess.length_real_to_raw(x[:-1]), GaussianProcess.rc_real_to_raw(x[-1]).reshape(1)]))
	__slots__: typing.Final[tuple] = ("__PHASEDIM", "__AVERAGE_CONSTANT", "__kernel", "x_ind", "y_ind", "x_all", "y_all", "scale", "__raw_lengthscale", "__raw_cutoff", "__old_parameters", "__lr", "__k_inv_y", "__weights_updated")
	__PHASEDIM: typing.Final[int]
	__AVERAGE_CONSTANT: typing.Final[float]
	__kernel: typing.Final[wendland.wendland_rbf]
	x_ind: torch.Tensor
	y_ind: torch.Tensor
	x_all: torch.Tensor
	y_all: torch.Tensor
	scale: float
	__raw_lengthscale: torch.Tensor
	__raw_cutoff: torch.Tensor
	__old_parameters: torch.Tensor
	__lr: float | None
	__k_inv_y: torch.Tensor
	__weights_updated: bool

	def __init__(
		self,
		x_ind: torch.Tensor,
		y_ind: torch.Tensor,
		x_all: torch.Tensor,
		y_all: torch.Tensor,
		kernel: wendland.wendland_rbf,
		lengthscale_initial_value: torch.Tensor,
		cutoff: float,
		indent: int,
		train_rc: bool
	) -> None:
		self.__PHASEDIM = x_all.shape[-1]
		assert self.__PHASEDIM % 2 == 0
		DIM: typing.Final = x_all.shape[-1] // 2
		self.__AVERAGE_CONSTANT = (2.0 * math.pi) ** DIM / math.factorial(DIM)
		self.__kernel = kernel
		self.__weights_updated = False
		self.x_ind = x_ind.detach().clone()
		self.y_ind = y_ind.detach().clone()
		self.x_all = x_all.detach().clone()
		self.y_all = y_all.detach().clone()
		self.scale = 1.0 / y_all.abs().max().item()
		self.__raw_lengthscale = GaussianProcess.length_real_to_raw(lengthscale_initial_value).detach()
		self.__raw_cutoff = GaussianProcess.rc_real_to_raw(torch.tensor(cutoff)).detach()
		self.__lr = 1.0 if lengthscale_initial_value.numel() > 10 else None # use gradient descend if dimension is large, otherwise use newton method
		self.train(indent, train_rc)
		self.__old_parameters = self.raw_param.detach()
		self.__k_inv_y = self.__calculate_k_inv_y().detach()

	def get_chunk_size(self, x_test: torch.Tensor, print_log: bool = constant.DEBUG_MODE) -> int:
		r"""To get the chunk size for autograd

		Parameters
		----------
		x_test : torch.Tensor
			Test inputs
		print_log : bool, optional
			Whether to print log, by default constant.DEBUG_MODE

		Returns
		-------
		int
			The chunk size
		"""
		predict: typing.Final[torch.Tensor] = self.predict(x_test, requires_grad=True)
		N: typing.Final[int] = predict.numel()
		chunk_size: int = N
		change_to_2_power: bool = False
		power_of_2_max: typing.Final[int] = 64 # wavefront of 64 on AMD and warps of 32 on NV
		while chunk_size > 1:
			try:
				if print_log:
					print("\tChunk size for autograd test:", chunk_size)
				self.predict_derivative_over_internal(self.x_all.detach().requires_grad_(), chunk_size)
				break
			except torch.OutOfMemoryError:
				chunk_size //= 2
				if chunk_size < power_of_2_max and not change_to_2_power:
					chunk_size = power_of_2_max
					change_to_2_power = True
		print(f"Chunk size for autograd: {chunk_size}")
		return chunk_size

	@property
	def lengthscale(self) -> torch.Tensor:
		r"""To access the real lengthscale

		Returns
		-------
		torch.Tensor
			Lengthscale in kernel function
		"""
		return GaussianProcess.length_raw_to_real(self.__raw_lengthscale) # use self to allow subclass to override

	@property
	def raw_lengthscale(self) -> torch.Tensor:
		r"""To access the raw lengthscale

		Returns
		-------
		torch.Tensor
			Raw lengthscale
		"""
		return self.__raw_lengthscale

	@raw_lengthscale.setter
	def raw_lengthscale(self, value: torch.Tensor) -> None:
		self.__raw_lengthscale = value.detach().clone()
		self.__weights_updated = False

	@property
	def r_cutoff(self) -> float:
		r"""To access the cutoff radius in kernel function

		Returns
		-------
		float
			Cutoff radius in kernel function
		"""
		return GaussianProcess.rc_raw_to_real(self.__raw_cutoff).item()

	@property
	def raw_cutoff(self) -> torch.Tensor:
		r"""To access the raw cutoff radius in kernel function

		Returns
		-------
		torch.Tensor
			Raw cutoff radius in kernel function
		"""
		return self.__raw_cutoff

	@raw_cutoff.setter
	def raw_cutoff(self, value: torch.Tensor) -> None:
		self.__raw_cutoff = value.detach().clone()
		self.__weights_updated = False

	@property
	def raw_param(self) -> torch.Tensor:
		r"""To access the raw parameters

		Returns
		-------
		torch.Tensor
			The raw parameters
		"""
		return torch.cat([self.__raw_lengthscale, self.__raw_cutoff.reshape(1)])

	@raw_param.setter
	def raw_param(self, value: torch.Tensor) -> None:
		self.__raw_lengthscale = value[:-1].detach().clone()
		self.__raw_cutoff = value[-1].detach().clone()
		self.__weights_updated = False

	@property
	def param(self) -> torch.Tensor:
		"""To access the parameters

		Returns
		-------
		torch.Tensor
			The transformed parameters
		"""
		return torch.cat([self.lengthscale, torch.tensor([self.r_cutoff])])

	@property
	def old_param(self) -> torch.Tensor:
		r"""To access the old parameters

		Returns
		-------
		torch.Tensor
			The old parameters
		"""
		return self.__old_parameters

	def __calculate_k_inv_y(
		self,
		lengthscale: torch.Tensor | None = None,
		r_c: torch.Tensor | None = None,
		x_ind: torch.Tensor | None = None,
		x_all: torch.Tensor | None = None,
		y_all: torch.Tensor | None = None
	) -> torch.Tensor:
		r"""To calculate the weights, :math:`K^{-1}y`

		Parameters
		----------
		lengthscale : torch.Tensor | None, optional
			Lengthscale in kernel function, by default None (use current lengthscale)
		r_c : torch.Tensor | None, optional
			Cutoff radius in kernel function, by default None (use current cutoff)
		x_ind : torch.Tensor | None, optional
			Inducing points, by default None (use current inducing points)
		x_all : torch.Tensor | None, optional
			All training inputs, by default None (use current all training inputs)
		y_all : torch.Tensor | None, optional
			All training targets, by default None (use current all training targets)

		Returns
		-------
		torch.Tensor
			Weights, :math:`K^{-1}y`
		"""
		# return tsgu.sparse_generic_solve(
		# 	self.__kernel(
		# 		self.x_all,
		# 		lengthscale=GaussianProcess.raw_to_real(self.__raw_lengthscale if raw_lengthscale is None else raw_lengthscale),
		# 		r_c=GaussianProcess.rc_raw_to_real(self.__raw_cutoff if rc_raw is None else rc_raw)),
		# 	self.y_all,
		# 	bicgstab
		# )
		return linear_operator.utils.stable_pinverse(self.__kernel(
			self.x_all if x_all is None else x_all,
			self.x_ind if x_ind is None else x_ind,
			lengthscale=self.lengthscale if lengthscale is None else lengthscale,
			r_c=self.r_cutoff if r_c is None else r_c
		)) @ (self.y_all if y_all is None else y_all)

	def __update_weights(self) -> None:
		r"""To update the weights, :math:`K^{-1}y`
		"""
		if not self.__weights_updated:
			self.__k_inv_y = self.__calculate_k_inv_y().detach()
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

	def predict(
		self,
		x_test: torch.Tensor,
		raw_lengthscale: torch.Tensor | None = None,
		rc_raw: torch.Tensor | None = None,
		requires_grad: bool = False
	) -> torch.Tensor:
		r"""Instance of prediction

		Parameters
		----------
		x_test : torch.Tensor, shape of (N, PHASEDIM)
			Validation/Test inputs
		raw_lengthscale : torch.Tensor | None, optional
			Lengthscale in kernel function, by default None (use current lengthscale)
		rrc_raw_c : torch.Tensor | None, optional
			Cutoff radius in kernel function, by default None (use current cutoff)
		requires_grad : bool, optional
			Whether the prediction requires gradient, by default False

		Returns
		-------
		torch.Tensor, shape of (N,)
			Corresponding validation/test targets based on noise-free SR/PP mean.
		"""
		lengthscale: typing.Final[torch.Tensor] = self.lengthscale if raw_lengthscale is None else GaussianProcess.length_raw_to_real(raw_lengthscale)
		rc: typing.Final[torch.Tensor] = GaussianProcess.rc_raw_to_real(self.__raw_cutoff if rc_raw is None else rc_raw)
		if requires_grad:
			return self.__kernel(x_test, self.x_ind, lengthscale=lengthscale, r_c=rc) @ self.__calculate_k_inv_y(lengthscale, rc)
		else:
			return (self.__kernel(x_test, self.x_ind, lengthscale=lengthscale, r_c=rc) @ self.k_inv_y).detach()

	def predict_derivative_over_input(self, x_test: torch.Tensor) -> InputDerivativeReturn:
		r"""To give the derivative of prediction over the input

		Parameters
		----------
		x_test : torch.Tensor, shape of (N, PHASEDIM)
			Validation/Test inputs

		Returns
		-------
		InputDerivativeReturn
			Prediction (shape of (N,)), and its derivative over input (shape of (N, PHASEDIM))
		"""
		with torch.no_grad():
			x_test = x_test.reshape(-1, x_test.shape[-1]).detach().requires_grad_()
		predict: typing.Final[torch.Tensor] = self.predict(x_test, requires_grad=True)
		return GaussianProcess.InputDerivativeReturn(predict=predict.detach(), derivative=torch.autograd.grad(predict, x_test, torch.ones_like(predict), False, False, True, True, False, True)[0].detach())

	def predict_derivative_over_internal(self, x_test: torch.Tensor, chunk_size: int = 1) -> InternalDerivativeReturn:
		r"""To calculate the derivative of prediction over all related quantities

		Parameters
		----------
		x_test : torch.Tensor, shape of (N, PHASEDIM)
			Validation/Test inputs
		chunk_size : int, optional
			The number of VJP in parallel, used to avoid OOM, by default 1 (least OOM)

		Returns
		-------
		InternalDerivativeReturn
			derivative over training inducing features (shape of (N, m, PHASEDIM)),
			derivative over training feature (shape of (N, M, PHASEDIM)),
			derivative over training label (shape of (N, M)),
			and derivative over raw parameters (shape of (N, PHASEDIM + 1))
		"""
		with torch.no_grad():
			x_ind: typing.Final[torch.Tensor] = self.x_ind.detach().requires_grad_()
			x_all: typing.Final[torch.Tensor] = self.x_all.detach().requires_grad_()
			y_all: typing.Final[torch.Tensor] = self.y_all.detach().requires_grad_()
			raw_param: typing.Final[torch.Tensor] = self.raw_param.detach().requires_grad_()
		lengthscale: typing.Final[torch.Tensor] = GaussianProcess.length_raw_to_real(raw_param[:-1])
		r_c: typing.Final[torch.Tensor] = GaussianProcess.rc_raw_to_real(raw_param[-1])
		predict: typing.Final[torch.Tensor] = self.__kernel(x_test.detach(), x_ind, lengthscale=lengthscale, r_c=r_c) @ self.__calculate_k_inv_y(lengthscale, r_c, x_ind, x_all, y_all)
		N: typing.Final[int] = predict.numel()
		eye: typing.Final[torch.Tensor] = torch.eye(N)
		chunk_range: typing.Final[range] = range(0, N, chunk_size)
		return GaussianProcess.InternalDerivativeReturn(
			inducing_derivative=torch.cat([torch.autograd.grad(predict, x_ind, eye[i:min(i+chunk_size, N)], True, False, True, True, True, True)[0].detach() for i in chunk_range], dim=0),
			feature_derivative=torch.cat([torch.autograd.grad(predict, x_all, eye[i:min(i+chunk_size, N)], True, False, True, True, True, True)[0].detach() for i in chunk_range], dim=0),
			label_derivative=torch.cat([torch.autograd.grad(predict, y_all, eye[i:min(i+chunk_size, N)], True, False, True, True, True, True)[0].detach() for i in chunk_range], dim=0),
			raw_param_derivative=torch.cat([torch.autograd.grad(predict, raw_param, eye[i:min(i+chunk_size, N)], True, False, True, True, True, True)[0].detach() for i in chunk_range], dim=0)
		)

	def update_param(self, new_param: torch.Tensor) -> None:
		r"""To update the parameters

		Parameters
		----------
		new_param : torch.Tensor
			The updated parameter
		"""
		self.__old_parameters = self.raw_param.detach().clone()
		self.__raw_lengthscale = new_param[:-1].detach().clone()
		self.__raw_cutoff = new_param[-1].detach().clone()
		self.__weights_updated = False

	def update(
		self,
		x_ind: torch.Tensor,
		y_ind: torch.Tensor,
		x_all: torch.Tensor,
		y_all: torch.Tensor
	) -> None:
		r"""To update the training features and labels of the model

		Parameters
		----------
		x_ind : torch.Tensor, of shape (N_IND_PT, PHASEDIM)
			Inducing Points
		y_ind : torch.Tensor, of shape (N_IND_PT)
			Targets of inducing points
		x_all : torch.Tensor, of shape (N_ALL_PT, PHASEDIM)
			All training inputs
		y_all : torch.Tensor, of shape (N_ALL_PT)
			All training targets
		"""
		assert x_all.shape[-1] == self.__PHASEDIM and y_all.ndim == 1 and x_all.shape[0] == y_all.shape[0]
		self.x_ind = x_ind.reshape(-1, x_ind.shape[-1]).detach().clone()
		self.y_ind = y_ind.reshape(-1).detach().clone()
		self.x_all = x_all.reshape(-1, x_all.shape[-1]).detach().clone()
		self.y_all = y_all.reshape(-1).detach().clone()
		self.scale = 1.0 / y_all.abs().max().item()
		self.__weights_updated = False

	def error(self) -> torch.Tensor:
		r"""Error function of subset of regressor (SR) / projected process (PP)

		This function gives the sum of squared error

		Returns
		-------
		torch.Tensor
			The sum of squared prediction error
		"""
		return torch.sum((self.y_all - self.predict(self.x_all)) ** 2) * (self.scale ** 2)

	def loss_func(self, raw_lengthcscale: torch.Tensor, rc_raw: torch.Tensor) -> torch.Tensor:
		"""The default loss function for predictors, for optimization routine to minimize, whose parameter is the raw lengthscale and return a 0-dim Tensor

			Note that as for optimization, the real lengthscale is the transformation of raw lengthscale by `raw_to_real`

			This design is for the convenience of optimization, since the lengthscale should be positive, and using raw lengthscale can guarantee the positivity without extra constraints.

		Parameters
		----------
		raw_lengthcscale : torch.Tensor
			The raw parameters, which will be transformed to real lengthscale by `raw_to_real` and used in prediction and error calculation
		rc_raw: torch.Tensor
			The raw cutoff

		Returns
		-------
		torch.Tensor
			The loss, could be squared error, negative log marginal likelihood, or other loss function, as long as it is a 0-dim Tensor and can be optimized by optimization routine
		"""
		return torch.sum(torch.square(self.y_all - self.predict(self.x_all, raw_lengthcscale, rc_raw, True))) * (self.scale ** 2)

	def train(
		self,
		indent: int,
		train_rc: bool,
		print_log: bool = constant.DEBUG_MODE
	) -> None:
		r"""To train the parameters

		Parameters
		----------
		indent : int
			The indent for printing log
		train_rc : bool
			Whether to train the cutoff
		print_log : bool, optional
			Whether to print the log to console, by default `constant.DEBUG_MODE`
		"""
		def raw_to_real(raw_param: torch.Tensor) -> torch.Tensor:
			r"""To transform raw parameters to real parameters

			Parameters
			----------
			raw_param : torch.Tensor
				The raw parameters

			Returns
			-------
			torch.Tensor
				The real parameters
			"""
			return torch.cat([GaussianProcess.length_raw_to_real(raw_param[:-1]), GaussianProcess.rc_raw_to_real(raw_param[-1:])])

		loss_func: typing.Final[collections.abc.Callable[[torch.Tensor], torch.Tensor]] = lambda raw_param: self.loss_func(raw_param[:-1], raw_param[-1]) if train_rc else self.loss_func(raw_param, self.__raw_cutoff)
		# train model
		print(f"{indent * "\t"}scale = {self.scale}\n{indent * "\t"}", end="")
		opt.Optimizer.print_model(self.lengthscale)
		if self.__lr is None:
			loss: float = math.inf
			param: torch.Tensor = self.raw_param.detach().requires_grad_(True) if train_rc else self.__raw_lengthscale.clone().detach().requires_grad_(True)
			while True:
				print(f"{indent * "\t"}Optimization with Newton method:")
				result = opt.NewtonMethod(param, loss_func, indent + 1, print_log)
				print(f"{indent * "\t"}Iter = {result.num_iter} - {result.message}")
				opt.Optimizer.print_stuff(result.func_value, raw_to_real(result.param), None, extra_start_str=indent)
				if result.message in (opt.Optimizer.ResultMessage.GRAD,) or result.func_value >= loss:
					break
				else:
					loss = result.func_value
					param = result.param
				print(f"{indent * "\t"}Optimization with Gradient Descend method:")
				result = opt.GradientDescend(param, loss_func, indent + 1, print_log=print_log)
				print(f"{indent * "\t"}Iter = {result.num_iter} - {result.message}")
				opt.Optimizer.print_stuff(result.func_value, raw_to_real(result.param), result.lr, extra_start_str=indent)
				if result.message in (opt.Optimizer.ResultMessage.GRAD,) or result.func_value >= loss:
					break
				else:
					loss = result.func_value
					param = result.param
		else:
			result = opt.GradientDescend(self.__raw_lengthscale, loss_func, indent, self.__lr, print_log)
			print(f"{indent * "\t"}Iter = {result.num_iter} - {result.message}")
			opt.Optimizer.print_stuff(result.func_value, raw_to_real(result.param), result.lr, extra_start_str=indent)
		with torch.no_grad():
			self.__old_parameters = self.raw_param.detach().clone()
			if train_rc:
				self.__raw_lengthscale = result.param[:-1].detach().clone()
				self.__raw_cutoff = result.param[-1].detach().clone()
			else:
				self.__raw_lengthscale = result.param.detach().clone()
			if self.__lr is not None:
				self.__lr = result.lr
		self.__weights_updated = False

	@property
	def connectivity_and_total_weight(self) -> tuple[float, float, float, float]:
		# k_mat: typing.Final[torch.Tensor] = self.__kernel(self.x_all, lengthscale=GaussianProcess.raw_to_real(self.__raw_lengthscale.detach()), r_c=self.__cutoff) * torch.sqrt(torch.abs(self.y_all.reshape(1, -1))) * torch.sqrt(torch.abs(self.y_all.reshape(-1, 1)))
		k_mat = self.__kernel(self.x_all, lengthscale=self.lengthscale, r_c=self.__raw_cutoff)
		k_eigh: typing.Final[float] = torch.linalg.eigvalsh(k_mat.sum(-1).diag() - k_mat)[1].item()
		k_weight: typing.Final[float] = torch.tril(k_mat, -1).sum().item()
		k_mat = k_mat * torch.sqrt(torch.abs(self.y_all.reshape(1, -1))) * torch.sqrt(torch.abs(self.y_all.reshape(-1, 1)))
		return k_eigh, k_weight, torch.linalg.eigvalsh(k_mat.sum(-1).diag() - k_mat)[1].item(), torch.tril(k_mat, -1).sum().item()

	@property
	def population(self) -> float:
		r"""To get the population of the model if it is a diagonal Gaussian process regressor

		Returns
		-------
		float
			The population of the model
		"""
		return self.__AVERAGE_CONSTANT * self.lengthscale.prod().item() * self.k_inv_y.sum().item() * self.__kernel.integral(self.__PHASEDIM - 1, self.r_cutoff)

	def population_with_lengthscale(self, raw_lengthcscale: torch.Tensor, rc_raw: torch.Tensor) -> torch.Tensor:
		r"""To get the purity of the model

		Parameters
		----------
		raw_lengthcscale : torch.Tensor
			The raw parameters, which will be transformed to real lengthscale by `raw_to_real` and used in prediction and error calculation
		rc_raw: torch.Tensor
			The raw cutoff

		Returns
		-------
		float
			The purity of the model
		"""
		lengthscale: typing.Final[torch.Tensor] = GaussianProcess.length_raw_to_real(raw_lengthcscale)
		r_c: typing.Final[torch.Tensor] = GaussianProcess.rc_raw_to_real(rc_raw)
		return self.__AVERAGE_CONSTANT * lengthscale.prod() * self.__calculate_k_inv_y(lengthscale, r_c).sum() * self.__kernel.integral_d_1(r_c)

	@property
	def coordinates(self) -> torch.Tensor:
		r"""To get the coordinates of the model if it is a diagonal Gaussian process regressor

		Returns
		-------
		torch.Tensor
			The average coordinates of the model
		"""
		return self.__AVERAGE_CONSTANT * self.lengthscale.prod() * (self.k_inv_y[:, None] * self.x_ind).sum(0) * self.__kernel.integral(self.__PHASEDIM - 1, self.r_cutoff)

	@property
	def square_coordinates(self) -> torch.Tensor:
		r"""To get the square coordinates of the model if it is a diagonal Gaussian process regressor

		Returns
		-------
		torch.Tensor, shape of (PHASEDIM, PHASEDIM)
			The average square coordinates of the model
		"""
		return self.__AVERAGE_CONSTANT * self.lengthscale.prod() * ((self.k_inv_y[:, None, None] * self.x_ind[:, :, None] * self.x_ind[:, None, :]).sum(0) * self.__kernel.integral(self.__PHASEDIM - 1, self.r_cutoff) + self.k_inv_y.sum() * torch.diagflat(self.lengthscale ** 2) / self.__PHASEDIM * self.__kernel.integral(self.__PHASEDIM + 1, self.r_cutoff))

	@property
	def purity(self) -> float:
		r"""To get the purity of the model

		Returns
		-------
		float
			The purity of the model
		"""
		return (self.k_inv_y.reshape(1, -1) @ self.__kernel.square_integral(self.x_ind, lengthscale=self.lengthscale, r_c=self.r_cutoff) @ self.k_inv_y.reshape(-1, 1)).item() * self.lengthscale.prod().item()

	def purity_with_lengthscale(self, raw_lengthcscale: torch.Tensor, rc_raw: torch.Tensor) -> torch.Tensor:
		r"""To get the purity of the model

		Parameters
		----------
		raw_lengthcscale : torch.Tensor
			The raw parameters, which will be transformed to real lengthscale by `raw_to_real` and used in prediction and error calculation
		rc_raw: torch.Tensor
			The raw cutoff

		Returns
		-------
		float
			The purity of the model
		"""
		lengthscale: typing.Final[torch.Tensor] = GaussianProcess.length_raw_to_real(raw_lengthcscale)
		r_c: typing.Final[torch.Tensor] = GaussianProcess.rc_raw_to_real(rc_raw)
		k_inv_y: typing.Final[torch.Tensor] = self.__calculate_k_inv_y(lengthscale, r_c)
		return (k_inv_y.reshape(1, -1) @ self.__kernel.square_integral(self.x_ind, lengthscale=lengthscale, r_c=r_c) @ k_inv_y.reshape(-1, 1)).reshape([]) * lengthscale.prod()

	def get_marginal(self, x_test: torch.Tensor, dimensions: collections.abc.Sequence[int]) -> torch.Tensor:
		r"""To get the marginal distribution of current gaussian process regression

		Parameters
		----------
		x_test : torch.Tensor, shape of (N, len(dimensions))
			Validation/Test inputs
		dimensions : collections.abc.Sequence[int]
			The dimensions to be kept, must not have any repeat

		Returns
		-------
		torch.Tensor, shape of (N,)
			Corresponding validation/test targets based on noise-free SR/PP mean.
		"""
		phasedim: typing.Final[int] = self.x_all.shape[-1]
		prefactor: typing.Final[float] = math.sqrt((2.0 * torch.pi) ** (phasedim - len(dimensions))) * self.lengthscale[[i for i in range(phasedim) if i not in dimensions]].prod().item()
		return prefactor * self.__kernel(x_test, self.x_ind[:, dimensions], lengthscale=self.lengthscale[dimensions], r_c=self.r_cutoff).to_dense() @ self.k_inv_y
