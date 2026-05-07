r"""pes.impl
============
This is the implementaion of the diabatic potential
"""
import math
import typing

import torch

import constant

from . import models

torch.set_default_dtype(constant.DTYPE)
torch.set_default_device(constant.DEVICE)


@typing.final
class PotentialQuantity(typing.NamedTuple):
	diabatic_potential: torch.Tensor | None
	diabatic_to_adiabatic: torch.Tensor | None
	adiabatic_potential: torch.Tensor | None
	coupling: torch.Tensor | None
	second_order_coupling: torch.Tensor | None
	force: torch.Tensor | None


@typing.final
class Potential:
	r"""Quantities from potential, including basis transform, potential, forces, and coupling

	All the quantities are calculated only when needed

	Parameters
	----------
	model : type[models.ModelBase]
		The model type
	config : pes.ModelConfig
		Configuration of the model

	Methods
	-------
	diabatic_potential_on_all_grids()
		To get the diabatic potential at all coordinate grids
	diabatic_to_adiabatic_on_all_grids()
		To get the basis transformation matrices at all coordinate grids
	adiabatic_potential_on_all_grids()
		To get the adiabatic potential at all coordinate grids by diagonalization
	force_on_all_grids()
		To get the adiabatic Hellmann-Feynmann force :math:`<\psi_i|-\frac{\partial\hat{H}}{\partial\mathbf{R}}|\psi_j>` at all coordinate grids
	coupling_on_all_grids()
		To get the first order non-adiabatic coupling :math:`<\psi_i|\frac{\partial}{\partial\mathbf{R}}|\psi_j>` at all coordinate grids
	second_order_coupling_on_all_grids()
		To get the second order non-adiabatic coupling :math:`<\psi_i|\frac{\partial^2}{\partial R_d^2}|\psi_j>\forall d` at all coordinate grids
	"""
	__slots__: typing.ClassVar[tuple] = ("__model", "__config")
	__model: typing.Final[type[models.ModelBase]]
	__config: typing.Final[models.ModelConfig]

	@models.model_validation
	def __init__(
		self,
		model: type[models.ModelBase],
		config: models.ModelConfig
	) -> None:
		self.__model = model
		self.__config = config

	@property
	def config(self) -> models.ModelConfig:
		r"""To get the configuration of the model

		Returns
		-------
		models.ModelConfig
			Configuration of the model potential
		"""
		return self.__config

	def __need_manual(self, method: str, need: bool = True) -> bool:
		r"""To decide whether need to calculate the quantity manually

		Parameters
		----------
		method : str
			Name of the method
		need : bool, optional
			Whether the quantity need to be calculated or not, by default `True`

		Returns
		-------
		bool
			Whether need to calculate the quantity manually
		"""
		def is_implemented_since[T](derived_cls: type[T], method_name: str, base_cls: type[T]) -> bool:
			r"""To check if `derived_cls` or any of its ancestors (up to but not including `base_cls`) has implemented method_name.

			Parameters
			----------
			derived_cls : type[object]
				Derived classes, check if the method is implemented in it.
			method_name : str
				The name of the method, to see if it is implemented
			base_cls : type[object]
				The base class. `method_name` raises `NotImplementedError` there, in general.

			Returns
			-------
			bool
				True if the method has been overridden somewhere between `base_cls` and `derived_cls` in the inheritance chain
			"""
			# Get the method from base_cls
			base_descriptor = base_cls.__dict__.get(method_name)
			if base_descriptor is None:
				return False
			# Walk through MRO from cls up to (but not including) base_cls
			for mro_cls in derived_cls.__mro__:
				if mro_cls == base_cls:
					break
				current_descriptor = mro_cls.__dict__.get(method_name)
				if current_descriptor is not None and current_descriptor is not base_descriptor:
					return True
			return False

		return need and not is_implemented_since(self.__model, method, models.ModelBase)

	def __C(self, V: torch.Tensor) -> torch.Tensor:
		r"""To get the basis transformation matrices

		Parameters
		----------
		V : torch.Tensor
			Diabatic potential

		Returns
		-------
		torch.Tensor
			Basis transformation matrices
		"""
		match self.__config.NUM_PES:
			case 1:
				return torch.eye(self.__config.NUM_PES)
			case 2:
				V00: typing.Final[torch.Tensor] = V[..., 0, 0] # ...
				V01: typing.Final[torch.Tensor] = V[..., 0, 1] # ...
				V11: typing.Final[torch.Tensor] = V[..., 1, 1] # ...
				diff: typing.Final[torch.Tensor] = V00 - V11 # ...
				V01_double: typing.Final[torch.Tensor] = 2.0 * V01 # ...
				discriminant: typing.Final[torch.Tensor] = torch.sqrt(torch.square(diff) + torch.square(V01_double)) # ...
				discriminant_absdiff: typing.Final[torch.Tensor] = discriminant + torch.abs(diff) # ...
				C_unnormalized: typing.Final[torch.Tensor] = torch.where(diff[..., torch.newaxis] < 0, torch.stack([-discriminant_absdiff, V01_double, V01_double, discriminant_absdiff], -1), torch.stack([-V01_double, discriminant_absdiff, discriminant_absdiff, V01_double], -1)).reshape_as(V)
				return C_unnormalized / torch.linalg.norm(C_unnormalized, axis=-2, keepdims=True)
			case _:
				return torch.linalg.eigh(V.cpu()).eigenvectors.to(torch.get_default_device())

	def __E(self, V: torch.Tensor) -> torch.Tensor:
		r"""To get the adiabatic potential, if needed

		Parameters
		----------
		V : torch.Tensor
			Diabatic potential

		Returns
		-------
		torch.Tensor
			Adiabatic potential
		"""
		match self.__config.NUM_PES:
			case 1:
				return V
			case 2:
				V00: typing.Final[torch.Tensor] = V[..., 0, 0] # ...
				V01: typing.Final[torch.Tensor] = V[..., 0, 1] # ...
				V11: typing.Final[torch.Tensor] = V[..., 1, 1] # ...
				diff: typing.Final[torch.Tensor] = V00 - V11 # ...
				V01_double: typing.Final[torch.Tensor] = 2.0 * V01 # ...
				discriminant: typing.Final[torch.Tensor] = torch.sqrt(torch.square(diff) + torch.square(V01_double)) # ...
				return ((V00 + V11)[..., torch.newaxis] + torch.tensor([-1.0, 1.0]) * discriminant[..., torch.newaxis]) / 2.0
			case _:
				return torch.linalg.eigvalsh(V.cpu()).to(torch.get_default_device()) # pylint: disable=not-callable

	def __C_and_E(self, V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		r"""To get the the basis transformation matrices and the adiabatic potential, if needed

		Parameters
		----------
		V : torch.Tensor
			Diabatic potential

		Returns
		-------
		tuple[torch.Tensor, torch.Tensor]
			Basis transformation matrices, adiabatic potential
		"""
		match self.__config.NUM_PES:
			case 1:
				return torch.eye(self.__config.NUM_PES), V
			case 2:
				V00: typing.Final[torch.Tensor] = V[..., 0, 0] # ...
				V01: typing.Final[torch.Tensor] = V[..., 0, 1] # ...
				V11: typing.Final[torch.Tensor] = V[..., 1, 1] # ...
				diff: typing.Final[torch.Tensor] = V00 - V11 # ...
				V01_double: typing.Final[torch.Tensor] = 2.0 * V01 # ...
				discriminant: typing.Final[torch.Tensor] = torch.sqrt(torch.square(diff) + torch.square(V01_double)) # ...
				discriminant_absdiff: typing.Final[torch.Tensor] = discriminant + torch.abs(diff) # ...
				C_unnormalized: typing.Final[torch.Tensor] = torch.where(diff[..., torch.newaxis] < 0, torch.stack([-discriminant_absdiff, V01_double, V01_double, discriminant_absdiff], -1), torch.stack([-V01_double, discriminant_absdiff, discriminant_absdiff, V01_double], -1)).reshape_as(V)
				return C_unnormalized / torch.linalg.norm(C_unnormalized, axis=-2, keepdims=True), ((V00 + V11)[..., torch.newaxis] + torch.tensor([-1.0, 1.0]) * discriminant[..., torch.newaxis]) / 2.0
			case _:
				result: typing.Final = torch.linalg.eigh(V) # pylint: disable=not-callable
				return result.eigenvectors.to(torch.get_default_device()), result.eigenvalues.to(torch.get_default_device())
		return C, E

	def __D(
		self,
		x: torch.Tensor,
		C: torch.Tensor,
		matrix_to_supervector_shape: tuple[int, ...],
		matrix_eye_broadcast_shape: tuple[int, ...],
		matrix_eye_repeat_shape: tuple[int, ...],
		deriv_result_shape: tuple[int, ...],
	) -> torch.Tensor:
		r"""To calculate the first order nonadiabatic coupling under adiabatic basis

		Parameters
		----------
		x : torch.Tensor
			All cooridnates
		C : torch.Tensor
			Basis transformation matrices
		matrix_to_supervector_shape : tuple[int, ...]
			Shape to reshape (..., NUM_PES, NUM_PES) to (..., NUM_ELM)
		matrix_eye_broadcast_shape : tuple[int, ...]
			Shape to broadcast (NUM_ELM, NUM_ELM) identity to (NUM_ELM, 1, ..., 1, NUM_ELM)
		matrix_eye_repeat_shape : tuple[int, ...]
			Shape to repeat (NUM_ELM, 1, ..., 1, NUM_ELM) identity to (NUM_ELM, ..., NUM_ELM)
		deriv_result_shape : tuple[int, ...]
			Shape of result, (..., DIM, NUM_PES, NUM_PES)

		Returns
		-------
		torch.Tensor
			Nonadiabatic coupling
		"""
		result: typing.Final[torch.Tensor] = C[..., torch.newaxis, :, :].mH @ torch.autograd.grad(C.reshape(matrix_to_supervector_shape), x, torch.eye(self.__config.NUM_ELM).reshape(matrix_eye_broadcast_shape).repeat(matrix_eye_repeat_shape), True, True, True, True, True, True)[0].moveaxis(0, -1).reshape(deriv_result_shape)
		return (result - result.mH) / 2.0

	def __F(
		self,
		x: torch.Tensor,
		C: torch.Tensor,
		E: torch.Tensor,
		V: torch.Tensor,
		vector_eye_broadcast_shape: tuple[int, ...],
		vector_eye_repeat_shape: tuple[int, ...],
		matrix_to_supervector_shape: tuple[int, ...],
		matrix_eye_broadcast_shape: tuple[int, ...],
		matrix_eye_repeat_shape: tuple[int, ...],
		deriv_result_shape: tuple[int, ...],
	) -> torch.Tensor:
		r"""To calculate the adiabatic force

		Parameters
		----------
		x : torch.Tensor
			All cooridnates
		C : torch.Tensor
			Basis transformation matrices
		E : torch.Tensor
			Adiabatic potential
		V : torch.Tensor
			Diabatic potential
		vector_eye_broadcast_shape : tuple[int, ...]
			Shape to broadcast (NUM_PES, NUM_PES) identity to (NUM_PES, 1, ..., 1, NUM_PES)
		vector_eye_repeat_shape : tuple[int, ...]
			Shape to repeat (NUM_PES, 1, ..., 1, NUM_PES) identity to (NUM_PES, ..., NUM_PES)
		matrix_to_supervector_shape : tuple[int, ...]
			Shape to reshape (..., NUM_PES, NUM_PES) to (..., NUM_ELM)
		matrix_eye_broadcast_shape : tuple[int, ...]
			Shape to broadcast (NUM_ELM, NUM_ELM) identity to (NUM_ELM, 1, ..., 1, NUM_ELM)
		matrix_eye_repeat_shape : tuple[int, ...]
			Shape to repeat (NUM_ELM, 1, ..., 1, NUM_ELM) identity to (NUM_ELM, ..., NUM_ELM)
		deriv_result_shape : tuple[int, ...]
			Shape of result, (..., DIM, NUM_PES, NUM_PES)

		Returns
		-------
		torch.Tensor
			Adiabatic force
		"""
		dE_dR: typing.Final[torch.Tensor] = torch.autograd.grad(E, x, torch.eye(self.__config.NUM_PES).reshape(vector_eye_broadcast_shape).repeat(vector_eye_repeat_shape), True, True, True, True, True, True)[0].moveaxis(0, -1) # HF force
		# pylint: disable-next=no-member
		diaF: typing.Final = -torch.autograd.grad(V.reshape(matrix_to_supervector_shape), x, torch.eye(self.__config.NUM_ELM).reshape(matrix_eye_broadcast_shape).repeat(matrix_eye_repeat_shape), True, True, True, True, True, True)[0].moveaxis(0, -1).reshape(deriv_result_shape)
		result: typing.Final[torch.Tensor] = C[..., torch.newaxis, :, :].mH @ diaF @ C[..., torch.newaxis, :, :]
		return (result + result.mH) / 2.0 - torch.diag_embed(dE_dR + torch.diagonal(result, 0, -2, -1))

	def __G(
		self,
		x: torch.Tensor,
		D: torch.Tensor,
		tensor_to_supervector_shape: tuple[int,...],
		tensor_eye_broadcast_shape: tuple[int,...],
		tensor_eye_repeat_shape: tuple[int,...],
		deriv_result_shape: tuple[int, ...],
	) -> torch.Tensor:
		r"""To calculate the second order coupling

		Parameters
		----------
		x : torch.Tensor
			All cooridnates
		D : torch.Tensor
			First order coupling
		tensor_to_supervector_shape : tuple[int,...]
			Shape to reshape (..., DIM, NUM_PES, NUM_PES) to (..., DIM, NUM_ELM)
		tensor_eye_broadcast_shape : tuple[int,...]
			Shape to broadcast (NUM_ELM, NUM_ELM) identity to (NUM_ELM, 1, ..., 1, 1, NUM_ELM)
		tensor_eye_repeat_shape : tuple[int,...]
			Shape to repeat (NUM_ELM, 1, ..., 1, 1, NUM_ELM) identity to (NUM_ELM, ..., DIM, NUM_ELM)
		deriv_result_shape : tuple[int, ...]
			Shape of result, (..., DIM, NUM_PES, NUM_PES)

		Returns
		-------
		torch.Tensor
			The second order non-adiabatic coupling
		"""
		return D @ D + torch.autograd.grad(D.reshape(tensor_to_supervector_shape), x, torch.eye(self.__config.NUM_ELM).reshape(tensor_eye_broadcast_shape).repeat(tensor_eye_repeat_shape), True, True, True, True, True, True)[0].moveaxis(0, -1).reshape(deriv_result_shape)

	def adiabatic_potential(self, x: torch.Tensor) -> torch.Tensor:
		r"""To calculate adiabatic potential

		Parameters
		----------
		x : torch.Tensor, shape of (..., DIM)
			All cooridnates

		Returns
		-------
		torch.Tensor, shape of (..., NUM_PES)
			adiabatic potential
		"""
		if self.__need_manual("adiabatic_potential"):
			return self.__E(self.__model(x)).detach()
		else:
			return self.__model.adiabatic_potential(x).detach()

	def coupling(self, x: torch.Tensor) -> torch.Tensor:
		r"""To calculate the first order nonadiabatic coupling under adiabatic basis

		Parameters
		----------
		x : torch.Tensor, shape of (..., DIM)
			All cooridnates

		Returns
		-------
		torch.Tensor, shape of (..., DIM, NUM_PES, NUM_PES)
			Nonadiabatic coupling
		"""
		if self.__need_manual("coupling"):
			x = x.detach().requires_grad_()
			return self.__D(
				x,
				self.__C(self.__model(x)),
				x.shape[:-1] + (self.__config.NUM_ELM,),
				(self.__config.NUM_ELM,) + (1,) * (x.ndim - 1) + (self.__config.NUM_ELM,),
				(1,) + x.shape[:-1] + (1,),
				x.shape + (self.__config.NUM_PES, self.__config.NUM_PES)
			).detach()
		else:
			return self.__model.coupling(x).detach()

	def adiabatic_potential_and_coupling(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		r"""To calculate adiabatic potential and non-adiabatic coupling under adiabatic basis

		Parameters
		----------
		x : torch.Tensor, shape of (..., DIM)
			All cooridnates

		Returns
		-------
		tuple[torch.Tensor, torch.Tensor]
			Adiabatic potential and coupling
		"""
		need_manual_C: typing.Final[bool] = self.__need_manual("diabatic_to_adiabatic")
		need_manual_D: typing.Final[bool] = self.__need_manual("coupling")
		need_manual_E: typing.Final[bool] = self.__need_manual("adiabatic_potential")
		if (need_manual_C or need_manual_E) and need_manual_D:
			x = x.detach().requires_grad_()
			C, E = self.__C_and_E(self.__model(x))
			return E.detach(), self.__D(
				x,
				C,
				x.shape[:-1] + (self.__config.NUM_ELM,),
				(self.__config.NUM_ELM,) + (1,) * (x.ndim - 1) + (self.__config.NUM_ELM,),
				(1,) + x.shape[:-1] + (1,),
				x.shape + (self.__config.NUM_PES, self.__config.NUM_PES)
			).detach()
		elif need_manual_D:
			E = self.__model.adiabatic_potential(x)
			x = x.detach().requires_grad_()
			return E.detach(), self.__D(
				x,
				self.__model.diabatic_to_adiabatic(x),
				x.shape[:-1] + (self.__config.NUM_ELM,),
				(self.__config.NUM_ELM,) + (1,) * (x.ndim - 1) + (self.__config.NUM_ELM,),
				(1,) + x.shape[:-1] + (1,),
				x.shape + (self.__config.NUM_PES, self.__config.NUM_PES)
			).detach()
		else:
			return self.__model.adiabatic_potential(x).detach(), self.__model.coupling(x).detach()

	def force(self, x: torch.Tensor) -> torch.Tensor:
		r"""To calculate the adiabatic force

		Parameters
		----------
		x : torch.Tensor, shape of (..., DIM)
			All cooridnates

		Returns
		-------
		torch.Tensor, shape of (..., DIM, NUM_PES, NUM_PES)
			Adiabatic force
		"""
		if self.__need_manual("adiabatic_force"):
			x = x.detach().requires_grad_()
			V: typing.Final[torch.Tensor] = self.__model(x)
			need_manual_E: typing.Final[bool] = self.__need_manual("adiabatic_potential")
			# who need C: D, F
			need_manual_C: typing.Final[bool] = self.__need_manual("diabatic_to_adiabatic")
			if need_manual_C or need_manual_E:
				C, E = self.__C_and_E(V)
			else: # not need_manual_C and not need_manual_E
				C = self.__model.diabatic_to_adiabatic(x)
				E = self.__model.adiabatic_potential(x)
			return self.__F(
				x,
				C,
				E,
				V,
				(self.__config.NUM_PES,) + (1,) * (x.ndim - 1) + (self.__config.NUM_PES,),
				(1,) + x.shape[:-1] + (1,),
				x.shape[:-1] + (self.__config.NUM_ELM,),
				(self.__config.NUM_ELM,) + (1,) * (x.ndim - 1) + (self.__config.NUM_ELM,),
				(1,) + x.shape[:-1] + (1,),
				x.shape + (self.__config.NUM_PES, self.__config.NUM_PES)
			).detach()
		else:
			return self.__model.adiabatic_force(x).detach()

	def coupling_and_second_order_coupling(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		r"""To calculate the first and second order nonadiabatic coupling under adiabatic basis

		Parameters
		----------
		x : torch.Tensor, shape of (..., DIM)
			All cooridnates

		Returns
		-------
		tuple[torch.Tensor, torch.Tensor]
			The first and second order coupling
		"""
		x = x.detach().requires_grad_()
		C: typing.Final = (self.__C(self.__model(x)) if self.__need_manual("diabatic_to_adiabatic") else self.__model.diabatic_to_adiabatic(x)) if self.__need_manual("coupling") else None
		D: typing.Final[torch.Tensor] = self.__D(
			x,
			C,
			x.shape[:-1] + (self.__config.NUM_ELM,),
			(self.__config.NUM_ELM,) + (1,) * (x.ndim - 1) + (self.__config.NUM_ELM,),
			(1,) + x.shape[:-1] + (1,),
			x.shape + (self.__config.NUM_PES, self.__config.NUM_PES)
		) if C is not None else self.__model.coupling(x)
		return D.detach(), self.__G(
			x,
			D,
			x.shape + (self.__config.NUM_ELM,),
			(self.__config.NUM_ELM,) + (1,) * x.ndim + (self.__config.NUM_ELM,),
			(1,) + x.shape + (1,),
			x.shape + (self.__config.NUM_PES, self.__config.NUM_PES)
		).detach() if self.__need_manual("second_order_coupling") else self.__model.second_order_coupling(x).detach()


	def __call__(
		self,
		x: torch.Tensor,
		*,
		diabatic_potential: bool = False,
		diabatic_to_adiabatic: bool = False,
		adiabatic_potential: bool = False,
		coupling: bool = False,
		second_order_coupling: bool = False,
		force: bool = False,
	) -> PotentialQuantity:
		r"""To get the selected quantities for all inputs

		Returns
		-------
		torch.Tensor, dtype of `torch.double`, shape of (N_GRIDS, ..., NUM_PES, NUM_PES)
			Diabatic potential at all coordinate grids

		Parameters
		----------
		x : torch.Tensor, shape of (..., DIM)
			All cooridnates
		diabatic_potential : bool
			To calculate diabatic potential
		diabatic_to_adiabatic : bool
			To calculate diabatic to adiabatic basis transformation matrix
		adiabatic_potential : bool
			To calculate adiabatic potential
		coupling : bool
			To calculate non-adiabatic coupling under adiabatic basis
		second_order_coupling : bool
			To calculate second-order non-adiabatic coupling under adiabatic basis
		force : bool
			To calculate adiabatic force

		Returns
		-------
		PotentialQuantity
			Keys are the same as the parameters, and the values are the corresponding quantities
		"""

		assert any((diabatic_potential, diabatic_to_adiabatic, adiabatic_potential, coupling, second_order_coupling, force))
		x = x.detach().requires_grad_()
		# who need D: G
		need_manual_G: typing.Final[bool] = self.__need_manual("second_order_coupling", second_order_coupling)
		need_D: typing.Final[bool] = coupling or need_manual_G
		need_manual_D: typing.Final[bool] = self.__need_manual("coupling", need_D)
		# who need adiaV: F
		need_manual_F: typing.Final[bool] = self.__need_manual("adiabatic_force", force)
		need_E: typing.Final[bool] = adiabatic_potential or need_manual_F
		need_manual_E: typing.Final[bool] = self.__need_manual("adiabatic_potential", need_E)
		# who need C: D, F
		need_C: typing.Final[bool] = diabatic_to_adiabatic or need_manual_D or need_manual_F
		need_manual_C: typing.Final[bool] = self.__need_manual("diabatic_to_adiabatic", need_C)
		# who need V: C, E, F
		need_V: typing.Final[bool] = diabatic_potential or need_manual_C or need_manual_E or need_manual_F
		# shapes
		vector_eye_broadcast_shape: typing.Final[tuple[int, ...] | None] = (self.__config.NUM_PES,) + (1,) * (x.ndim - 1) + (self.__config.NUM_PES,) if need_manual_F else None
		vector_eye_repeat_shape: typing.Final[tuple[int, ...] | None] = (1,) + x.shape[:-1] + (1,) if need_manual_F else None
		matrix_to_supervector_shape: typing.Final[tuple[int, ...] | None] = x.shape[:-1] + (self.__config.NUM_ELM,) if need_manual_D or need_manual_F else None
		matrix_eye_broadcast_shape: typing.Final[tuple[int, ...] | None] = (self.__config.NUM_ELM,) + (1,) * (x.ndim - 1) + (self.__config.NUM_ELM,) if need_manual_D or need_manual_F else None
		matrix_eye_repeat_shape: typing.Final[tuple[int, ...] | None] = (1,) + x.shape[:-1] + (1,) if need_manual_D or need_manual_F else None
		tensor_to_supervector_shape: typing.Final[tuple[int, ...] | None] = x.shape + (self.__config.NUM_ELM,) if need_manual_G else None
		tensor_eye_broadcast_shape: typing.Final[tuple[int, ...] | None] = (self.__config.NUM_ELM,) + (1,) * x.ndim + (self.__config.NUM_ELM,) if need_manual_G else None
		tensor_eye_repeat_shape: typing.Final[tuple[int, ...] | None] = (1,) + x.shape + (1,) if need_manual_G else None
		deriv_result_shape: typing.Final[tuple[int, ...] | None] = x.shape + (self.__config.NUM_PES, self.__config.NUM_PES) if need_manual_D or need_manual_F or need_manual_G else None
		V: typing.Final[torch.Tensor | None] = self.__model(x) if need_V else None
		C: torch.Tensor | None = None
		E: torch.Tensor | None = None
		if need_C and not need_manual_C:
			C = self.__model.diabatic_to_adiabatic(x)
		if need_E and not need_manual_E:
			E = self.__model.adiabatic_potential(x)
		if need_manual_C or need_manual_E:
			assert V is not None
			if not need_manual_C:
				E = self.__E(V)
			elif not need_manual_E:
				C = self.__C(V)
			else: # need_manual_C and need_manual_E
				C, E = self.__C_and_E(V)
		D: typing.Final[torch.Tensor | None] = (self.__D(x, C, matrix_to_supervector_shape, matrix_eye_broadcast_shape, matrix_eye_repeat_shape, deriv_result_shape) if need_manual_D else self.__model.coupling(x)) if need_D else None # type: ignore[reportArgumentType]
		F: torch.Tensor | None = None
		if force:
			if need_manual_F:
				assert C is not None
				assert E is not None
				assert V is not None
				assert vector_eye_broadcast_shape is not None
				assert vector_eye_repeat_shape is not None
				assert matrix_to_supervector_shape is not None
				assert matrix_eye_broadcast_shape is not None
				assert matrix_eye_repeat_shape is not None
				assert deriv_result_shape is not None
				F = self.__F(x, C, E, V, vector_eye_broadcast_shape, vector_eye_repeat_shape, matrix_to_supervector_shape, matrix_eye_broadcast_shape, matrix_eye_repeat_shape, deriv_result_shape)
			else:
				F = self.__model.adiabatic_force(x)
		G: torch.Tensor | None = None
		if second_order_coupling:
			if need_manual_G:
				assert D is not None
				assert tensor_to_supervector_shape is not None
				assert tensor_eye_broadcast_shape is not None
				assert tensor_eye_repeat_shape is not None
				assert deriv_result_shape is not None
				G = self.__G(x, D, tensor_to_supervector_shape, tensor_eye_broadcast_shape, tensor_eye_repeat_shape, deriv_result_shape)
			else:
				G = self.__model.second_order_coupling(x)
		return PotentialQuantity(
			diabatic_potential=V.detach() if diabatic_potential and V is not None else None,
			diabatic_to_adiabatic=C.detach() if diabatic_to_adiabatic and C is not None else None,
			adiabatic_potential=E.detach() if adiabatic_potential and E is not None else None,
			coupling=D.detach() if coupling and D is not None else None,
			second_order_coupling=G.detach() if second_order_coupling and G is not None else None,
			force=F.detach() if force and F is not None else None
		)


@typing.final
class InitialDistribution:
	r"""To generate the initial partial Wigner-transformed density matrix

	Parameters
	----------
	model : Potential
		Quantities derived from potential
	r0 : torch.Tensor, shape of (PHASEDIM,)
		Initial center of each dimension
	sigma_r0 : torch.Tensor, shape of (PHASEDIM,)
		Initial standard deviation of each dimension
	init_ppl_and_phase : torch.Tensor, shape of (NUM_PES,)
		The population and phase factor on each surface, whose squared sum is normalized. Default to all on the ground state.
	diabatic : bool, optional
		Whether to construct in diabatic basis or adiabatic basis, by default `False` (adiabatic basis)

	Methods
	-------
	r0()
		To get the centers
	sigma_r0()
		To get the widths
	weight()
		To get the weight of each element
	config()
		To get the configuration of the model

	Notes
	-----
	The density is a multidimensional gaussian distribution with given center and width.

	The population on each surface and their phase factor difference is set in the function.

	The off-diagonal elements guarantee the purity of the initial distribution to be 1, i.e., pure state.
	"""
	__slots__: tuple = ("__model", "__r0", "__sigma_r0", "__weight_phase", "__factors", "__order")
	__model: typing.Final[Potential]
	__r0: typing.Final[torch.Tensor]
	__sigma_r0: typing.Final[torch.Tensor]
	__weight_phase: typing.Final[torch.Tensor]
	__factors: typing.Final[torch.Tensor]
	__order: typing.Final[typing.Literal[0,1,2]]

	def __init__(
		self,
		model: Potential,
		r0: torch.Tensor,
		sigma_r0: torch.Tensor,
		init_ppl_and_phase: torch.Tensor,
		*,
		diabatic: bool = False
	):
		self.__model = model
		self.__r0 = r0
		self.__sigma_r0 = sigma_r0
		weight_phase: typing.Final[torch.Tensor] = init_ppl_and_phase.conj()[:, torch.newaxis] * init_ppl_and_phase # shape of (NUM_PES, NUM_PES)
		self.__weight_phase = (weight_phase + weight_phase.T.conj()) / 2.0 # remove numerical error, shape of (NUM_PES, NUM_PES)
		self.__factors = self.__weight_phase / (2.0 * math.pi) ** model.config.DIM / self.__sigma_r0.prod() # divide by normalization factor, shape of (NUM_ELM,)
		if diabatic:
			self.__order = 0 # direct transform
		elif torch.any(init_ppl_and_phase == 0).item():
			self.__order = 2 # second order correction needed for elements to be populated
		else:
			self.__order = 1

	@property
	def r0(self) -> torch.Tensor:
		r"""To get the centers

		Returns
		-------
		torch.Tensor
			Initial center of each dimension
		"""
		return self.__r0

	@property
	def sigma_r0(self) -> torch.Tensor:
		r"""To get the widths

		Returns
		-------
		torch.Tensor
			Initial standard deviation of each dimension
		"""
		return self.__sigma_r0
	
	@property
	def weight(self) -> torch.Tensor:
		r"""To get the weight of each element

		Returns
		-------
		torch.Tensor
			Initial weight (without phase factor) of each element
		"""
		return torch.abs(self.__weight_phase)

	@property
	def config(self) -> models.ModelConfig:
		r"""To get the configuration of the model

		Returns
		-------
		models.ModelConfig
			Configuration of the model
		"""
		return self.__model.config

	def __call__(self, r: torch.Tensor) -> torch.Tensor:
		r"""To calculate the initial density of the given element at the given phase point

		Parameters
		----------
		r : torch.Tensor, shape of (..., PHASEDIM)
			Phase space coordinates of initerest

		Returns
		-------
		torch.Tensor, shape of (..., NUM_PES, NUM_PES)
			Density matrices of the given elements
		"""
		x: typing.Final[torch.Tensor] = r[..., :self.__model.config.DIM].detach()
		p: typing.Final[torch.Tensor] = r[..., self.__model.config.DIM:].detach()
		if self.__order > 0:
			p.requires_grad_()
			r = torch.cat((x, p), -1)
		full_mat_0: typing.Final[torch.Tensor] = torch.exp(-(((r - self.__r0) / self.__sigma_r0) ** 2).sum(-1) / 2.0)
		result: torch.Tensor = full_mat_0[:, torch.newaxis, torch.newaxis] * self.__factors
		if self.__order > 0:
			dXdP: typing.Final[torch.Tensor] = torch.autograd.grad(
				full_mat_0.reshape(*full_mat_0.shape, 1),
				p,
				torch.ones((1, *r.shape[:-1], 1), dtype=full_mat_0.dtype, device=full_mat_0.device),
				create_graph=self.__order > 1,
				is_grads_batched=True,
				materialize_grads=True
			)[0][0, ...] # same shape as p
			D: torch.Tensor
			if self.__order == 1:
				D = self.__model.coupling(x).to(torch.cdouble)
			else:
				D, G = self.__model.coupling_and_second_order_coupling(x)
				D = D.to(torch.cdouble)
				G = G.to(torch.cdouble)
				result += -constant.HBAR ** 2 / 8.0 * (torch.autograd.grad(
					dXdP,
					p,
					torch.ones_like(dXdP),
					materialize_grads=True
				)[0][..., torch.newaxis, torch.newaxis] * (G @ self.__factors + (G @ self.__factors).conj().mT + 2.0 * D @ self.__factors @ D)).sum(-3)
			result += constant.HBAR / 2.0 * 1.0j * (dXdP[..., torch.newaxis, torch.newaxis] * (self.__factors @ D + D @ self.__factors)).sum(-3)
		return result.detach()
