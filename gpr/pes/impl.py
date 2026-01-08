r"""pes.impl
============
This is the implementaion of the diabatic potential
"""
import collections.abc
import math
import typing

import torch

import constant

from . import models

torch.set_default_dtype(constant.DTYPE)
torch.set_default_device(constant.DEVICE)


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
	# comment for layout conflict, may uncomment for static check
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

	def __C_and_E(
		self,
		V: torch.Tensor,
		*,
		diabatic_to_adiabatic: bool,
		adiabatic_potential: bool
	) -> tuple[torch.Tensor | None, torch.Tensor | None]:
		r"""To get the the basis transformation matrices and/or the adiabatic potential, if needed

		Parameters
		----------
		V : torch.Tensor
			Diabatic potential
		diabatic_to_adiabatic : bool
			To calculate diabatic to adiabatic basis transformation matrix
		adiabatic_potential : bool
			To calculate adiabatic potential

		Returns
		-------
		tuple[torch.Tensor | None, torch.Tensor | None]
			Basis transformation matrices, adiabatic potential
		"""
		C: torch.Tensor | None = None
		E: torch.Tensor | None = None
		match self.__config.NUM_PES:
			case 1:
				if diabatic_to_adiabatic:
					C = torch.eye(self.__config.NUM_PES)
				if adiabatic_potential:
					E = V
			case 2:
				V00: typing.Final[torch.Tensor] = V[..., 0, 0] # ...
				V01: typing.Final[torch.Tensor] = V[..., 0, 1] # ...
				V11: typing.Final[torch.Tensor] = V[..., 1, 1] # ...
				diff: typing.Final[torch.Tensor] = V00 - V11 # ...
				V01_double: typing.Final[torch.Tensor] = 2.0 * V01 # ...
				discriminant: typing.Final[torch.Tensor] = torch.sqrt(torch.square(diff) + torch.square(V01_double)) # ...
				if diabatic_to_adiabatic:
					discriminant_absdiff: typing.Final[torch.Tensor] = discriminant + torch.abs(diff) # ...
					C_unnormalized: typing.Final[torch.Tensor] = torch.where(diff[..., torch.newaxis] < 0, torch.stack([-discriminant_absdiff, V01_double, V01_double, discriminant_absdiff], -1), torch.stack([-V01_double, discriminant_absdiff, discriminant_absdiff, V01_double], -1)).reshape_as(V)
					C = C_unnormalized / torch.linalg.norm(C_unnormalized, axis=-2, keepdims=True)
				if adiabatic_potential:
					E = ((V00 + V11)[..., torch.newaxis] + torch.tensor([-1.0, 1.0]) * discriminant[..., torch.newaxis]) / 2.0
			case _:
				if diabatic_to_adiabatic:
					C = torch.linalg.eigh(V.cpu()).eigenvectors.to(torch.get_default_device())
				if adiabatic_potential:
					E = torch.linalg.eigvalsh(V.cpu()).to(torch.get_default_device()) # pylint: disable=not-callable
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
	) -> dict[str, torch.Tensor]:
		r"""To get the selected quantities for all inputs

		Returns
		-------
		torch.Tensor, dtype of `torch.double`, shape of (N_GRIDS, ..., NUM_PES, NUM_PES)
			Diabatic potential at all coordinate grids

		Parameters
		----------
		x : torch.Tensor
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
		dict[str, torch.Tensor]
			Keys are the same as the parameters, and the values are the corresponding quantities
		"""
		def is_implemented_since[T](derived_cls: type[T], method_name: str, base_cls: type[T]) -> bool:
			"""Check if `derived_cls` or any of its ancestors (up to but not including `base_cls`) has implemented method_name.

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

		assert any((diabatic_potential, diabatic_to_adiabatic, adiabatic_potential, coupling, second_order_coupling, force))
		x = x.detach().requires_grad_()
		result: typing.Final[dict[str, torch.Tensor]] = {}
		# who need D: G
		need_manual: typing.Final[collections.abc.Callable[[bool, str], bool]] = lambda need, method: need and not is_implemented_since(self.__model, method, models.ModelBase)
		need_manual_G: typing.Final[bool] = need_manual(second_order_coupling, "second_order_coupling")
		need_D: typing.Final[bool] = coupling or need_manual_G
		need_manual_D: typing.Final[bool] = need_manual(need_D, "coupling")
		# who need adiaV: F
		need_manual_F: typing.Final[bool] = need_manual(force, "adiabatic_force")
		need_E: typing.Final[bool] = adiabatic_potential or need_manual_F
		need_manual_E: typing.Final[bool] = need_manual(need_E, "adiabatic_potential")
		# who need C: D, F
		need_C: typing.Final[bool] = diabatic_to_adiabatic or need_manual_D or need_manual_F
		need_manual_C: typing.Final[bool] = need_manual(need_C, "diabatic_to_adiabatic")
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
		if diabatic_potential:
			assert V is not None
			result["diabatic_potential"] = V
		C: torch.Tensor | None = None
		E: torch.Tensor | None = None
		if need_C and not need_manual_C:
			C = self.__model.diabatic_to_adiabatic(x)
		if need_E and not need_manual_E:
			E = self.__model.adiabatic_potential(x)
		if need_manual_C or need_manual_E:
			assert V is not None
			C, E = self.__C_and_E(V, diabatic_to_adiabatic=need_manual_C, adiabatic_potential=need_manual_E)
		if diabatic_to_adiabatic:
			assert C is not None
			result["diabatic_to_adiabatic"] = C.detach()
		if adiabatic_potential:
			assert E is not None
			result["adiabatic_potential"] = E.detach()
		D: typing.Final[torch.Tensor | None] = (self.__D(x, C, matrix_to_supervector_shape, matrix_eye_broadcast_shape, matrix_eye_repeat_shape, deriv_result_shape) if need_manual_D else self.__model.coupling(x)) if need_D else None # type: ignore[reportArgumentType]
		if coupling:
			assert D is not None
			result["coupling"] = D.detach()
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
				result["force"] = self.__F(x, C, E, V, vector_eye_broadcast_shape, vector_eye_repeat_shape, matrix_to_supervector_shape, matrix_eye_broadcast_shape, matrix_eye_repeat_shape, deriv_result_shape).detach()
			else:
				result["force"] = self.__model.adiabatic_force(x).detach()
		if second_order_coupling:
			if need_manual_G:
				assert D is not None
				assert tensor_to_supervector_shape is not None
				assert tensor_eye_broadcast_shape is not None
				assert tensor_eye_repeat_shape is not None
				assert deriv_result_shape is not None
				result["second_order_coupling"] = self.__G(x, D, tensor_to_supervector_shape, tensor_eye_broadcast_shape, tensor_eye_repeat_shape, deriv_result_shape).detach()
			else:
				result["second_order_coupling"] = self.__model.second_order_coupling(x).detach()
		return result


@typing.final
class InitialDistribution:
	r"""To generate the initial partial Wigner-transformed density matrix

	Parameters
	----------
	r0 : torch.Tensor, shape of (PHASEDIM,)
		Initial center of each dimension
	sigma_r0 : torch.Tensor, shape of (PHASEDIM,)
		Initial standard deviation of each dimension
	initial_population : torch.Tensor, shape of (NUM_PES,)
		The population on each surface, whose squared sum is normalized. Default to all on the ground state.
	initial_phase_factor : torch.Tensor, shape of (NUM_PES,)
		The phase factor for each surface, in unit of arc.

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
	__slots__: tuple = ("__potential", "__r0", "__sigma_r0", "__weight_phase", "__factors", "__order")
	__potential: typing.Final[Potential]
	__r0: typing.Final[torch.Tensor]
	__sigma_r0: typing.Final[torch.Tensor]
	__weight_phase: typing.Final[torch.Tensor]
	__factors: typing.Final[torch.Tensor]
	__order: typing.Final[typing.Literal[0,1,2]]

	def __init__(
		self,
		potential: Potential,
		r0: torch.Tensor,
		sigma_r0: torch.Tensor,
		init_ppl_and_phase: torch.Tensor,
		*,
		diabatic: bool = False
	):
		self.__potential = potential
		self.__r0 = r0
		self.__sigma_r0 = sigma_r0
		weight_phase: typing.Final[torch.Tensor] = init_ppl_and_phase.conj()[:, torch.newaxis] * init_ppl_and_phase # shape of (NUM_PES, NUM_PES)
		self.__weight_phase = (weight_phase + weight_phase.T.conj()) / 2.0 # remove numerical error, shape of (NUM_PES, NUM_PES)
		self.__factors = self.__weight_phase / (2.0 * math.pi) ** potential.config.DIM / self.__sigma_r0.prod() # divide by normalization factor, shape of (NUM_ELM,)
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
		return self.__potential.config

	def __call__(self, r: torch.Tensor) -> torch.Tensor:
		r"""To calculate the initial density of the given element at the given phase point

		Parameters
		----------
		r : torch.Tensor, shape of (..., PHASEDIM)
			Phase space coordinates of initerest

		Returns
		-------
		torch.Tensor, shape of (..., NUM_PES, NUM_PES)
			Density of the given element
		"""
		x: typing.Final[torch.Tensor] = r[..., :self.__potential.config.DIM].detach()
		p: typing.Final[torch.Tensor] = r[..., self.__potential.config.DIM:].detach()
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
			potential_quantities: typing.Final[dict[str, torch.Tensor]] = self.__potential(x, coupling=True, second_order_coupling=self.__order > 1)
			D: typing.Final[torch.Tensor] = potential_quantities["coupling"].to(torch.cdouble)
			result += constant.HBAR / 2.0 * 1.0j * (dXdP[..., torch.newaxis, torch.newaxis] * (self.__factors @ D + D @ self.__factors)).sum(-3)
			if self.__order > 1:
				G: typing.Final[torch.Tensor] = potential_quantities["second_order_coupling"].to(torch.cdouble)
				result += -constant.HBAR ** 2 / 8.0 * (torch.autograd.grad(
					dXdP,
					p,
					torch.ones_like(dXdP),
					materialize_grads=True
				)[0][..., torch.newaxis, torch.newaxis] * (G @ self.__factors + (G @ self.__factors).conj().mT + 2.0 * D @ self.__factors @ D)).sum(-3)
		return result.detach()
