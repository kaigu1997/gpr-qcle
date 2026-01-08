r"""pes.model.base
==================
The base class for all models
"""
import abc
import collections.abc
import dataclasses
import functools
import inspect
import typing

import torch

import constant

torch.set_default_dtype(constant.DTYPE)
torch.set_default_device(constant.DEVICE)


class ModelBase(abc.ABC):
	r"""The basis of all models

	Attributes
	----------
	coupling : collections.abc.Callable[[torch.Tensor], torch.Tensor] | None
		Non-adiabatic coupling. None if the model is diabatic

	Methods
	----------
	num_pes()
		The number of potential energy surfaces
	dim()
		The dimension of classical degree of freedom
	max_potential()
		To get the maximum value the potential could reach
	coupling_generator()
		To generate the coupling function
	has_diabatic()
		Whether the model is diabatic or adiabatic only
	"""

	@staticmethod
	@abc.abstractmethod
	def num_pes() -> typing.Literal[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
		r"""The number of potential energy surfaces

		Returns
		-------
		int
			The number of potential energy surfaces
		"""

	@staticmethod
	@abc.abstractmethod
	def dim() -> typing.Literal[1,2,3,4,5,6,7,8,9,10]:
		r"""The dimension of classical degree of freedom

		Returns
		-------
		int
			The dimension of classical degree of freedom
		"""

	@classmethod
	@abc.abstractmethod
	def max_potential(cls) -> float:
		r"""To get the maximum value the potential could reach

		Returns
		-------
		float
			Maximum potential
		"""

	@classmethod
	@abc.abstractmethod
	def _potential(cls, x: torch.Tensor) -> torch.Tensor:
		r"""To get the diabatic potential at all given grids

		Parameters
		----------
		x : torch.Tensor, dtype of `torch.double`, shape of (..., DIM)
			Positions of interest

		Returns
		-------
		torch.Tensor, dtype of `torch.double`, shape of (..., NUM_PES, NUM_PES)
			Potential at all given grids
		"""

	@classmethod
	def adiabatic_potential(cls, x: torch.Tensor) -> torch.Tensor:
		r"""To get the adiabatic potential at all given grids

		Parameters
		----------
		x : torch.Tensor, dtype of `torch.double`, shape of (..., DIM)
			Positions of interest

		Returns
		-------
		torch.Tensor, dtype of `torch.double`, shape of (..., NUM_PES, NUM_PES)
			Adiabatic potential at all coordinate grids

		Raises
		------
		NotImplementedError
			This function is optional. Derived class may not implement it and the error will be raise once called in that case.
		"""
		raise NotImplementedError(f"{cls.__name__} should implement {inspect.stack()[0][3]} for calling.")

	@classmethod
	def diabatic_to_adiabatic(cls, x: torch.Tensor) -> torch.Tensor:
		"""To get the basis transform matrix coupling at all given grids

		Parameters
		----------
		x : torch.Tensor, dtype of `torch.double`, shape of (..., DIM)
			Positions of interest

		Returns
		-------
		torch.Tensor, dtype of `torch.double`, shape of (..., DIM, NUM_PES, NUM_PES)
			Non-adiabatic coupling at all given grids, of each dimension

		Raises
		------
		NotImplementedError
			This function is optional. Derived class may not implement it and the error will be raise once called in that case.
		"""
		raise NotImplementedError(f"{cls.__name__} should implement {inspect.stack()[0][3]} for calling.")

	@classmethod
	def coupling(cls, x: torch.Tensor) -> torch.Tensor:
		"""To get the non-adiabatic coupling at all given grids

		Parameters
		----------
		x : torch.Tensor, dtype of `torch.double`, shape of (..., DIM)
			Positions of interest

		Returns
		-------
		torch.Tensor, dtype of `torch.double`, shape of (..., DIM, NUM_PES, NUM_PES)
			Non-adiabatic coupling at all given grids, of each dimension

		Raises
		------
		NotImplementedError
			This function is optional. Derived class may not implement it and the error will be raise once called in that case.
		"""
		raise NotImplementedError(f"{cls.__name__} should implement {inspect.stack()[0][3]} for calling.")

	@classmethod
	def adiabatic_force(cls, x: torch.Tensor) -> torch.Tensor:
		"""To get the Hellmann-Feynmann forces at all given grids

		Parameters
		----------
		x : torch.Tensor, dtype of `torch.double`, shape of (..., DIM)
			Positions of interest

		Returns
		-------
		torch.Tensor, dtype of `torch.double`, shape of (..., DIM, NUM_PES, NUM_PES)
			Force at all given grids, of each dimension

		Raises
		------
		NotImplementedError
			This function is optional. Derived class may not implement it and the error will be raise once called in that case.
		"""
		raise NotImplementedError(f"{cls.__name__} should implement {inspect.stack()[0][3]} for calling.")

	@classmethod
	def second_order_coupling(cls, x: torch.Tensor) -> torch.Tensor:
		"""To get the second order non-adiabatic coupling  at all given grids

		Parameters
		----------
		x : torch.Tensor, dtype of `torch.double`, shape of (..., DIM)
			Positions of interest

		Returns
		-------
		torch.Tensor, dtype of `torch.double`, shape of (..., DIM, NUM_PES, NUM_PES)
			Second order coupling at all given grids, of each dimension

		Raises
		------
		NotImplementedError
			This function is optional. Derived class may not implement it and the error will be raise once called in that case.
		"""
		raise NotImplementedError(f"{cls.__name__} should implement {inspect.stack()[0][3]} for calling.")

	def __new__(cls, x: torch.Tensor) -> torch.Tensor:
		r"""The diabatic potential energy

		Parameters
		----------
		x : torch.Tensor, dtype of `torch.double`, shape of (..., DIM)
			Positions of interest

		Returns
		-------
		torch.Tensor, dtype of `torch.double`, shape of (..., NUM_PES, NUM_PES)
			Potential at all coordinate grids
		"""
		assert x.shape[-1] == cls.dim()
		return cls._potential(x)


@typing.overload
def model_validation[**P, T](func: collections.abc.Callable[P, T]) -> collections.abc.Callable[P, T]:
	...


@typing.overload
def model_validation[**P, T](func: None = None, *, model_param_name: str = "model") -> collections.abc.Callable[[collections.abc.Callable[P, T]], collections.abc.Callable[P, T]]:
	...


def model_validation[**P, T](func: collections.abc.Callable[P, T] | None = None, model_param_name: str = "model") -> collections.abc.Callable[P, T] | collections.abc.Callable[[collections.abc.Callable[P, T]], collections.abc.Callable[P, T]]:
	r"""To validate that the model is not virtual

	Parameters
	----------
	func : collections.abc.Callable[P, T]
		The function containing model parameter
	model_param_name : str, optional
		The name of the model parameter, by default "model"

	Returns
	-------
	collections.abc.Callable[[collections.abc.Callable[P, T]], collections.abc.Callable[P, T]]
		A decorator, which takes a function and returns a wrapped function
	"""
	def decorator(func: collections.abc.Callable[P, T]) -> collections.abc.Callable[P, T]:
		r"""The decorator

		Parameters
		----------
		func : collections.abc.Callable[P, T]
			The function containing model parameter

		Returns
		-------
		collections.abc.Callable[P, T]
			Decorated function

		Raises
		------
		TypeError
			In case the model does not pass validation (that the model is virtual)
		"""
		@functools.wraps(func)
		def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
			r"""The wrapper

			Returns
			-------
			T
				Return type of the original function

			Raises
			------
			TypeError
				In case the model does not pass validation (that the model is virtual)
			"""
			sig = inspect.signature(func)
			bound_args = sig.bind(*args, **kwargs)
			bound_args.apply_defaults()
			if model_param_name in bound_args.arguments:
				value = bound_args.arguments[model_param_name]
				if inspect.isabstract(value):
					raise TypeError(f"Parameter '{model_param_name}' of type {value.__name__} MUST NOT BE virtual")
			return func(*args, **kwargs)
		return wrapper

	if func is not None and callable(func):
		return decorator(func)
	return decorator


@typing.final
@dataclasses.dataclass(init=False, slots=True)
class ModelConfig:
	"""The configuration of the model

	Parameters
	----------
	model : _ModelProtocal
		In fact it should be `ModelBase`, but this is defined later

	Attributes
	----------
	NUM_PES : int
		The number of potential energy surfaces
	DIM : int
		The dimension of classical degree of freedom
	NUM_ELM : int
		The number of elements in partial Wigner transformed phase space representation, square of `NUM_PES`
	PHASEDIM : int
		The number of dimension in phase space, twice of `DIM`
	DIM_RANGE : tuple[int, ...]
		`range(DIM)`
	PES_RANGE : tuple[int, ...]
		`range(PES)`
	P_DIM_RANGE : tuple[int, ...]
		`range(DIM, PHASEDIM)`, indices for p in phase space
	PHASEDIM_RANGE : tuple[int, ...]
		Combination of `DIM_RANGE` and `P_DIM_RANGE`
	DIM1_EINSUM_INDEX : typing.LiteralString
		First DIM dimensions of x coordinates
	DIM2_EINSUM_INDEX : typing.LiteralString]
		First DIM dimensions of p coordinates
	WIGNER_DIM_EINSUM_INDEX : typing.LiteralString]
		First DIM dimensions of y_coord
	PES1_EINSUM_INDEX : typing.LiteralString]
		Row Index of potential energy surface
	PES2_EINSUM_INDEX : typing.LiteralString]
		Column Index of potential energy surface
	DIM_EINSUM_INDEX : typing.LiteralString]
		Last dimension of x/p coordinates
	SUM_DIMS_FOR_PWTDM_MARGINAL : list[tuple[int, ...]]]
		For each element, it is a tuple containing all values in `PHASEDIM_RANGE` except for index of itself
	"""
	NAME: typing.Final
	NUM_PES: typing.Final
	DIM: typing.Final
	NUM_ELM: typing.Final
	NUM_TRIG: typing.Final
	PHASEDIM: typing.Final
	DIM_RANGE: typing.Final[tuple[int, ...]]
	PES_RANGE: typing.Final[tuple[int, ...]]
	ELEMENT_RANGE: typing.Final[tuple[int, ...]]
	TRIG_RANGE: typing.Final[tuple[int, ...]]
	P_DIM_RANGE: typing.Final[tuple[int, ...]]
	PHASEDIM_RANGE: typing.Final[tuple[int, ...]]
	LAST_DIM_RANGE: typing.Final[tuple[int, ...]]
	TRIL_ROW_INDICES: typing.Final[tuple[int, ...]]
	TRIL_COL_INDICES: typing.Final[tuple[int, ...]]
	TRIL_ELEMENT_INDICES: typing.Final[tuple[int, ...]]
	FLATTEN_TRIL_INDEX: typing.Final[tuple[int, ...]]
	SUM_DIMS_FOR_PWTDM_MARGINAL: typing.Final[list[tuple[int, ...]]]

	def __init__(self, model: type[ModelBase]) -> None:
		self.NAME = model.__name__.lower()
		self.NUM_PES = model.num_pes()
		self.DIM = model.dim()
		self.NUM_ELM = self.NUM_PES ** 2
		self.NUM_TRIG = self.NUM_PES * (self.NUM_PES + 1) // 2
		self.PHASEDIM = self.DIM * 2
		self.DIM_RANGE = tuple(range(self.DIM))
		self.PES_RANGE = tuple(range(self.NUM_PES))
		self.ELEMENT_RANGE = tuple(range(self.NUM_ELM))
		self.TRIG_RANGE = tuple(range(self.NUM_TRIG))
		self.P_DIM_RANGE = tuple(range(self.DIM, self.DIM * 2))
		self.PHASEDIM_RANGE = self.DIM_RANGE + self.P_DIM_RANGE
		self.LAST_DIM_RANGE = tuple(range(-self.DIM, 0))
		tril_indices = torch.tril_indices(self.NUM_PES, self.NUM_PES)
		self.TRIL_ROW_INDICES = tuple(tril_indices[0].tolist())
		self.TRIL_COL_INDICES = tuple(tril_indices[1].tolist())
		self.TRIL_ELEMENT_INDICES = tuple(r * self.NUM_PES + c for r, c in zip(self.TRIL_ROW_INDICES, self.TRIL_COL_INDICES))
		flatten_lower_trig: torch.Tensor = torch.zeros(self.NUM_PES, self.NUM_PES, dtype=torch.int)
		flatten_lower_trig[self.TRIL_ROW_INDICES, self.TRIL_COL_INDICES] = torch.arange(self.NUM_TRIG, dtype=torch.int)
		self.FLATTEN_TRIL_INDEX = tuple((flatten_lower_trig + torch.tril(flatten_lower_trig, -1).T).flatten().tolist())
		self.SUM_DIMS_FOR_PWTDM_MARGINAL = [tuple(jDim for jDim in self.PHASEDIM_RANGE if jDim != iDim) for iDim in self.PHASEDIM_RANGE]
