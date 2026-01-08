r"""pes.model.const_model
=========================
An adiabatic model that only has
constant potential energy surfaces
and constant non-adiabatic coupling.
"""
import typing

import torch

import constant

from . import base

torch.set_default_dtype(constant.DTYPE)
torch.set_default_device(constant.DEVICE)


@typing.final
class CONST(base.ModelBase):
	r"""Constant energy and NAC
	"""
	__CST_C: typing.Final = 0.0
	__CST_D: typing.Final = 1.0
	__CST_E: typing.Final = 0.05

	@staticmethod
	def num_pes():
		return 2

	@staticmethod
	def dim():
		return 1

	@classmethod
	def max_potential(cls) -> float:
		return cls.__CST_E

	@classmethod
	def _potential(cls, x: torch.Tensor) -> torch.Tensor:
		shape: typing.Final[tuple] = x.shape[:-1] if x.ndim > 1 else ()
		x_in_func: typing.Final[torch.Tensor] = cls.__CST_D * x + cls.__CST_C
		cos_x: typing.Final[torch.Tensor] = torch.cos(x_in_func)
		sin_x: typing.Final[torch.Tensor] = torch.sin(x_in_func)
		return cls.__CST_E * torch.stack([sin_x ** 2, cos_x * sin_x, cos_x * sin_x, cos_x ** 2], -1).reshape(shape + (cls.num_pes(), cls.num_pes()))

	@classmethod
	def adiabatic_potential(cls, x: torch.Tensor) -> torch.Tensor:
		shape: typing.Final[tuple] = x.shape[:-1] if x.ndim > 1 else ()
		return cls.__CST_E * torch.stack([torch.zeros(shape), torch.ones(shape)], -1).reshape(shape + (cls.num_pes(),)) + (x[torch.isfinite(x)] * 0).sum()

	@classmethod
	def diabatic_to_adiabatic(cls, x: torch.Tensor) -> torch.Tensor:
		shape: typing.Final[tuple] = x.shape[:-1] if x.ndim > 1 else ()
		x_in_func: typing.Final[torch.Tensor] = cls.__CST_D * x + cls.__CST_C
		cos_x: typing.Final[torch.Tensor] = torch.cos(x_in_func)
		sin_x: typing.Final[torch.Tensor] = torch.sin(x_in_func)
		return torch.stack([cos_x, sin_x, -sin_x, cos_x], -1).reshape(shape + (cls.num_pes(), cls.num_pes()))

	@classmethod
	def coupling(cls, x: torch.Tensor) -> torch.Tensor:
		shape: typing.Final[tuple] = x.shape[:-1] if x.ndim > 1 else ()
		return cls.__CST_D * torch.stack([torch.zeros(shape), torch.ones(shape), -torch.ones(shape), torch.zeros(shape)], -1).reshape(shape + (cls.dim(), cls.num_pes(), cls.num_pes())) + (x[torch.isfinite(x)] * 0).sum()

	@classmethod
	def adiabatic_force(cls, x: torch.Tensor) -> torch.Tensor:
		shape: typing.Final[tuple] = x.shape[:-1] if x.ndim > 1 else ()
		return -cls.__CST_D * cls.__CST_E * torch.stack([torch.zeros(shape), torch.ones(shape), torch.ones(shape), torch.zeros(shape)], -1).reshape(shape + (cls.dim(), cls.num_pes(), cls.num_pes())) + (x[torch.isfinite(x)] * 0).sum()
