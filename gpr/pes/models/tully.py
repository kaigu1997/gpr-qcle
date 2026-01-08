r"""pes.model.tully
===================
Models from John C Tully's paper "Molecular dynamics with electronic transitions"
"""
import typing

import torch

import constant

from . import base

torch.set_default_dtype(constant.DTYPE)
torch.set_default_device(constant.DEVICE)


@typing.final
class SAC(base.ModelBase):
	r"""Single Avoided Crossing, John Tully's first model
	"""
	__SAC_A: typing.Final = 0.01
	__SAC_B: typing.Final = 1.6
	__SAC_C: typing.Final = 0.005
	__SAC_D: typing.Final = 1.0

	@staticmethod
	@typing.final
	def num_pes():
		return 2

	@staticmethod
	@typing.final
	def dim():
		return 1

	@classmethod
	def max_potential(cls) -> float:
		return cls.__SAC_A

	@classmethod
	def _potential(cls, x: torch.Tensor) -> torch.Tensor:
		shape: typing.Final[tuple] = x.shape[:-1] if x.ndim > 1 else ()
		v00: torch.Tensor = torch.sign(x[..., 0]) * cls.__SAC_A * (1.0 - torch.exp(-cls.__SAC_B * torch.abs(x[..., 0])))
		v01: torch.Tensor = cls.__SAC_C * torch.exp(-cls.__SAC_D * torch.square(x[..., 0]))
		return torch.stack([v00, v01, v01, -v00], -1).reshape(shape + (cls.num_pes(), cls.num_pes()))


@typing.final
class DAC(base.ModelBase):
	r"""Dual Avoided Crossing, John Tully's second model

	This model would better use dt<=1.0au and dx<=0.05au
	"""
	__DAC_A: typing.Final = 0.10
	__DAC_B: typing.Final = 0.28
	__DAC_C: typing.Final = 0.015
	__DAC_D: typing.Final = 0.06
	__DAC_E: typing.Final = 0.05

	@staticmethod
	@typing.final
	def num_pes():
		return 2

	@staticmethod
	@typing.final
	def dim():
		return 1

	@classmethod
	def max_potential(cls) -> float:
		return cls.__DAC_E

	@classmethod
	def _potential(cls, x: torch.Tensor) -> torch.Tensor:
		shape: typing.Final[tuple] = x.shape[:-1] if x.ndim > 1 else ()
		v01: torch.Tensor = cls.__DAC_C * torch.exp(-cls.__DAC_D * torch.square(x[..., 0]))
		return torch.stack([torch.zeros(shape), v01, v01, cls.__DAC_E - cls.__DAC_A * torch.exp(-cls.__DAC_B * torch.square(x[..., 0]))], -1).reshape(shape + (cls.num_pes(), cls.num_pes()))


@typing.final
class ECR(base.ModelBase):
	r"""Extended Coupling with Reflection, John Tully's third model
	"""
	__ECR_A: typing.Final = 6e-4
	__ECR_B: typing.Final = 0.10
	__ECR_C: typing.Final = 0.90

	@staticmethod
	@typing.final
	def num_pes():
		return 2

	@staticmethod
	@typing.final
	def dim():
		return 1

	@classmethod
	def max_potential(cls) -> float:
		return 2.0 * cls.__ECR_B

	@classmethod
	def _potential(cls, x: torch.Tensor) -> torch.Tensor:
		shape: typing.Final[tuple] = x.shape[:-1] if x.ndim > 1 else ()
		v01: torch.Tensor = cls.__ECR_B * (1.0 - torch.sign(x[..., 0]) * (torch.exp(-cls.__ECR_C * torch.abs(x[..., 0])) - 1.0))
		return torch.stack([torch.full(shape, cls.__ECR_A), v01, v01, torch.full(shape, -cls.__ECR_A)], -1).reshape(shape + (cls.num_pes(), cls.num_pes()))
