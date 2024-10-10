import math
import os
import sys
import traceback
import typing

import gpytorch
import gpytorch.constraints
import numpy as np
import numpy.typing as npt
import torch

sys.path.append(os.path.dirname(__file__))

import sample

torch.set_default_dtype(torch.float64)


class NoConstraint(gpytorch.constraints.Interval):
	def __init__(self, initial_value=None):
		super().__init__(
			lower_bound=-math.inf,
			upper_bound=math.inf,
			transform=lambda x: x,
			inv_transform=lambda x: x,
			initial_value=initial_value,
		)

	def __repr__(self) -> str:
		return __class__.__name__ + "()"

	def transform(self, tensor: torch.Tensor) -> torch.Tensor:
		return tensor

	def inverse_transform(self, transformed_tensor: torch.Tensor) -> torch.Tensor:
		return transformed_tensor


class GPR(gpytorch.models.ExactGP):
	COV_NAME = "cov"
	__slots__ = ("__mean", "__cov")
	def __init__(self, x: torch.Tensor, y: torch.Tensor, likelihood: gpytorch.likelihoods.Likelihood, kernel: gpytorch.kernels.Kernel):
		super().__init__(x, y, likelihood)
		self.__mean: gpytorch.means.Mean = gpytorch.means.ZeroMean()
		self.__cov: gpytorch.kernels.Kernel = kernel

	@property
	def cov(self) -> gpytorch.kernels.Kernel:
		return self.__cov

	def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
		Mean = self.__mean(x)
		assert isinstance(Mean, torch.Tensor)
		return gpytorch.distributions.MultivariateNormal(Mean, self.cov(x))


def square_solver(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
	val, vec = torch.linalg.eigh(A) # A = vec @ val.diag() @ vec.T
	return vec @ (1.0 / val).diag_embed() @ vec.mH @ B


def preconditioned_lstsq_solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
	_, s, vh = torch.linalg.svd(A) # A: m-n, _: m-m, s: n, vh: n-n
	q = vh.mH @ (1.0 / s.abs()).diag_embed()
	return q @ torch.linalg.lstsq(A @ q, B, rcond=0.0, driver="gels").solution


def default_predict(
	cov: gpytorch.kernels.Kernel,
	x: torch.Tensor,
	y: torch.Tensor,
	x_test: torch.Tensor
) -> torch.Tensor:
	return (cov(x_test, x) @ square_solver(cov(x).to_dense(), y)).to_dense()


class PredType(typing.Protocol):
	def __call__(
		self,
		cov: gpytorch.kernels.Kernel,
		x: torch.Tensor,
		y: torch.Tensor,
		x_test: torch.Tensor,
		**kwargs: typing.Any
	) -> torch.Tensor:
		...


class ApproxPredType(typing.Protocol):
	def __call__(
		self,
		cov: gpytorch.kernels.Kernel,
		x: torch.Tensor,
		y: torch.Tensor,
		x_test: torch.Tensor,
		x_all: torch.Tensor,
		y_all: torch.Tensor,
		**kwargs: typing.Any
	) -> torch.Tensor:
		...


def gpytorch_gpr(model: GPR, pts: sample.data, x_test: torch.Tensor | None = None, predictor: PredType | None = None, **kwargs) -> npt.NDArray[np.double]:
	if x_test is None:
		x_test = pts.x_test_t
	try:
		result: torch.Tensor = predictor(model.cov, pts.x_t, pts.y_t, x_test, **kwargs) if predictor is not None else default_predict(model.cov, pts.x_t, pts.y_t, x_test)
		return result.detach().numpy().reshape(sample.distribution.N_GRIDS, sample.distribution.N_GRIDS)
	except Exception as e:
		traceback.print_exception(e)
		return np.zeros((sample.distribution.N_GRIDS, sample.distribution.N_GRIDS), np.double)
