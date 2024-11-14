import os
import sys

import gpytorch
import torch

sys.path.append(os.path.dirname(__file__))

import gp
import opt
import plot
import sample

torch.set_default_dtype(torch.float64)


def se_generator(pred: gp.PredType[gpytorch.kernels.Kernel], x_all: torch.Tensor, y_all: torch.Tensor) -> opt.LossFuncType:
	def se(
		cov: gpytorch.kernels.Kernel,
		x: torch.Tensor,
		y: torch.Tensor,
	) -> torch.Tensor:
		return ((y_all - pred(cov, x, y, x_all)) ** 2).sum()
	return se


def get_kinvy_and_kinv_diag(
	cov: gpytorch.kernels.Kernel,
	x: torch.Tensor,
	y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	cov_mat: torch.Tensor = cov(x).to_dense()
	cov_mat = (cov_mat + cov_mat.mT) / 2.0
	return (gp.square_solver(cov_mat, y), torch.diagonal(gp.square_solver(cov_mat, torch.eye(x.shape[-2])), 0, -2, -1))


def loocv_generator(over_var: bool = False) -> opt.LossFuncType:
	def loocv(
		cov: gpytorch.kernels.Kernel,
		x: torch.Tensor,
		y: torch.Tensor,
	) -> torch.Tensor:
		kinv_y: torch.Tensor
		kinv_diag: torch.Tensor
		kinv_y, kinv_diag = get_kinvy_and_kinv_diag(cov, x, y)
		if over_var:
			return (kinv_y ** 2 / kinv_diag.abs()).sum()
		else:
			return ((kinv_y / kinv_diag) ** 2).sum()
	return loocv


def lool_generator(coeff: float = -1.0) -> opt.LossFuncType:
	def lool(
		cov: gpytorch.kernels.Kernel,
		x: torch.Tensor,
		y: torch.Tensor,
	) -> torch.Tensor:
		kinv_y: torch.Tensor
		kinv_diag: torch.Tensor
		kinv_y, kinv_diag = get_kinvy_and_kinv_diag(cov, x, y)
		return (kinv_y ** 2 / kinv_diag.abs()).sum() + coeff * kinv_diag.square().log().sum() / 2.0 # == log(kinv).sum(), incase kinv is negative
	return lool


def loosvl_generator(coeff: float = -1.0) -> opt.LossFuncType:
	def loosvl(
		cov: gpytorch.kernels.Kernel,
		x: torch.Tensor,
		y: torch.Tensor,
	) -> torch.Tensor:
		kinv_y: torch.Tensor
		kinv_diag: torch.Tensor
		kinv_y, kinv_diag = get_kinvy_and_kinv_diag(cov, x, y)
		return kinv_y.square().sum() + coeff * kinv_diag.square().log().sum() / 2.0 # == log(kinv).sum(), incase kinv is negative
	return loosvl


def looph_generator(coeff: float = -1.0, delta: float = 3.0) -> opt.LossFuncType:
	delta_square: float = delta ** 2
	def looph(
		cov: gpytorch.kernels.Kernel,
		x: torch.Tensor,
		y: torch.Tensor,
	) -> torch.Tensor:
		kinv_y: torch.Tensor
		kinv_diag: torch.Tensor
		kinv_y, kinv_diag = get_kinvy_and_kinv_diag(cov, x, y)
		return 2.0 * delta_square * (torch.sqrt(1.0 + kinv_y ** 2 / delta_square / kinv_diag.abs()) - 1.0).sum() + coeff * kinv_diag.square().log().sum() / 2.0
	return looph


def loocv_se_generator(x_all: torch.Tensor, y_all: torch.Tensor, loss: opt.LossFuncType = loocv_generator()) -> opt.LossFuncType:
	def loocv_se(
		cov: gpytorch.kernels.Kernel,
		x: torch.Tensor,
		y: torch.Tensor,
	) -> torch.Tensor:
		return loss(cov, x, y) + se_generator(gp.default_predict, x_all, y_all)(cov, x, y)
	return loocv_se


def pinv_pred_generator(x_all: torch.Tensor, y_all: torch.Tensor) -> gp.PredType[gpytorch.kernels.Kernel]:
	def pinv_pred(
		cov: gpytorch.kernels.Kernel,
		x: torch.Tensor,
		y: torch.Tensor,
		x_test: torch.Tensor
	) -> torch.Tensor:
		return (cov(x_test, x) @ gp.preconditioned_lstsq_solve(cov(x_all, x).to_dense(), y_all[..., None])).to_dense()[..., 0]
	return pinv_pred


def se_trials(pts: sample.data, pc: plot.PlotConstants) -> None:
	pinv_pred: gp.PredType[gpytorch.kernels.Kernel] = pinv_pred_generator(pts.x_all_t, pts.y_all_t)
	erf: opt.LossFuncType
	name: str
	pred: gp.PredType[gpytorch.kernels.Kernel]
	for erf, name, pred in zip(
		[loocv_se_generator(pts.x_all_t, pts.y_all_t), se_generator(pinv_pred, pts.x_all_t, pts.y_all_t)],
		["loocv_se", "pinv_se"],
		[gp.default_predict, pinv_pred]
	):
		print(f"gpytorch GPR using RBF kernel and {name}", flush=True)
		model: gp.GPR | None = opt.gpytorch_train(
			pts.x_t,
			pts.y_t,
			gpytorch.kernels.RBFKernel(sample.distribution.DIM),
			erf,
			initial_values=[torch.ones(sample.distribution.DIM)],
		)
		if model is not None:
			plot.plot(
				gp.gpytorch_gpr(model.cov, pts, predictor=pred),
				pc,
				pts,
				name
			)
