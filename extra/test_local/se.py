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


def loocv_se(
	cov: gpytorch.kernels.Kernel,
	x: torch.Tensor,
	y: torch.Tensor,
	x_all: torch.Tensor,
	y_all: torch.Tensor,
	**kwargs
) -> torch.Tensor:
	cov_mat: torch.Tensor = cov(x).to_dense()
	return ((gp.square_solver(cov_mat, y) / torch.diagonal(gp.square_solver(cov_mat, torch.eye(x.shape[-2])), 0, -2, -1)) ** 2).sum() + ((gp.default_predict(cov, x, y, x_all) - y_all) ** 2).sum()


def approx_se(
	cov: gpytorch.kernels.Kernel,
	x: torch.Tensor,
	y: torch.Tensor,
	pred: gp.ApproxPredType,
	x_all: torch.Tensor,
	y_all: torch.Tensor,
	**kwargs
) -> torch.Tensor:
	return ((y_all - pred(cov, x, y, x_all, x_all, y_all, **kwargs)) ** 2).sum()


def pinv_pred(
	cov: gpytorch.kernels.Kernel,
	x: torch.Tensor,
	y: torch.Tensor,
	x_test: torch.Tensor,
	x_all: torch.Tensor,
	y_all: torch.Tensor,
	**kwargs
) -> torch.Tensor:
	return (cov(x_test, x) @ gp.preconditioned_lstsq_solve(cov(x_all, x).to_dense(), y_all[..., None])).to_dense()[..., 0]


def se_trials(pts: sample.data, pc: plot.PlotConstants) -> None:
	for erf, pred in zip([loocv_se, approx_se], [None, pinv_pred]):
		print("gpytorch GPR using RBF kernel and {}".format(erf.__name__.replace('_', ' ')), flush=True)
		model: gp.GPR | None = opt.gpytorch_train(
			pts.x_t,
			pts.y_t,
			gpytorch.kernels.RBFKernel(sample.distribution.DIM),
			erf,
			initial_values=[torch.ones(sample.distribution.DIM)],
			x_all=pts.x_all_t,
			y_all=pts.y_all_t,
			pred=pred
		)
		if model is not None:
			plot.plot(
				gp.gpytorch_gpr(model, pts, predictor=pred, x_all=pts.x_all_t, y_all=pts.y_all_t),
				pc,
				pts,
				erf.__name__
			)
