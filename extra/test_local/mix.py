import collections.abc
import contextlib
import functools
import io
import os
import sys

import gpytorch
import numpy as np
import numpy.typing as npt
import sklearn.neighbors
import torch

sys.path.append(os.path.dirname(__file__))

import gp
import local
import opt
import plot
import sample
import se


def get_all_predictions(
	covs: gpytorch.kernels.Kernel | collections.abc.Sequence[gpytorch.kernels.Kernel],
	x: torch.Tensor,
	y: torch.Tensor,
	x_test: torch.Tensor,
	predictors: collections.abc.Sequence[gp.PredType[gpytorch.kernels.Kernel]],
) -> torch.Tensor:
	results = torch.empty(x_test.shape[:-1] + (len(predictors),))
	if isinstance(covs, gpytorch.kernels.Kernel):
		for i, pred in enumerate(predictors):
			results[..., i] = pred(covs, x, y, x_test)
	else:
		assert len(covs) == results.shape[-1]
		for i, (cov, pred) in enumerate(zip(covs, predictors)):
			results[..., i] = pred(cov, x, y, x_test)
	return results


def get_best_method_idx(
	covs: gpytorch.kernels.Kernel | collections.abc.Sequence[gpytorch.kernels.Kernel],
	x: torch.Tensor,
	y: torch.Tensor,
	x_test: torch.Tensor,
	y_test: torch.Tensor,
	predictors: collections.abc.Sequence[gp.PredType[gpytorch.kernels.Kernel]],
) -> torch.Tensor:
	return torch.argmin(torch.abs(get_all_predictions(covs, x, y, x_test, predictors) - y_test[..., None]), -1)


def mix_min_se_generator(
	x_all: torch.Tensor,
	y_all: torch.Tensor,
	predictors: collections.abc.Sequence[gp.PredType[gpytorch.kernels.Kernel]]
) -> opt.LossFuncType:
	def mix_min_se(
		cov: gpytorch.kernels.Kernel,
		x: torch.Tensor,
		y: torch.Tensor
	) -> torch.Tensor:
		return torch.min(torch.square(get_all_predictions(cov, x, y, x_all, predictors) - y_all[..., None]), -1).values.sum()
	return mix_min_se


def pred_mix_generator(
	local_voters: int,
	predictors: collections.abc.Sequence[gp.PredType[gpytorch.kernels.Kernel]],
	x_all: torch.Tensor,
	best_method_idx: torch.Tensor
) -> gp.PredType[gpytorch.kernels.Kernel]:
	def mix_pred(
		cov: gpytorch.kernels.Kernel,
		x: torch.Tensor,
		y: torch.Tensor,
		x_test: torch.Tensor
	) -> torch.Tensor:
		nonlocal best_method_idx
		length: int = len(predictors)
		best_method_idx = best_method_idx.to(torch.int64)
		assert torch.all(best_method_idx >= 0).item() and torch.all(best_method_idx < length).item()
		# nn is from x but use_local is from x_all
		results: torch.Tensor = get_all_predictions(cov, x, y, x_test, predictors)
		method_weight: torch.Tensor = (best_method_idx[local.get_neighbor_ind(x_all, x_test, local_voters)][..., None] == torch.arange(length)).to(torch.double).sum(-2) / local_voters # sum over neighbors
		return (method_weight * results).sum(-1)
	return mix_pred


def model_mix_pred_generator(
	local_voters: int,
	predictors: collections.abc.Sequence[gp.PredType[gpytorch.kernels.Kernel]],
	x_all: torch.Tensor,
	best_method_idx: torch.Tensor
) -> gp.PredType[collections.abc.Sequence[gpytorch.kernels.Kernel]]:
	def model_mix_pred(
		covs: collections.abc.Sequence[gpytorch.kernels.Kernel],
		x: torch.Tensor,
		y: torch.Tensor,
		x_test: torch.Tensor
	) -> torch.Tensor:
		nonlocal best_method_idx
		assert len(covs) == len(predictors)
		length: int = len(covs)
		best_method_idx = best_method_idx.to(torch.int64)
		assert torch.all(best_method_idx >= 0).item() and torch.all(best_method_idx < length).item()
		# nn is from x but use_local is from x_all
		results: torch.Tensor = torch.empty(x_test.shape[:-1] + (length,))
		for i, (cov, pred) in enumerate(zip(covs, predictors)):
			results[..., i] = pred(cov, x, y, x_test)
			# predict based on votes of the locals
		use_local_weight: torch.Tensor = (best_method_idx[local.get_neighbor_ind(x_all, x_test, local_voters)][..., None] == torch.arange(length)).to(torch.double).sum(-2) / local_voters # sum over neighbors
		return (use_local_weight * results).sum(-1)
	return model_mix_pred


def mix_pred_trial(
	pts: sample.data,
	pc: plot.PlotConstants,
	num_neighbor: int,
	nn: sklearn.neighbors.NearestNeighbors,
	pts_neighbor_ind: torch.Tensor
) -> None:
	print("\ngpytorch GPR using RBF kernel with mixed exact GP, pseudo inverse and local GP")
	preds: list[gp.PredType[gpytorch.kernels.Kernel]] = [gp.default_predict, se.pinv_pred_generator(pts.x_all_t, pts.y_all_t), local.local_predict_generator(num_neighbor, nn, pts_neighbor_ind)]
	model: gp.GPR | None = opt.gpytorch_train(
		pts.x_t,
		pts.y_t,
		gpytorch.kernels.RBFKernel(sample.distribution.DIM),
		mix_min_se_generator(pts.x_all_t, pts.y_all_t, preds),
		initial_values=[torch.ones(sample.distribution.DIM)]
	)
	assert model is not None
	best_method_idx: torch.Tensor = get_best_method_idx(model.cov, pts.x_t, pts.y_t, pts.x_all_t, pts.y_all_t, preds)
	plot.label_scatter(
		pts.x_all,
		best_method_idx.detach().numpy(),
		[pred.__name__ if pred is not None else "" for pred in preds],
		"smallest_error_in_mix_predictors"
	)
	plot.plot(
		gp.gpytorch_gpr(
			model.cov,
			pts,
			predictor=pred_mix_generator(num_neighbor, preds, pts.x_all_t, best_method_idx)
		),
		pc,
		pts,
		"mix_methods"
	)


def model_mix_trial(
	covs: collections.abc.Sequence[gpytorch.kernels.Kernel],
	pts: sample.data,
	pc: plot.PlotConstants,
	predictors: collections.abc.Sequence[gp.PredType[gpytorch.kernels.Kernel]]
) -> None:
	print("\nWeighted average prediction of different methods")
	print("Weight of each method is the ratio of points with smallest prediction error of that method")
	best_method_idx = get_best_method_idx(covs, pts.x_t, pts.y_t, pts.x_all_t, pts.y_all_t, predictors)
	plot.label_scatter(
		pts.x_all,
		best_method_idx.detach().numpy(),
		[pred.__name__ if pred is not None else "" for pred in predictors],
		"smallest_error_in_mix_model"
	)
	for lv in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, pts.y_all_t.numel()]:
		print(f"# Local voters = {lv}")
		plot.plot(
			gp.gpytorch_gpr(
				covs,
				pts,
				predictor=model_mix_pred_generator(lv, predictors, pts.x_all_t, best_method_idx)
			),
			pc,
			pts,
			f"local_global_model_mix_{lv}"
		)


def each_local_parameter_trial(
	pts: sample.data,
	pc: plot.PlotConstants,
	pts_neighbor_ind: torch.Tensor
) -> None:
	def local_predictor(
		cov: gpytorch.kernels.Kernel,
		x: torch.Tensor,
		y: torch.Tensor,
		x_test: torch.Tensor,
		*,
		x_nn: torch.Tensor,
		y_nn: torch.Tensor
	) -> torch.Tensor:
		return local.normalize_generator(gp.default_predict)(cov, x_nn, y_nn, x_test)

	print("\nOptimize single point error for each point")
	for loss_func, param in zip([se.looph_generator(), se.looph_generator(coeff=1.0), se.lool_generator(), se.lool_generator(coeff=1.0), se.loocv_generator(), se.loocv_generator(True), se.loosvl_generator(), se.loosvl_generator(1.0)], [{}, {"coeff": 1.0}, {}, {"coeff": 1.0}, {}, {"over_var": True}, {}, {"coeff": 1.0}]):
		func_name: str = loss_func.__name__ + ("" if len(param) == 0 else "_") + str(param)[1:-1].replace("'", "").replace("\"", "").replace(": ", "=").replace(", ", "_")
		print(f"\nLoss function is {func_name}")
		covs: list[gpytorch.kernels.Kernel] = []
		preds: list[gp.PredType[gpytorch.kernels.Kernel]] = []
		names: tuple = ("error_of_loss_function", "error_on_the_point", "default_model", "pred_i_cov_i", "inv@cov-I", "cov@inv-I", "inv@cov+noise-I", "cov+noise@inv-I", "inv_lst@cov-I", "cov@inv_lst-I")
		loss: npt.NDArray[np.double] = np.empty((pts.y.size, len(names)), np.double)
		for idx in range(pts.y.size):
			with contextlib.redirect_stdout(io.StringIO()):
				x_idx: torch.Tensor = torch.cat([pts.x_t[idx].reshape(1, sample.distribution.DIM), pts.x_t[pts_neighbor_ind[idx]]])
				y_idx: torch.Tensor = torch.cat([pts.y_t[idx].reshape(1), pts.y_t[pts_neighbor_ind[idx]]])
				model: gp.GPR | None = opt.gpytorch_train(
					x_idx,
					y_idx,
					gpytorch.kernels.RBFKernel(sample.distribution.DIM),
					loss_func,
					print_log=False,
					initial_values=[torch.ones(sample.distribution.DIM)]
				)
			assert model is not None
			covs.append(model.cov)
			preds.append(
				functools.partial(
					local_predictor,
					x_nn=x_idx,
					y_nn=y_idx
				)
			)
			loss[idx, 0] = loss_func(model.cov, x_idx, y_idx).item()
			loss[idx, 1] = torch.abs(local.normalize_generator(gp.default_predict)(model.cov, pts.x_t[pts_neighbor_ind[idx]], pts.y_t[pts_neighbor_ind[idx]], pts.x_t[idx].reshape(1, sample.distribution.DIM)) - pts.y_t[idx].reshape(1)).item()
			loss[idx, 2] = torch.abs(local.normalize_generator(gp.default_predict)(model.cov, x_idx, y_idx, pts.x_t[idx].reshape(1, sample.distribution.DIM)) - pts.y_t[idx].reshape(1)).item()
			loss[idx, 3] = torch.abs(preds[idx](covs[idx], pts.x_t, pts.y_t, pts.x_t[idx].reshape(1, -1)) - pts.y_t[idx].reshape(1)).item()
			loss[idx, 4] = torch.dist(gp.square_solver(model.cov(x_idx).to_dense(), torch.eye(y_idx.numel())) @ model.cov(x_idx).to_dense(), torch.eye(y_idx.numel()))
			loss[idx, 5] = torch.dist(model.cov(x_idx).to_dense() @ gp.square_solver(model.cov(x_idx).to_dense(), torch.eye(y_idx.numel())), torch.eye(y_idx.numel()))
			cov_mat: torch.Tensor = model.cov(x_idx).to_dense()
			cov_mat = (cov_mat + cov_mat.T) / 2.0
			noise: float = (loss[idx, 4] + loss[idx, 5]) / 2.0 * 1e-4
			if loss[idx, 4] > 1e-3 or loss[idx, 5] > 1e-3:
				norm: torch.Tensor = torch.linalg.norm(cov_mat, None, 0, True)
				print(
					f"{idx}: f({opt.format_array(None, pts.x[idx])}) = {pts.y[idx]}",
					noise,
					cov_mat,
					cov_mat @ cov_mat.T / norm / norm.T,
					torch.linalg.eigvalsh(cov_mat),
					torch.linalg.eigvalsh(cov_mat + noise * torch.eye(y_idx.numel())),
					sep="\n"
				)
			loss[idx, 6] = torch.dist(gp.square_solver(model.cov(x_idx).to_dense() + noise * torch.eye(y_idx.numel()), torch.eye(y_idx.numel())) @ model.cov(x_idx).to_dense(), torch.eye(y_idx.numel()))
			loss[idx, 7] = torch.dist(model.cov(x_idx).to_dense() @ gp.square_solver(model.cov(x_idx).to_dense() + noise * torch.eye(y_idx.numel()), torch.eye(y_idx.numel())), torch.eye(y_idx.numel()))
			loss[idx, 8] = torch.dist(gp.preconditioned_lstsq_solve(model.cov(x_idx).to_dense(), torch.eye(y_idx.numel())) @ model.cov(x_idx).to_dense(), torch.eye(y_idx.numel()))
			loss[idx, 9] = torch.dist(model.cov(x_idx).to_dense() @ gp.preconditioned_lstsq_solve(model.cov(x_idx).to_dense(), torch.eye(y_idx.numel())), torch.eye(y_idx.numel()))
		best_method_idx: torch.Tensor = torch.cat([torch.arange(pts.y.size), get_best_method_idx(covs, pts.x_t, pts.y_t, pts.extra_x_t, pts.extra_y_t, preds)])
		num_use: torch.Tensor = (best_method_idx[pts.y.size:, None] == torch.arange(pts.y.size)).to(torch.int64).sum(0)
		print(
			f"{opt.format_array("#use in extra points", num_use)}",
			f"Max use = {num_use.max().item()} at {num_use.argmax().item()} centered at f({opt.format_array(None, pts.x[int(num_use.argmax().item())], ", ")}) = {pts.y[int(num_use.argmax().item())]} with {opt.format_array("lengthscale", covs[int(num_use.argmax().item())].lengthscale)}",
			f"Nonzeros are at {opt.format_array(None, torch.argwhere(num_use != 0))} with values {opt.format_array(None, num_use[num_use != 0])}",
			sep="\n"
		)
		# best_method_idx: torch.Tensor = torch.arange(pts.y.size)
		plot.label_scatter(
			pts.x_all,
			# pts.x,
			best_method_idx.detach().numpy(),
			(torch.arange(pts.y.size).detach().numpy() + 1).astype(np.str_).tolist(),
			f"smallest_error_in_local_model_with_{func_name}"
		)
		for i, name in enumerate(names):
			plot.error_scatter(pts.x, loss[:, i], f"{name}_with_{func_name}", True)
		for i in range(covs[0].lengthscale.numel()):
			plot.error_scatter(pts.x, np.array([cov.lengthscale.ravel()[i].item() for cov in covs]), f"lengthscale_{i}_with_{func_name}", False)
		for lv in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
			print(f"# Local voters = {lv}")
			plot.plot(
				gp.gpytorch_gpr(
					covs,
					pts,
					predictor=model_mix_pred_generator(lv, preds, pts.x_all_t, best_method_idx)
					# predictor=model_mix_pred_generator(lv, preds, pts.x_t, best_method_idx)
				),
				pc,
				pts,
				f"local_mix_{lv}_with_{func_name}"
			)
