import math
import os
import sys
import traceback

import gpytorch
import numpy as np
import numpy.typing as npt
import sklearn.neighbors
import torch

sys.path.append(os.path.dirname(__file__))

import gp
import opt
import plot
import sample
import se


def normalize_generator(pred: gp.PredType[gp.KernelsType]) -> gp.PredType[gp.KernelsType]:
	def normalize_pred(covs: gp.KernelsType, x: torch.Tensor, y: torch.Tensor, x_test: torch.Tensor) -> torch.Tensor:
		x_mean: torch.Tensor
		x_stddev: torch.Tensor
		x_mean, x_stddev = torch.std_mean(x, 0)
		y_mean: torch.Tensor
		y_stddev: torch.Tensor
		y_mean, y_stddev = torch.std_mean(y)
		return pred(covs, (x - x_mean) / x_stddev, (y - y_mean) / y_stddev, (x_test - x_mean) / x_stddev) * y_stddev + y_mean
	return normalize_pred


def get_neighbor_ind(
	x: torch.Tensor,
	x_test: torch.Tensor,
	num_neighbor: int,
	nn: sklearn.neighbors.NearestNeighbors | None = None,
	neighbor_ind: torch.Tensor | None = None,
) -> torch.Tensor:
	if neighbor_ind is not None and neighbor_ind.shape == x_test.shape[:-1] + (num_neighbor,):
		return neighbor_ind
	elif nn is not None and nn.get_params()["n_neighbors"] == num_neighbor:
		return torch.from_numpy(nn.kneighbors(x_test.detach().numpy().reshape(-1, sample.distribution.DIM), return_distance=False)).reshape(x_test.shape[:-1] + (num_neighbor,))
	else:
		nn = sklearn.neighbors.NearestNeighbors(n_neighbors=num_neighbor, n_jobs=-1).fit(x.detach().numpy())
		return torch.from_numpy(nn.kneighbors(x_test.detach().numpy().reshape(-1, sample.distribution.DIM), return_distance=False)).reshape(x_test.shape[:-1] + (num_neighbor,))


def local_predict_generator(
	num_neighbor: int,
	nn: sklearn.neighbors.NearestNeighbors | None = None,
	neighbor_ind: torch.Tensor | None = None
) -> gp.PredType[gpytorch.kernels.Kernel]:
	def local_predict(
		covs: gpytorch.kernels.Kernel,
		x: torch.Tensor,
		y: torch.Tensor,
		x_test: torch.Tensor,
	) -> torch.Tensor:
		nonlocal neighbor_ind
		neighbor_ind = get_neighbor_ind(x, x_test, num_neighbor, nn, neighbor_ind)
		neighbors: torch.Tensor = x.reshape(-1, sample.distribution.DIM)[neighbor_ind]
		return (covs(x_test[..., None, :], neighbors) @ gp.square_solver(covs(neighbors).to_dense(), y[neighbor_ind, None])).to_dense().reshape(x_test.shape[:-1])
	return local_predict


def local_pred_as_cutoff_generator(
	num_neighbor: int,
	nn: sklearn.neighbors.NearestNeighbors | None = None,
	neighbor_ind: torch.Tensor | None = None,
	cutoff_judge: gp.PredType[gpytorch.kernels.Kernel] | None = None,
	other_pred: gp.PredType[gpytorch.kernels.Kernel] | None = None
) -> gp.PredType[gpytorch.kernels.Kernel]:
	local_predict: gp.PredType[gpytorch.kernels.Kernel] = local_predict_generator(num_neighbor, nn, neighbor_ind)
	def local_pred_as_cutoff(
		covs: gpytorch.kernels.Kernel,
		x: torch.Tensor,
		y: torch.Tensor,
		x_test: torch.Tensor
	) -> torch.Tensor:
		result_local: torch.Tensor = local_predict(covs, x, y, x_test)
		if cutoff_judge is not None:
			need_local_gp: torch.Tensor = cutoff_judge(covs, x, y, x_test)
			print(f"{x_test.numel() // sample.distribution.DIM} inputs, where {torch.count_nonzero(need_local_gp).item()} need local GP.")
			return torch.where(
				need_local_gp,
				result_local,
				(other_pred or gp.default_predict)(covs, x, y, x_test)
			)
		else:
			return result_local
	return local_pred_as_cutoff


def variance_cutoff_generator(
	x_all: torch.Tensor,
	y_all: torch.Tensor,
	variance_cutoff_factor: float
) -> gp.PredType[gpytorch.kernels.Kernel]:
	pinv_pred: gp.PredType[gpytorch.kernels.Kernel] = se.pinv_pred_generator(x_all, y_all)
	def variance_cutoff(
		covs: gpytorch.kernels.Kernel,
		x: torch.Tensor,
		y: torch.Tensor,
		x_test: torch.Tensor
	) -> torch.Tensor:
		kxm: torch.Tensor = covs(x_test, x).to_dense()
		k_inv_y: torch.Tensor = gp.preconditioned_lstsq_solve(covs(x_all, x).to_dense(), y_all)
		var: torch.Tensor = covs(x_test, diag=True).to_dense() - torch.einsum("ij,jk,ik->i", kxm, gp.square_solver(covs(x).to_dense(), torch.eye(x.shape[-2])), kxm) # diagonal only
		sigma_f2: float = (k_inv_y.reshape(1, -1) @ covs(x).to_dense() @ k_inv_y.reshape(-1, 1)).item() / y_all.numel()
		return pinv_pred(covs, x, y, x_test) ** 2 < variance_cutoff_factor * sigma_f2 * var
	return variance_cutoff


def pinv_local_pred_variance_cutoff_trial(
	pinv_model: gp.GPR,
	pts: sample.data,
	pc: plot.PlotConstants,
	num_neighbor: int = 32
) -> None:
	print("\ngpytorch GPR using RBF kernel with pseudo inverse and local GP, pseudo inverse for parameters only")
	print("Check how center local GP should go")
	nn: sklearn.neighbors.NearestNeighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=num_neighbor, n_jobs=-1).fit(pts.x)
	for factor in ["1e0", "1e1", "1e2", "1e3", "1e4", "1e5", "1e6", "1e7", "1e8", math.inf]:
		print(f"Cutoff = {factor}", flush=True)
		plot.plot(
			gp.gpytorch_gpr(
				pinv_model.cov,
				pts,
				predictor=local_pred_as_cutoff_generator(
					num_neighbor,
					nn,
					cutoff_judge=variance_cutoff_generator(pts.x_all_t, pts.y_all_t, float(factor)),
					other_pred=se.pinv_pred_generator(pts.x_all_t, pts.y_all_t)
				)
			),
			pc,
			pts,
			f"pinv_local_var_cut_{factor}",
			False
		)


def density_cutoff_generator(
	x_all: torch.Tensor,
	y_all: torch.Tensor,
	importance_cutoff_factor: float
) -> gp.PredType[gpytorch.kernels.Kernel]:
	pinv_pred: gp.PredType[gpytorch.kernels.Kernel] = se.pinv_pred_generator(x_all, y_all)
	def density_cutoff(
		covs: gpytorch.kernels.Kernel,
		x: torch.Tensor,
		y: torch.Tensor,
		x_test: torch.Tensor
	) -> torch.Tensor:
		result: torch.Tensor = pinv_pred(covs, x, y, x_test)
		return torch.abs(result) >= importance_cutoff_factor * torch.max(torch.abs(result))
	return density_cutoff


def pinv_local_pred_density_cutoff_trial(
	pinv_model: gp.GPR,
	pts: sample.data,
	pc: plot.PlotConstants,
	num_neighbor: int = 32
) -> None:
	print("\ngpytorch GPR using RBF kernel with pseudo inverse and local GP, pseudo inverse for parameters only")
	print("Check how far local GP should go")
	nn: sklearn.neighbors.NearestNeighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=num_neighbor, n_jobs=-1).fit(pts.x)
	for factor in [f"1e-{i}" for i in range(17)] + [0.0]:
		print(f"Cutoff = {factor}", flush=True)
		plot.plot(
			gp.gpytorch_gpr(
				pinv_model.cov,
				pts,
				predictor=local_pred_as_cutoff_generator(
					num_neighbor,
					nn,
					cutoff_judge=density_cutoff_generator(pts.x_all_t, pts.y_all_t, float(factor)),
					other_pred=se.pinv_pred_generator(pts.x_all_t, pts.y_all_t)
				)
			),
			pc,
			pts,
			f"pinv_local_den_cut_{factor}",
			False
		)


def deviate_cutoff_generator(
	dist_ratio: float,
	num_neighbor: int,
	nn: sklearn.neighbors.NearestNeighbors | None = None,
	neighbor_ind: torch.Tensor | None = None,
) -> gp.PredType[gpytorch.kernels.Kernel]:
	def deviate_cutoff(
		covs: gpytorch.kernels.Kernel,
		x: torch.Tensor,
		y: torch.Tensor,
		x_test: torch.Tensor
	) -> torch.Tensor:
		neighbors: torch.Tensor = x[get_neighbor_ind(x, x_test, num_neighbor, nn, neighbor_ind)]
		return torch.all(torch.abs(x_test - torch.mean(neighbors, -2)) < dist_ratio * torch.std(neighbors, -2), -1).reshape(x_test.shape[:-1])
	return deviate_cutoff


def pinv_local_pred_deviate_cutoff_trial(
	pinv_model: gp.GPR,
	pts: sample.data,
	pc: plot.PlotConstants,
	num_neighbor: int = 32
) -> None:
	print("\ngpytorch GPR using RBF kernel with pseudo inverse and local GP, pseudo inverse for parameters only")
	print("Check how far local GP should go")
	nn: sklearn.neighbors.NearestNeighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=num_neighbor, n_jobs=-1).fit(pts.x)
	for factor in [f"0.{i}" for i in range(1, 10)] + [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
		print(f"Cutoff = {factor}", flush=True)
		plot.plot(
			gp.gpytorch_gpr(
				pinv_model.cov,
				pts,
				predictor=local_pred_as_cutoff_generator(
					num_neighbor,
					nn,
					cutoff_judge=deviate_cutoff_generator(float(factor), num_neighbor, nn),
					other_pred=se.pinv_pred_generator(pts.x_all_t, pts.y_all_t)
				)
			),
			pc,
			pts,
			f"pinv_local_dev_cut_{factor}",
			False
		)


def get_central_neighbor_indices(nn: sklearn.neighbors.NearestNeighbors, central_pts: npt.NDArray[np.double]) -> torch.Tensor:
	n_neighbor: int = nn.get_params()["n_neighbors"]
	central_neighbor_ind = nn.kneighbors(central_pts, n_neighbor + 1, False) # N_MEMBER * N_NEIGHBOR
	assert isinstance(central_neighbor_ind, np.ndarray)
	return torch.from_numpy(central_neighbor_ind[central_neighbor_ind != np.arange(central_pts.shape[0]).reshape(-1, 1)].reshape(central_pts.shape[0], n_neighbor)).detach()


def get_all_neighbor_indices(nn: sklearn.neighbors.NearestNeighbors, central_pts: npt.NDArray[np.double], extra_pts: npt.NDArray[np.double]) -> torch.Tensor:
	return torch.cat([get_central_neighbor_indices(nn, central_pts), torch.from_numpy(nn.kneighbors(extra_pts, return_distance=False))])


def single_local_parameter_trial(
	global_model: gp.GPR,
	pts: sample.data,
	global_predictor: gp.PredType[gpytorch.kernels.Kernel] | None,
	num_neighbor: int = 32
) -> None:
	print("\nOptimize one point error who is the largest from global prediction by local GPR")
	pred: torch.Tensor = (global_predictor or gp.default_predict)(global_model.cov, pts.x_t, pts.y_t, pts.x_all_t)
	diff: torch.Tensor = pred - pts.y_all_t
	max_diff_idx: int = int(torch.argmax(torch.abs(diff)).item())
	print(f"Max diff = {torch.max(torch.abs(diff)).item()} at {max_diff_idx}, ({sample.format_array(None, pts.x_all[max_diff_idx])}), in {"center" if max_diff_idx < pts.x.shape[0] else "extra"}")
	nn: sklearn.neighbors.NearestNeighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=num_neighbor, n_jobs=-1).fit(pts.x)
	neighbor_ind: torch.Tensor
	select_x: torch.Tensor = pts.x_all_t[max_diff_idx].reshape(1, sample.distribution.DIM)
	select_y: torch.Tensor = pts.y_all_t[max_diff_idx].reshape(1, 1)
	if max_diff_idx < pts.x.shape[0]:
		central_neighbor_ind = nn.kneighbors(select_x.detach().numpy(), num_neighbor + 1, False) # N_MEMBER * N_NEIGHBOR
		assert isinstance(central_neighbor_ind, np.ndarray)
		neighbor_ind = torch.from_numpy(central_neighbor_ind[central_neighbor_ind != max_diff_idx].reshape(1, num_neighbor)).detach()
	else:
		neighbor_ind = torch.from_numpy(nn.kneighbors(pts.x_all[max_diff_idx].reshape(1, sample.distribution.DIM), return_distance=False))
	single_pt_local_pred: gp.PredType[gpytorch.kernels.Kernel] = local_predict_generator(num_neighbor, nn, neighbor_ind)
	model: gp.GPR = opt.gpytorch_train(
		select_x,
		select_y,
		gpytorch.kernels.RBFKernel(sample.distribution.DIM),
		lambda cov, x, y: (single_pt_local_pred(cov, pts.x_t, pts.y_t, x) - y).square().sum(),
		initial_values=[p.data for p in global_model.parameters()]
	) or global_model
	opt.print_stuff(
		model,
		torch.abs(single_pt_local_pred(model.cov, pts.x_t, pts.y_t, select_x) - select_y).item()
	)


def local_pred_trial(pinv_model: gp.GPR, pts: sample.data, pc: plot.PlotConstants) -> tuple[int, sklearn.neighbors.NearestNeighbors, torch.Tensor]:
	print("\ngpytorch GPR using RBF kernel with local GP")
	print("Check how large scale the local GP should have")
	smallest_error: float = math.inf
	best_nn: int = -1
	its_nn: sklearn.neighbors.NearestNeighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=32)
	its_pts_nn: torch.Tensor = torch.Tensor()
	for n_neighbor in [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]:
		nn: sklearn.neighbors.NearestNeighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbor, n_jobs=-1).fit(pts.x)
		pts_neighbor_ind: torch.Tensor = get_all_neighbor_indices(nn, pts.x, pts.extra_x)
		local_predict: gp.PredType[gpytorch.kernels.Kernel] = local_predict_generator(n_neighbor, nn, pts_neighbor_ind)
		loss_func: opt.LossFuncType = se.se_generator(local_predict, pts.x_all_t, pts.y_all_t)
		model: gp.GPR = opt.gpytorch_train(
			pts.x_t,
			pts.y_t,
			gpytorch.kernels.RBFKernel(sample.distribution.DIM),
			loss_func,
			initial_values=[torch.ones(sample.distribution.DIM)]
		) or pinv_model
		err: float
		try:
			err = loss_func(model.cov, pts.x_t, pts.y_t).item()
		except Exception as e:
			traceback.print_exception(e)
			err = math.inf
		print(f"#Neighbor = {n_neighbor}, err = {err}", flush=True)
		if err < smallest_error:
			best_nn = n_neighbor
			its_nn = nn
			its_pts_nn = pts_neighbor_ind
		plot.plot(
			gp.gpytorch_gpr(model.cov, pts, predictor=local_predict),
			pc,
			pts,
			f"local_{n_neighbor}"
		)
	return best_nn, its_nn, its_pts_nn


def local_pinv_pred_generator(
	num_neighbor: int,
	nn: sklearn.neighbors.NearestNeighbors | None = None,
	neighbor_ind: torch.Tensor | None = None,
	extra_neighbor: int | None = None,
	extra_nn: sklearn.neighbors.NearestNeighbors | None = None,
	extra_neighbor_ind: torch.Tensor | None = None
) -> gp.PredType[gpytorch.kernels.Kernel]:
	local_predict: gp.PredType[gpytorch.kernels.Kernel] = local_predict_generator(num_neighbor, nn, neighbor_ind)
	def local_pinv_pred(
		covs: gpytorch.kernels.Kernel,
		x: torch.Tensor,
		y: torch.Tensor,
		x_test: torch.Tensor
	) -> torch.Tensor:
		nonlocal neighbor_ind, extra_neighbor_ind
		neighbor_ind = get_neighbor_ind(x, x_test, num_neighbor, nn, neighbor_ind)
		if extra_neighbor is None or extra_neighbor <= num_neighbor:
			return local_predict(covs, x, y, x_test)
		else:
			neighbors: torch.Tensor = x.reshape(-1, sample.distribution.DIM)[neighbor_ind]
			if extra_neighbor >= y.numel(): # all points, special case, remove nn
				return (covs(x_test[..., None, :], neighbors) @ gp.preconditioned_lstsq_solve(covs(x.reshape((x_test.ndim - 1) * (1,) + x.shape), neighbors).to_dense(), y.reshape((x_test.ndim - 1) * (1,) + y.shape + (1,)))).to_dense().reshape(x_test.shape[:-1])
			else: # extra nn needed
				extra_neighbor_ind = get_neighbor_ind(x, x_test, extra_neighbor, extra_nn, extra_neighbor_ind)
				return (covs(x_test[..., None, :], neighbors) @ gp.preconditioned_lstsq_solve(covs(x.reshape(-1, sample.distribution.DIM)[extra_neighbor_ind], neighbors).to_dense(), y[extra_neighbor_ind, None])).to_dense().reshape(x_test.shape[:-1])
	return local_pinv_pred


def local_pinv_pred_trial(
	local_model: gp.GPR,
	pts: sample.data,
	pc: plot.PlotConstants,
	num_neighbor: int,
	nn: sklearn.neighbors.NearestNeighbors,
	pts_neighbor_ind: torch.Tensor
) -> None:
	print("\ngpytorch GPR using RBF kernel with local GP")
	print("Check if local GP should use pseudo inverse")
	for extra_neighbor in [num_neighbor, 2 * num_neighbor, 4 * num_neighbor, pts.y.size]:
		print(f"# Extra Neighbor = {extra_neighbor}")
		extra_nn: None | sklearn.neighbors.NearestNeighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=extra_neighbor, n_jobs=-1).fit(pts.x) if num_neighbor < extra_neighbor < pts.y.size else None
		extra_ind: None | torch.Tensor = get_all_neighbor_indices(extra_nn, pts.x, pts.extra_x) if extra_nn is not None else None
		pred: gp.PredType[gpytorch.kernels.Kernel] = local_pinv_pred_generator(num_neighbor, nn, pts_neighbor_ind, extra_neighbor, extra_nn, extra_ind)
		model: gp.GPR = opt.gpytorch_train(
			pts.x_t,
			pts.y_t,
			gpytorch.kernels.RBFKernel(sample.distribution.DIM),
			se.se_generator(pred, pts.x_all_t, pts.y_all_t),
			initial_values=[p.data for p in local_model.parameters()]
		) or local_model
		plot.plot(
			gp.gpytorch_gpr(
				model.cov,
				pts,
				predictor=pred,
			),
			pc,
			pts,
			f"local_pinv_{extra_neighbor}"
		)
