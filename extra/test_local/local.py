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


def get_neighbor_ind_used(
	x: torch.Tensor,
	x_test: torch.Tensor,
	num_neighbor: int,
	nn: sklearn.neighbors.NearestNeighbors | None,
	neighbor_ind: torch.Tensor | None,
	need_local_gp: torch.Tensor
) -> torch.Tensor:
	if neighbor_ind is not None and neighbor_ind.shape == x_test.shape[:-1] + (num_neighbor,):
		return neighbor_ind[need_local_gp]
	elif nn is not None and nn.get_params()["n_neighbors"] == num_neighbor:
		return torch.from_numpy(nn.kneighbors(x_test[need_local_gp].detach().numpy(), return_distance=False))
	else:
		nn = sklearn.neighbors.NearestNeighbors(n_neighbors=num_neighbor, n_jobs=-1).fit(x.detach().numpy())
		return torch.from_numpy(nn.kneighbors(x_test[need_local_gp].detach().numpy(), return_distance=False))


def pinv_local_pred_variance_cutoff(
	cov: gpytorch.kernels.Kernel,
	x: torch.Tensor,
	y: torch.Tensor,
	x_test: torch.Tensor,
	x_all: torch.Tensor,
	y_all: torch.Tensor,
	variance_cutoff_factor: float,
	num_neighbor: int,
	nn: sklearn.neighbors.NearestNeighbors | None = None,
	neighbor_ind: torch.Tensor | None = None,
	**kwargs
) -> torch.Tensor:
	kxm: torch.Tensor = cov(x_test, x).to_dense()
	k_inv_y: torch.Tensor = gp.preconditioned_lstsq_solve(cov(x_all, x).to_dense(), y_all)
	pred: torch.Tensor = kxm @ k_inv_y
	var: torch.Tensor = cov(x_test, diag=True).to_dense() - torch.einsum("ij,jk,ik->i", kxm, gp.square_solver(cov(x).to_dense(), torch.eye(x.shape[-2])), kxm) # diagonal only
	sigma_f2: float = (k_inv_y.reshape(1, -1) @ cov(x).to_dense() @ k_inv_y.reshape(-1, 1)).item() / y_all.numel()
	result: torch.Tensor = pred.clone()
	need_local_gp: torch.Tensor = pred ** 2 < variance_cutoff_factor * sigma_f2 * var
	num_need_local_gp: int = int(torch.count_nonzero(need_local_gp).item())
	print("{} inputs, where {} need local GP.".format(x_test.numel() // sample.distribution.DIM, num_need_local_gp))
	if num_need_local_gp > 0:
		neighbor_ind_used: torch.Tensor = get_neighbor_ind_used(x, x_test, num_neighbor, nn, neighbor_ind, need_local_gp)
		result[need_local_gp] = (cov(x_test[need_local_gp, None], x[neighbor_ind_used]) @ gp.square_solver(cov(x[neighbor_ind_used]).to_dense(), y[neighbor_ind_used, None])).to_dense().reshape(-1)
	return result


def pinv_local_pred_variance_cutoff_trial(pinv_model: gp.GPR, pts: sample.data, pc: plot.PlotConstants) -> None:
	print("\ngpytorch GPR using RBF kernel with pseudo inverse and local GP, pseudo inverse for parameters only")
	print("Check how center local GP should go")
	num_neighbor_trial: int = 32
	nn: sklearn.neighbors.NearestNeighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=num_neighbor_trial, n_jobs=-1).fit(pts.x)
	for factor in ["1e0", "1e1", "1e2", "1e3", "1e4", "1e5", "1e6", "1e7", "1e8", math.inf]:
		print("Cutoff = {}".format(factor), flush=True)
		plot.plot(
			gp.gpytorch_gpr(pinv_model, pts, predictor=pinv_local_pred_variance_cutoff, x_all=pts.x_all_t, y_all=pts.y_all_t, variance_cutoff_factor=float(factor), num_neighbor=num_neighbor_trial, nn=nn),
			pc,
			pts,
			"pinv_local_var_cut_{}".format(factor),
			False
		)


def pinv_local_pred_density_cutoff(
	cov: gpytorch.kernels.Kernel,
	x: torch.Tensor,
	y: torch.Tensor,
	x_test: torch.Tensor,
	x_all: torch.Tensor,
	y_all: torch.Tensor,
	importance_cutoff_factor: float,
	num_neighbor: int,
	nn: sklearn.neighbors.NearestNeighbors | None = None,
	neighbor_ind: torch.Tensor | None = None,
	**kwargs
) -> torch.Tensor:
	result: torch.Tensor = se.pinv_pred(cov, x, y, x_test, x_all, y_all).clone()
	need_local_gp: torch.Tensor = torch.abs(result) >= importance_cutoff_factor * torch.max(torch.abs(result))
	num_need_local_gp: int = int(torch.count_nonzero(need_local_gp).item())
	print("{} inputs, where {} need local GP.".format(x_test.numel() // sample.distribution.DIM, num_need_local_gp))
	if num_need_local_gp > 0:
		neighbor_ind_used: torch.Tensor = get_neighbor_ind_used(x, x_test, num_neighbor, nn, neighbor_ind, need_local_gp)
		result[need_local_gp] = (cov(x_test[need_local_gp, None], x[neighbor_ind_used]) @ gp.square_solver(cov(x[neighbor_ind_used]).to_dense(), y[neighbor_ind_used, None])).to_dense().reshape(-1)
	return result


def pinv_local_pred_density_cutoff_trial(pinv_model: gp.GPR, pts: sample.data, pc: plot.PlotConstants) -> None:
	print("\ngpytorch GPR using RBF kernel with pseudo inverse and local GP, pseudo inverse for parameters only")
	print("Check how far local GP should go")
	num_neighbor_trial: int = 32
	nn: sklearn.neighbors.NearestNeighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=num_neighbor_trial, n_jobs=-1).fit(pts.x)
	for factor in ["1e-{}".format(i) for i in range(17)] + [0.0]:
		print("Cutoff = {}".format(factor), flush=True)
		plot.plot(
			gp.gpytorch_gpr(pinv_model, pts, predictor=pinv_local_pred_density_cutoff, x_all=pts.x_all_t, y_all=pts.y_all_t, importance_cutoff_factor=float(factor), num_neighbor=num_neighbor_trial, nn=nn),
			pc,
			pts,
			"pinv_local_den_cut_{}".format(factor),
			False
		)


def pinv_local_pred_deviate_cutoff(
	cov: gpytorch.kernels.Kernel,
	x: torch.Tensor,
	y: torch.Tensor,
	x_test: torch.Tensor,
	x_all: torch.Tensor,
	y_all: torch.Tensor,
	dist_ratio: float,
	num_neighbor: int,
	nn: sklearn.neighbors.NearestNeighbors | None = None,
	neighbor_ind: torch.Tensor | None = None,
	**kwargs
) -> torch.Tensor:
	result: torch.Tensor = se.pinv_pred(cov, x, y, x_test, x_all, y_all).clone()
	neighbor_ind = get_neighbor_ind(x, x_test, num_neighbor, nn, neighbor_ind)
	neighbors: torch.Tensor = x[neighbor_ind]
	need_local_gp: torch.Tensor = torch.all(torch.abs(x_test - torch.mean(neighbors, -2)) < dist_ratio * torch.std(neighbors, -2), -1).reshape(x_test.shape[:-1])
	num_need_local_gp: int = int(torch.count_nonzero(need_local_gp).item())
	print("{} inputs, where {} need local GP.".format(x_test.numel() // sample.distribution.DIM, num_need_local_gp))
	if num_need_local_gp > 0:
		neighbor_used: torch.Tensor = neighbors[need_local_gp]
		result[need_local_gp] = (cov(x_test[need_local_gp, None], neighbor_used) @ gp.square_solver(cov(neighbor_used).to_dense(), y[neighbor_ind[need_local_gp], None])).to_dense().reshape(-1)
	return result


def pinv_local_pred_deviate_cutoff_trial(pinv_model: gp.GPR, pts: sample.data, pc: plot.PlotConstants) -> None:
	print("\ngpytorch GPR using RBF kernel with pseudo inverse and local GP, pseudo inverse for parameters only")
	print("Check how far local GP should go")
	num_neighbor_trial: int = 32
	nn: sklearn.neighbors.NearestNeighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=num_neighbor_trial, n_jobs=-1).fit(pts.x)
	for factor in ["0.{}".format(i) for i in range(1, 10)] + [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
		print("Cutoff = {}".format(factor), flush=True)
		plot.plot(
			gp.gpytorch_gpr(pinv_model, pts, predictor=pinv_local_pred_deviate_cutoff, x_all=pts.x_all_t, y_all=pts.y_all_t, dist_ratio=float(factor), num_neighbor=num_neighbor_trial, nn=nn),
			pc,
			pts,
			"pinv_local_dev_cut_{}".format(factor),
			False
		)


def local_predict(
	cov: gpytorch.kernels.Kernel,
	x: torch.Tensor,
	y: torch.Tensor,
	x_test: torch.Tensor,
	x_all: torch.Tensor,
	y_all: torch.Tensor,
	num_neighbor: int,
	nn: sklearn.neighbors.NearestNeighbors | None = None,
	neighbor_ind: torch.Tensor | None = None,
	**kwargs
) -> torch.Tensor:
	neighbor_ind = get_neighbor_ind(x, x_test, num_neighbor, nn, neighbor_ind)
	neighbors: torch.Tensor = x.reshape(-1, sample.distribution.DIM)[neighbor_ind]
	return (cov(x_test[..., None, :], neighbors, **kwargs) @ gp.square_solver(cov(neighbors, **kwargs).to_dense(), y[neighbor_ind, None])).to_dense().reshape(x_test.shape[:-1])


def get_central_neighbor_indices(nn: sklearn.neighbors.NearestNeighbors, central_pts: npt.NDArray[np.double]) -> torch.Tensor:
	n_neighbor: int = nn.get_params()["n_neighbors"]
	central_neighbor_ind = nn.kneighbors(central_pts, n_neighbor + 1, False) # N_MEMBER * N_NEIGHBOR
	assert isinstance(central_neighbor_ind, np.ndarray)
	return torch.from_numpy(central_neighbor_ind[central_neighbor_ind != np.arange(central_pts.shape[0]).reshape(-1, 1)].reshape(central_pts.shape[0], n_neighbor)).detach()


def local_parameter_trial(
	global_model: gp.GPR,
	pts: sample.data,
	global_predictor: gp.ApproxPredType | None,
	num_neighbor: int = 32,
	**kwargs
) -> None:
	print("\nOptimize one point error who is the largest from global prediction by local GPR")
	pred: torch.Tensor = global_predictor(global_model.cov, pts.x_t, pts.y_t, pts.x_all_t, pts.x_all_t, pts.y_all_t, **kwargs) if global_predictor is not None else gp.default_predict(global_model.cov, pts.x_t, pts.y_t, pts.x_all_t)
	diff: torch.Tensor = pred - pts.y_all_t
	max_diff_idx: int = int(torch.argmax(torch.abs(diff)).item())
	print("Max diff = {} at {}, ({}), in {}".format(
		torch.max(torch.abs(diff)).item(),
		max_diff_idx,
		opt.format_array(None, pts.x_all[max_diff_idx]),
		"center" if max_diff_idx < pts.x.shape[0] else "extra"
	))
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
	model: gp.GPR = opt.gpytorch_train(
		select_x,
		select_y,
		gpytorch.kernels.RBFKernel(sample.distribution.DIM),
		lambda cov, x, y, **kwargs: (local_predict(cov, pts.x_t, pts.y_t, x, pts.x_all_t, pts.y_all_t, num_neighbor, nn, neighbor_ind, **kwargs) - y).square().sum(),
		initial_values=[p.data for p in global_model.parameters()]
	) or global_model
	opt.print_stuff(
		model,
		torch.abs(local_predict(model.cov, pts.x_t, pts.y_t, select_x, pts.x_all_t, pts.y_all_t, num_neighbor, nn, neighbor_ind) - select_y).item()
	)


def local_pred_trial(pinv_model: gp.GPR, pts: sample.data, pc: plot.PlotConstants) -> tuple[int, sklearn.neighbors.NearestNeighbors, torch.Tensor]:
	print("\ngpytorch GPR using RBF kernel with local GP")
	print("Check how large scale the local GP should have")
	smallest_error: float = math.inf
	best_nn: int = -1
	its_nn: sklearn.neighbors.NearestNeighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=32)
	its_central_nn: torch.Tensor = torch.Tensor()
	for n_neighbor in [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]:
		nn: sklearn.neighbors.NearestNeighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbor, n_jobs=-1).fit(pts.x)
		central_neighbor_ind: torch.Tensor = get_central_neighbor_indices(nn, pts.x)
		model: gp.GPR = opt.gpytorch_train(
			pts.x_t,
			pts.y_t,
			gpytorch.kernels.RBFKernel(sample.distribution.DIM),
			se.approx_se,
			initial_values=[torch.ones(sample.distribution.DIM)],
			pred=local_predict,
			x_all=pts.x_all_t,
			y_all=pts.y_all_t,
			num_neighbor=n_neighbor,
			nn=nn,
			neighbor_ind=central_neighbor_ind
		) or pinv_model
		err: float
		try:
			err = se.approx_se(model.cov, pts.x_t, pts.y_t, local_predict, pts.x_all_t, pts.y_all_t, num_neighbor=n_neighbor, nn=nn, neighbor_ind=central_neighbor_ind).item()
		except Exception as e:
			traceback.print_exception(e)
			err = math.inf
		print("#Neighbor = {}, err = {}".format(n_neighbor, err), flush=True)
		if err < smallest_error:
			best_nn = n_neighbor
			its_nn = nn
			its_central_nn = central_neighbor_ind
		plot.plot(
			gp.gpytorch_gpr(model, pts, predictor=local_predict, x_all=pts.x_all_t, y_all=pts.y_all_t, num_neighbor=n_neighbor, nn=nn),
			pc,
			pts,
			"local_{}".format(n_neighbor)
		)
	return best_nn, its_nn, its_central_nn


def local_pinv_pred(
	cov: gpytorch.kernels.Kernel,
	x: torch.Tensor,
	y: torch.Tensor,
	x_test: torch.Tensor,
	x_all: torch.Tensor,
	y_all: torch.Tensor,
	num_neighbor: int,
	nn: sklearn.neighbors.NearestNeighbors | None = None,
	neighbor_ind: torch.Tensor | None = None,
	extra_neighbor: int | None = None,
	extra_nn: sklearn.neighbors.NearestNeighbors | None = None,
	extra_neighbor_ind: torch.Tensor | None = None,
	**kwargs
) -> torch.Tensor:
	neighbor_ind = get_neighbor_ind(x, x_test, num_neighbor, nn, neighbor_ind)
	if extra_neighbor is None or extra_neighbor <= num_neighbor:
		return local_predict(cov, x, y, x_test, x_all, y_all, num_neighbor, nn, neighbor_ind, **kwargs)
	else:
		neighbors: torch.Tensor = x.reshape(-1, sample.distribution.DIM)[neighbor_ind]
		if extra_neighbor >= y.numel(): # all points, special case, remove nn
			return (cov(x_test[..., None, :], neighbors) @ gp.preconditioned_lstsq_solve(cov(x.reshape((x_test.ndim - 1) * (1,) + x.shape), neighbors).to_dense(), y.reshape((x_test.ndim - 1) * (1,) + y.shape + (1,)))).to_dense().reshape(x_test.shape[:-1])
		else: # extra nn needed
			extra_neighbor_ind = get_neighbor_ind(x, x_test, extra_neighbor, extra_nn, extra_neighbor_ind)
			return (cov(x_test[..., None, :], neighbors) @ gp.preconditioned_lstsq_solve(cov(x.reshape(-1, sample.distribution.DIM)[extra_neighbor_ind], neighbors).to_dense(), y[extra_neighbor_ind, None])).to_dense().reshape(x_test.shape[:-1])


def local_pinv_pred_trial(local_model: gp.GPR, pts: sample.data, pc: plot.PlotConstants, num_neighbor: int, nn: sklearn.neighbors.NearestNeighbors) -> None:
	print("\ngpytorch GPR using RBF kernel with local GP")
	print("Check if local GP should use pseudo inverse")
	for extra_neighbor in [num_neighbor, 2 * num_neighbor, 4 * num_neighbor, pts.extra_y.size]:
		print("# Extra Neighbor = {}".format(extra_neighbor))
		plot.plot(
			gp.gpytorch_gpr(local_model, pts, predictor=local_pinv_pred, x_all=pts.extra_x_t, y_all=pts.extra_y_t, nn=nn, extra_neighbor=extra_neighbor),
			pc,
			pts,
			"local_pinv_{}".format(extra_neighbor))


def local_global_se(
	cov: gpytorch.kernels.Kernel,
	x: torch.Tensor,
	y: torch.Tensor,
	x_all: torch.Tensor,
	y_all: torch.Tensor,
	global_predictor: gp.ApproxPredType | None,
	num_neighbor: int,
	neighbor_ind: torch.Tensor | None = None,
	**kwargs
) -> torch.Tensor:
	result_global: torch.Tensor = global_predictor(cov, x, y, x_all, x_all, y_all, **kwargs) if global_predictor is not None else gp.default_predict(cov, x, y, x_all)
	result_local: torch.Tensor = local_predict(cov, x, y, x_all, x_all, y_all, num_neighbor, neighbor_ind=neighbor_ind, **kwargs)
	return torch.minimum(torch.abs(y_all - result_global), torch.abs(y_all - result_local)).square().sum()


def local_global_mix_pred(
	cov: gpytorch.kernels.Kernel,
	x: torch.Tensor,
	y: torch.Tensor,
	x_test: torch.Tensor,
	x_all: torch.Tensor,
	y_all: torch.Tensor,
	num_neighbor: int,
	global_predictor: gp.ApproxPredType | None = None,
	nn: sklearn.neighbors.NearestNeighbors | None = None,
	neighbor_ind: torch.Tensor | None = None,
	use_local: torch.Tensor | None = None,
	**kwargs
) -> torch.Tensor:
	if use_local is None:
		return local_predict(cov, x, y, x_test, x_all, y_all, num_neighbor, nn, neighbor_ind, **kwargs)
	else:
		# same as v3
		neighbor_ind = get_neighbor_ind(x, x_test, num_neighbor, nn, neighbor_ind)
		neighbors: torch.Tensor = x.reshape(-1, sample.distribution.DIM)[neighbor_ind]
		result_local: torch.Tensor = (cov(x_test[..., None, :], neighbors) @ gp.square_solver(cov(neighbors).to_dense(), y_all[neighbor_ind, None])).to_dense().reshape(x_test.shape[:-1])
		result_global: torch.Tensor = global_predictor(cov, x, y, x_test, x_all, y_all, **kwargs) if global_predictor is not None else gp.default_predict(cov, x, y, x_test)
		# predict based on votes of the locals
		use_local_weight: torch.Tensor = use_local[neighbor_ind].to(torch.double).sum(dim=-1)
		return use_local_weight / num_neighbor * result_local + (1.0 - use_local_weight / num_neighbor) * result_global


def local_global_mix_trial(local_model: gp.GPR, pts: sample.data, pc: plot.PlotConstants, num_neighbor: int, nn: sklearn.neighbors.NearestNeighbors, central_neighbor_ind: torch.Tensor) -> None:
	print("\ngpytorch GPR using RBF kernel with mixed pseudo inverse and local GP")
	for pred, name in zip([None, se.pinv_pred], ["loocv", "pinv"]):
		model: gp.GPR | None = opt.gpytorch_train(
			pts.x_t,
			pts.y_t,
			gpytorch.kernels.RBFKernel(sample.distribution.DIM),
			local_global_se,
			initial_values=[p.data for p in local_model.parameters()],
			initial_value_search=False,
			x_all=pts.x_all_t,
			y_all=pts.y_all_t,
			global_predictor=pred,
			neighbor_ind=central_neighbor_ind
		)
		if model is not None:
			result_global: torch.Tensor = pred(model.cov, pts.x_t, pts.y_t, pts.x_all_t, pts.x_all_t, pts.y_all_t) if pred is not None else gp.default_predict(model.cov, pts.x_t, pts.y_t, pts.x_all_t)
			result_local: torch.Tensor = local_global_mix_pred(model.cov, pts.x_t, pts.y_t, pts.x_all_t, pts.x_all_t, pts.y_all_t, num_neighbor, neighbor_ind=central_neighbor_ind)
			use_local: torch.Tensor = torch.abs(pts.extra_y_t - result_local) < torch.abs(pts.extra_y_t - result_global)
			plot.plot(
				gp.gpytorch_gpr(model, pts, predictor=local_global_mix_pred, x_all=pts.x_all_t, y_all=pts.y_all_t, nn=nn, use_local=use_local),
				pc,
				pts,
				"local_weighted_{}".format(name)
			)

def local_global_model_mix_pred(
	cov: gpytorch.kernels.Kernel,
	x: torch.Tensor,
	y: torch.Tensor,
	x_test: torch.Tensor,
	x_all: torch.Tensor,
	y_all: torch.Tensor,
	global_predictor: gp.ApproxPredType | None,
	cov_local: gpytorch.kernels.Kernel,
	num_neighbor: int,
	nn: sklearn.neighbors.NearestNeighbors | None = None,
	neighbor_ind: torch.Tensor | None = None,
	use_local: torch.Tensor | None = None,
	local_voters: int | None = None,
	**kwargs
) -> torch.Tensor:
	# nn is from x but use_local is from x_all
	result_global: torch.Tensor = global_predictor(cov, x, y, x_test, x_all, y_all, **kwargs) if global_predictor is not None else gp.default_predict(cov, x, y, x_test)
	if use_local is None:
		return result_global
	else:
		# same as v3
		result_local: torch.Tensor = local_predict(cov_local, x, y, x_test, x_all, y_all, num_neighbor, nn, neighbor_ind)
		if local_voters is None:
			local_voters = num_neighbor
		use_local_neighbor_ind: torch.Tensor = get_neighbor_ind(x_all, x_test, local_voters)
		# predict based on votes of the locals
		use_local_weight: torch.Tensor = use_local[use_local_neighbor_ind].to(torch.double).sum(dim=-1) / local_voters
		return use_local_weight * result_local + (1.0 - use_local_weight) * result_global


def local_global_model_mix_trial(
	local_model: gp.GPR,
	global_model: gp.GPR,
	pts: sample.data,
	pc: plot.PlotConstants,
	global_predictor: gp.ApproxPredType | None,
	num_neighbor: int,
	nn: sklearn.neighbors.NearestNeighbors,
	central_neighbor_ind: torch.Tensor,
	**kwargs
) -> None:
	print("\nWeighted average prediction of different methods")
	print("Weight of each method is the ratio of points with smallest prediction error of that method")
	pred_global: torch.Tensor = global_predictor(global_model.cov, pts.x_t, pts.y_t, pts.x_all_t, pts.x_all_t, pts.y_all_t, **kwargs) if global_predictor is not None else gp.default_predict(global_model.cov, pts.x_t, pts.y_t, pts.x_all_t)
	pred_local: torch.Tensor = local_predict(local_model.cov, pts.x_t, pts.y_t, pts.x_all_t, pts.x_all_t, pts.y_all_t, num_neighbor, neighbor_ind=central_neighbor_ind)
	use_local: torch.Tensor = torch.abs(pts.y_all_t - pred_local) < torch.abs(pts.y_all_t - pred_global)
	for lv in [1, 2, 3, 4, 7, 8, 15, 16, 31, 32]:
		plot.plot(
			gp.gpytorch_gpr(global_model, pts, predictor=local_global_model_mix_pred, x_all=pts.x_all_t, y_all=pts.y_all_t,  global_predictor=global_predictor, cov_local=local_model.cov, num_neighbor=num_neighbor, nn=nn, use_local=use_local, local_voters=lv),
			pc,
			pts,
			"local_global_model_mix_{}".format(lv)
		)


class SeparateRBFKernel(gpytorch.kernels.Kernel):
	__RAW_PARAM_NAME = "raw_length"

	def __init__(
		self,
		ard_num_dims: int,
		x: torch.Tensor,
		initial_value: torch.Tensor | None = None,
		**kwargs
	) -> None:
		super().__init__(ard_num_dims, **kwargs)
		assert x.ndim == 2 and x.shape[-1] == ard_num_dims
		self.__x: torch.Tensor = x
		self.__num_inputs: int = self.__x.shape[-2]
		if initial_value is not None:
			initial_value = initial_value.broadcast_to(self.__x.shape).detach()
		else:
			initial_value = torch.ones_like(self.__x)
		self.register_parameter(
			__class__.__RAW_PARAM_NAME,
			torch.nn.Parameter(initial_value)
		)
		self.register_constraint(__class__.__RAW_PARAM_NAME, gp.NoConstraint(initial_value))

	@property
	def length(self) -> torch.Tensor:
		assert hasattr(self, __class__.__RAW_PARAM_NAME)
		length = getattr(self, __class__.__RAW_PARAM_NAME)
		assert isinstance(length, torch.Tensor)
		return length

	@length.setter
	def length(self, value: torch.Tensor) -> None:
		assert hasattr(self, __class__.__RAW_PARAM_NAME)
		length = getattr(self, __class__.__RAW_PARAM_NAME)
		assert isinstance(length, torch.Tensor)
		assert self.ard_num_dims is not None
		length[...] = value.broadcast_to(self.__num_inputs, self.ard_num_dims)

	def forward(self, x1: torch.Tensor, x2: torch.Tensor | None = None, diag: bool = False, **kwargs) -> torch.Tensor:
		if x2 is None:
			x2 = x1
		if diag:
			assert torch.equal(x1, x2)
			return torch.ones(x1.shape[:-1])
		else:
			mask: torch.Tensor = torch.all(x2[..., None, :] == self.__x, -1).to(torch.int8) # shape of ... * N * L
			assert torch.all(mask.sum(-1) == 1).item()
			# self.length[torch.argmax(mask, -1)], shape of ... * N * D
			return torch.exp(-torch.sum(((x1[..., None, :] - x2[..., None, :, :]) / self.length[torch.argmax(mask, -1)][..., None, :, :]) ** 2, -1) / 2.0)


def local_separate_kernel_trial(pts: sample.data, pc: plot.PlotConstants, num_neighbor: int) -> None:
	print("\ngpytorch GPR using kernel for local GP")
	nn: sklearn.neighbors.NearestNeighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=num_neighbor, n_jobs=-1).fit(pts.x)
	central_neighbor_ind: torch.Tensor = get_central_neighbor_indices(nn, pts.x)
	model: gp.GPR | None = opt.gpytorch_train(
		pts.x_t,
		pts.y_t,
		SeparateRBFKernel(sample.distribution.DIM, pts.x_t),
		se.approx_se,
		initial_values=[torch.ones_like(pts.x_t)],
		pred=local_predict,
		x_all=pts.x_all_t,
		y_all=pts.y_all_t,
		num_neighbor=num_neighbor,
		nn=nn,
		neighbor_ind=central_neighbor_ind,
		ind=central_neighbor_ind
	)
	if model is None:
		return
	plot.plot(
		gp.gpytorch_gpr(model, pts, predictor=local_predict, x_all=pts.x_all_t, y_all=pts.y_all_t, num_neighbor=num_neighbor, nn=nn, ind=central_neighbor_ind),
		pc,
		pts,
		"local_separate_kernel",
		False
	)
