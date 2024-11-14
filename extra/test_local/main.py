#!/usr/bin/env python

# imports
import datetime
import gc
import os
import sys

import gpytorch
import sklearn.neighbors
import torch

torch.set_default_dtype(torch.float64)

sys.path.append(os.path.dirname(__file__))

import gp
import linear
import local
import mix
import plot
import sample
import se


def main(seed: int | None = None, search_param: bool = True) -> None:
	gc.set_debug(gc.DEBUG_UNCOLLECTABLE | gc.DEBUG_SAVEALL | gc.DEBUG_STATS)
	if gc.isenabled():
		gc.disable()

	dist: sample.distribution = sample.distribution(seed)
	pc: plot.PlotConstants = plot.PlotConstants(dist.y_test)

	pc.draw_function()
	print(datetime.datetime.now(), flush=True)
	gc.collect()

	# sample points
	pts: sample.data = sample.data(dist, pc.draw_sample_pts)
	print(datetime.datetime.now(), flush=True)
	gc.collect()

	se.se_trials(pts, pc)
	print(datetime.datetime.now(), flush=True)
	gc.collect()

	pinv_pred: gp.PredType[gpytorch.kernels.Kernel] = se.pinv_pred_generator(pts.x_all_t, pts.y_all_t)
	pinv_model: gp.GPR = plot.plot_lengthscale_error("pinv", pc, pts, se.se_generator(pinv_pred, pts.x_all_t, pts.y_all_t), predictor=pinv_pred) if search_param else plot.generate_and_plot_model("pinv", pc, dist, pts, se.se_generator(pinv_pred, pts.x_all_t, pts.y_all_t), pinv_pred)
	print(datetime.datetime.now(), flush=True)
	gc.collect()

	loocv_model: gp.GPR = plot.plot_lengthscale_error("loocv", pc, pts, se.loocv_se_generator(pts.x_all_t, pts.y_all_t)) if search_param else plot.generate_and_plot_model("loocv", pc, dist, pts, se.loocv_se_generator(pts.x_all_t, pts.y_all_t))
	print(datetime.datetime.now(), flush=True)
	gc.collect()

	local.pinv_local_pred_variance_cutoff_trial(pinv_model, pts, pc)
	print(datetime.datetime.now(), flush=True)
	gc.collect()

	local.pinv_local_pred_density_cutoff_trial(pinv_model, pts, pc)
	print(datetime.datetime.now(), flush=True)
	gc.collect()

	local.pinv_local_pred_deviate_cutoff_trial(pinv_model, pts, pc)
	print(datetime.datetime.now(), flush=True)
	gc.collect()

	local.single_local_parameter_trial(pinv_model, pts, pinv_pred)
	print(datetime.datetime.now(), flush=True)
	gc.collect()

	num_neighbor: int
	nn: sklearn.neighbors.NearestNeighbors
	pts_neighbor_ind: torch.Tensor
	if search_param:
		num_neighbor, nn, pts_neighbor_ind = local.local_pred_trial(pinv_model, pts, pc)
	else:
		num_neighbor = 32
		nn = sklearn.neighbors.NearestNeighbors(n_neighbors=num_neighbor, n_jobs=-1).fit(pts.x)
		pts_neighbor_ind = torch.cat([local.get_central_neighbor_indices(nn, pts.x), local.get_neighbor_ind(pts.x_t, pts.extra_x_t, num_neighbor, nn)])
	print(datetime.datetime.now(), flush=True)
	gc.collect()

	local_predict: gp.PredType[gpytorch.kernels.Kernel] = local.local_predict_generator(num_neighbor, nn, pts_neighbor_ind)
	local_model: gp.GPR = plot.plot_lengthscale_error("local", pc, pts, se.se_generator(local_predict, pts.x_all_t, pts.y_all_t), predictor=local_predict) if search_param else plot.generate_and_plot_model("local", pc, dist, pts, se.se_generator(local_predict, pts.x_all_t, pts.y_all_t), local_predict)
	print(datetime.datetime.now(), flush=True)
	gc.collect()

	local.local_pinv_pred_trial(local_model, pts, pc, num_neighbor, nn, pts_neighbor_ind)
	print(datetime.datetime.now(), flush=True)
	gc.collect()

	mix.mix_pred_trial(pts, pc, num_neighbor, nn, pts_neighbor_ind)
	print(datetime.datetime.now(), flush=True)
	gc.collect()

	mix.model_mix_trial([loocv_model.cov, pinv_model.cov, local_model.cov], pts, pc, [gp.default_predict, pinv_pred, local_predict])
	print(datetime.datetime.now(), flush=True)
	gc.collect()

	mix.each_local_parameter_trial(pts, pc, pts_neighbor_ind)
	print(datetime.datetime.now(), flush=True)
	gc.collect()

	linear.linear_trial(dist.rng)
	print(datetime.datetime.now(), flush=True)
	gc.collect()


if __name__ == "__main__":
	if len(sys.argv) > 1:
		for seed in sys.argv[1:]:
			main(int(seed))
	else:
		main()