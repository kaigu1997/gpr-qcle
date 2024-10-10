#!/usr/bin/env python

# imports
import datetime
import gc
import os
import sys

import sklearn.neighbors
import torch

torch.set_default_dtype(torch.float64)

sys.path.append(os.path.dirname(__file__))

import gp
import local
import plot
import sample
import se


def main(seed: int = 0) -> None:
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

	pinv_model: gp.GPR = plot.plot_lengthscale_error("pinv", pc, pts, se.approx_se, predictor=se.pinv_pred, x_all=pts.x_all_t, y_all=pts.y_all_t)
	print(datetime.datetime.now(), flush=True)
	gc.collect()

	loocv_model: gp.GPR = plot.plot_lengthscale_error("loocv", pc, pts, se.loocv_se, x_all=pts.x_all_t, y_all=pts.y_all_t)
	print(datetime.datetime.now(), flush=True)
	gc.collect()

	local.pinv_local_pred_variance_cutoff_trial(pinv_model, pts, pc)
	print(datetime.datetime.now(), flush=True)
	gc.collect()

	local.pinv_local_pred_deviate_cutoff_trial(pinv_model, pts, pc)
	print(datetime.datetime.now(), flush=True)
	gc.collect()

	local.pinv_local_pred_density_cutoff_trial(pinv_model, pts, pc)
	print(datetime.datetime.now(), flush=True)
	gc.collect()

	num_neighbor: int
	nn: sklearn.neighbors.NearestNeighbors
	central_neighbor_ind: torch.Tensor
	num_neighbor, nn, central_neighbor_ind = local.local_pred_trial(pinv_model, pts, pc)
	print(datetime.datetime.now(), flush=True)
	gc.collect()
	# num_neighbor: int = 32
	# nn: sklearn.neighbors.NearestNeighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=num_neighbor, n_jobs=-1).fit(pts.extra_x)
	# central_neighbor_ind: torch.Tensor = local.get_central_neighbor_indices(nn, pts.extra_x)

	local_model: gp.GPR = plot.plot_lengthscale_error("local", pc, pts, se.approx_se, predictor=local.local_predict, x_all=pts.extra_x_t, y_all=pts.extra_y_t, num_neighbor=num_neighbor, nn=nn, neighbor_ind=central_neighbor_ind)
	print(datetime.datetime.now(), flush=True)
	gc.collect()

	local.local_pinv_pred_trial(local_model, pts, pc, num_neighbor, nn)
	print(datetime.datetime.now(), flush=True)
	gc.collect()

	local.local_global_mix_trial(local_model, pts, pc, num_neighbor, nn, central_neighbor_ind)
	print(datetime.datetime.now(), flush=True)
	gc.collect()

	local.local_separate_kernel_trial(pts, pc, num_neighbor)
	print(datetime.datetime.now(), flush=True)
	gc.collect()

if __name__ == "__main__":
	if len(sys.argv) > 1:
		for seed in sys.argv[1:]:
			main(int(seed))
	else:
		main()