#!/usr/bin/env python3

"""
main
====

The main module.
"""
import argparse
import datetime
import gc
import io
import os
import subprocess
import sys
import tarfile
import time
import traceback
import typing

import numpy as np
import numpy.typing as npt
import scipy.interpolate

sys.path.append(os.path.dirname(__file__))

import expectation
import gp
import pes
import plot
import point
import utility

NUM_MC_PTS = 1_000_000
NUM_EVL_MC_PTS = 10_000


def parse_argument() -> tuple[bool, str]:
	"""
	To parse arguments

	Returns
	-------
	tuple[bool, str]
		Whether to plot or not, and whether to read grid solution or not
	"""
	parser: argparse.ArgumentParser = argparse.ArgumentParser(description="To evolve the grid solution")
	parser.add_argument("--plot", "-p", action="store_true", help="Whether to plot each frame or not")
	parser.add_argument("--read", "-r", default="", type=str, help="Whether to read grid solution or not; if read, provide the file name")
	result: dict[str, typing.Any] = vars(parser.parse_args())
	return result["plot"], result["read"]


def cutoff(x: float) -> float:
	"""
	To calculate the cutoff of the input

	Parameters
	----------
	x : float
		Any double-precision value

	Returns
	-------
	float
		Its cutoff

	Notes
	-----
	The function calculates the cutoff to closest 1eN, 2eN, 5eN. e.g.: 0.11->0.1, 8.2->5, 3626->2000
	"""
	# calculate the number of digits, or N
	log_x: float = np.log10(np.abs(x))
	n: int = int(np.floor(log_x))
	pow_x: int = 10 ** n
	# the resume
	resume: float = x / pow_x
	# choose the value: 1, 2, 5
	if resume < 2.0:
		return pow_x
	elif resume < 5.0:
		return 2.0 * pow_x
	else:
		return 5.0 * pow_x


def main(to_draw: bool, grid_solution_file: str) -> None:
	"""
	The main routine

	Parameters
	----------
	to_draw : bool
		Whether to draw it or not
	grid_solution_file : str
		The file name for grid solution. If empty string, grid solution will not be read

	Raises
	------
	NotImplementedError
		In case an unimplemented branch is reached
	"""
	gc.set_debug(gc.DEBUG_UNCOLLECTABLE | gc.DEBUG_SAVEALL | gc.DEBUG_STATS)
	if gc.isenabled():
		gc.disable()
	mass: npt.NDArray[np.double]
	r0: npt.NDArray[np.double]
	sigma_r0: npt.NDArray[np.double]
	dx: npt.NDArray[np.double]
	initial_population: npt.NDArray[np.double]
	initial_phase_factor: npt.NDArray[np.double]
	output_interval: float
	reopt_interval: float
	dt: float
	mass, r0, sigma_r0, dx, initial_population, initial_phase_factor, output_interval, reopt_interval, dt = plot.read_input()
	dx = np.minimum(dx, np.pi * pes.HBAR / (r0[pes.DIM:] + 3.0 * sigma_r0[pes.DIM:])) # 2 grids per de Broglie wavelength
	dt = cutoff(min(dt, pes.HBAR / (pes.V_MAX + (np.pi * pes.HBAR) ** 2 / 2.0 * (1.0 / mass / dx ** 2).sum())))
	dx = np.vectorize(cutoff)(dx)
	output_steps: int = int(round(output_interval / dt))
	reopt_steps: int = int(round(reopt_interval / output_interval))
	print(
		utility.format_array("Mass", mass),
		utility.format_array("Initial center", r0),
		utility.format_array("Initial deviation", sigma_r0),
		utility.format_array("Initial population", initial_population),
		utility.format_array("Initial phase_factor", initial_phase_factor),
		utility.format_array("Time step", dt),
		utility.format_array("Steps between output", output_steps),
		utility.format_array("Steps between optimization", reopt_steps),
		sep="\n"
	)
	initial_phase_factor = initial_phase_factor / 180.0 * np.pi # deg to arc
	n_grids: int = 4 * int(np.max(np.abs(r0[:pes.DIM] / dx))) + 1
	grids_each_dim: list[npt.NDArray[np.double]] = plot.get_grids(r0, n_grids)
	grid_coord: npt.NDArray[np.double] | None = None
	try:
		grid_coord = np.array(np.meshgrid(*grids_each_dim), dtype=np.double).reshape(pes.PHASEDIM, -1).T # shape of pes.PHASEDIM * (N_GRIDS ** pes.PHASEDIM)
	finally: # in case the memory requirement is too big
		pass
	total_ticks: int = int(np.floor(np.max(2.0 * np.abs(r0[:pes.DIM]) / (np.abs(r0[pes.DIM:]) / mass)))) * 2 + 1
	# read input
	pred: npt.NDArray[np.double] | None = None
	grid_data: npt.NDArray[np.double] | None = None
	if grid_coord is not None:
		try:
			pred = np.empty((pes.NUM_PES, pes.NUM_PES) + (n_grids,) * pes.PHASEDIM, np.double)
		finally: # in case the memory requirement is too big
			pass
	if grid_solution_file != "" and pred is not None:
		try:
			grid_data = plot.read_data(grid_solution_file) # shape of (N_TICKS * NUM_ELM * N_GRIDS...)
			assert n_grids == grid_data.shape[-1]
			total_ticks = grid_data.shape[0]
		except AssertionError: # assert n_grids == grid_data.shape[-1] fails
			grid_data = None # do not use grid data either
		finally:
			pass
	# sampling. Initial sample from gaussian directly
	init_dist: pes.InitialDistribution = pes.InitialDistribution(r0, sigma_r0, initial_population, initial_phase_factor)
	pts: point.Points = point.Points(init_dist)
	# the regressor
	predictors: gp.GPRPredictors = gp.GPRPredictors(sigma_r0)
	# average evaluators
	mca: expectation.MonteCarloAverage = expectation.MonteCarloAverage(NUM_MC_PTS)
	aia: expectation.AnalyticalAverager = expectation.AnalyticalAverager(predictors)
	epmca: expectation.EvolvingPointsMCAverage = expectation.EvolvingPointsMCAverage(NUM_EVL_MC_PTS, init_dist)
	# drawer
	dm_drawer: plot.DensityMatrixDrawer | None = None
	wfn_plotter: plot.WavefunctionPlotter | None = None
	if to_draw:
		if pes.PHASEDIM == 2 and pred is not None: # None prediction means nothing provided for drawing
			dm_drawer = plot.DensityMatrixDrawer(
				output_interval,
				total_ticks,
				init_dist,
				grids_each_dim[0],
				grids_each_dim[1],
				grid_data=grid_data,
				draw_scattered=True,
				draw_rescaled=True
			)
		wfn_plotter = plot.WavefunctionPlotter(
			output_interval,
			total_ticks,
			init_dist,
			grids_each_dim,
			draw_rescaled=False
		)
	# files for output
	pts_f: io.TextIOWrapper
	bln_f: io.TextIOWrapper
	all_f: io.TextIOWrapper
	mgn_f: io.TextIOWrapper
	ave_f: io.TextIOWrapper
	den_f: io.TextIOWrapper
	err_f: io.TextIOWrapper
	prm_f: io.TextIOWrapper
	scl_f: io.TextIOWrapper
	lss_f: io.TextIOWrapper
	with open(plot.POINTS_FILENAME + plot.DATA_EXTENSION, "w", encoding="UTF-8") as pts_f,\
		open(plot.BELONGING_FILENAME + plot.DATA_EXTENSION, "w", encoding="UTF-8") as bln_f,\
		open(plot.ALL_GRIDS_FILENAME + plot.DATA_EXTENSION, "w", encoding="UTF-8") as all_f,\
		open(plot.MARGINAL_FILENAME + plot.DATA_EXTENSION, "w", encoding="UTF-8") as mgn_f,\
		open(plot.AVERAGE_FILENAME + plot.DATA_EXTENSION, "w", encoding="UTF-8") as ave_f,\
		open("density" + plot.DATA_EXTENSION, "w", encoding="UTF-8") as den_f,\
		open(plot.ERROR_FILENAME + plot.DATA_EXTENSION, "w", encoding="UTF-8") as err_f,\
		open(plot.PARAMETER_FILENAME + plot.DATA_EXTENSION, "w", encoding="UTF-8") as prm_f,\
		open(plot.SCALE_FILENAME + plot.DATA_EXTENSION, "w", encoding="UTF-8") as scl_f,\
		open(plot.LOSS_FILENAME + plot.DATA_EXTENSION, "w", encoding="UTF-8") as lss_f:
		def train_pred_draw(iTick: int, to_train: bool = True) -> None:
			"""
			To train the parameter, doing prediction on all grids, and draw it

			Parameters
			----------
			iTick : int
				Current time tick, to access grid data and for plotting
			to_train : bool, optional
				Whether to adjust parameters or not, by default True

			Raises
			------
			NotImplementedError
				In case an unimplemented branch is reached
			"""
			# get scale
			scale: npt.NDArray[np.double] = pts.rescale_factor
			print(f"Tick {iTick}, {utility.format_array("scales", scale)}, {datetime.datetime.now()}", flush=True)
			# save points
			# index of central points corresponds to the element
			# index of extra points is the element index + num_elements
			# the first N points are central points, and then N*ratio points are the extra points
			np.savetxt(pts_f, np.concatenate(pts.center).T, footer="\n", comments="") # save as PHASEDIM * ALL_PTS
			pts.print_belonging(bln_f)
			# fit
			predictors.update(pts.center, pts.density, pts.num_center, scale)
			if to_train:
				predictors.train(ftol=1e-9, xtol=1e-5)
			# predict and marginal distribution
			marginal: npt.NDArray[np.double] = np.empty((pes.PHASEDIM, pes.NUM_PES, pes.NUM_PES, n_grids), np.double)
			for iPES in range(pes.NUM_PES):
				for jPES in range(iPES + 1):
					ElementIndex: int = iPES * pes.NUM_PES + jPES
					pred_element: npt.NDArray[np.cdouble] | None = None
					if grid_coord is not None and pred is not None:
						try:
							pred_element = predictors.predict(grid_coord, ElementIndex).reshape((n_grids,) * pes.PHASEDIM)
						finally:
							pass
					marginal_pred_element: npt.NDArray[np.cdouble] = np.array([predictors.get_marginal(iDim, grids_each_dim[iDim].reshape(-1, 1), ElementIndex) for iDim in range(pes.PHASEDIM)]) # shape of (PHASEDIM, N_GRIDS)
					if iPES == jPES:
						if pred is not None and pred_element is not None:
							pred[iPES, jPES] = pred_element.real
						marginal[:, iPES, jPES, :] = marginal_pred_element.real
					else:
						if pred is not None and pred_element is not None:
							pred[jPES, iPES] = pred_element.real
							pred[iPES, jPES] = pred_element.imag
						marginal[:, jPES, iPES, :] = marginal_pred_element.real
						marginal[:, iPES, jPES, :] = marginal_pred_element.imag
			if pred is not None:
				np.savetxt(all_f, pred.reshape(pes.NUM_ELM, n_grids ** pes.PHASEDIM), footer="\n", comments="")
			np.savetxt(mgn_f, marginal.reshape(pes.PHASEDIM * pes.NUM_ELM, n_grids), footer="\n", comments="")
			# calculate averages
			print(iTick * output_interval, end=" ", file=ave_f)
			mca.update_pts(pts.center, lambda x, idx: predictors.predict(x, idx, False))
			epmca.update_density(predictors.predict)
			aver: expectation.Averager
			for aver in [mca, aia, epmca]:
				print(*aver.population(), *aver.coordinates(), *aver.covariance()[np.tril_indices(pes.PHASEDIM)], aver.potential(), aver.kinetic(mass), *aver.purity().reshape(-1), end=" ", file=ave_f)
			print("", file=ave_f, flush=True)
			# calculate error, and predict density
			evolving_density: list[npt.NDArray[np.cdouble]] = pts.density
			if grid_data is not None: # interpolate the data
				grid_interpolate: list[npt.NDArray[np.cdouble]] = [np.array([], dtype=np.cdouble) for _ in range(pes.NUM_TRIG)]
				evolving_errors: npt.NDArray[np.double] = np.empty((pes.NUM_PES, pes.NUM_PES), np.double)
				for iPES in range(pes.NUM_PES):
					for jPES in range(iPES + 1):
						TrilIndex: int = pes.flatten_tril_index[iPES, jPES]
						interpolator_re: scipy.interpolate.RegularGridInterpolator = scipy.interpolate.RegularGridInterpolator(
							tuple(grids_each_dim),
							grid_data[iTick, jPES * pes.NUM_PES + iPES], # upper part
							"cubic",
							False,
							0.0
						)
						grid_interpolate[TrilIndex] = interpolator_re(pts.center[TrilIndex]).astype(np.cdouble)
						if iPES == jPES:
							evolving_errors[iPES, jPES] = np.sum((grid_interpolate[TrilIndex].real - evolving_density[TrilIndex].real) ** 2) / evolving_density[TrilIndex].size
						else:
							interpolator_im: scipy.interpolate.RegularGridInterpolator = scipy.interpolate.RegularGridInterpolator(
								tuple(grids_each_dim),
								grid_data[iTick, iPES * pes.NUM_PES + jPES], # strictly lower part
								"cubic",
								False,
								0.0
							)
							grid_interpolate[TrilIndex].imag = interpolator_im(pts.center[TrilIndex])
							evolving_errors[jPES, iPES] = np.sum((grid_interpolate[TrilIndex].real - evolving_density[TrilIndex].real) ** 2) / evolving_density[TrilIndex].size
							evolving_errors[iPES, jPES] = np.sum((grid_interpolate[TrilIndex].imag - evolving_density[TrilIndex].imag) ** 2) / evolving_density[TrilIndex].size
				evolving_errors = evolving_errors.reshape(-1)
				assert pred is not None # grid data will only be read when already enough space for prediction
				diff: npt.NDArray[np.double] = pred.reshape((pes.NUM_ELM,) + pred.shape[2:]) - grid_data
				original_errors: npt.NDArray[np.double] = np.sum(diff ** 2, (-2, -1))
				rescaled_errors: npt.NDArray[np.double] = original_errors * scale ** 2
				np.savetxt(err_f, (original_errors, rescaled_errors, evolving_errors), footer="\n", comments="")
				np.savetxt(den_f, np.concatenate(grid_interpolate).view(np.double).reshape(-1, 2).T) # 2 stands for real and imag
			else:
				np.savetxt(den_f, np.concatenate(evolving_density).view(np.double).reshape(-1, 2).T)
			np.savetxt(den_f, np.concatenate(evolving_density).view(np.double).reshape(-1, 2).T)
			np.savetxt(den_f, np.concatenate([predictors.predict(pt, idx) for pt, idx in zip(pts.center, pes.tril_element_indices)]).view(np.double).reshape(-1, 2).T, footer="\n", comments="")
			print("\n", end="\n", file=den_f)
			if to_draw: # plots used whether grid solution is given or not
				if pes.PHASEDIM == 2:
					assert pred is not None and dm_drawer is not None
					dm_drawer(iTick, pred.reshape((pes.NUM_ELM,) + pred.shape[2:]), pts.center, pts.num_center, scale)
				assert wfn_plotter is not None
				wfn_plotter(iTick, marginal.diagonal(axis1=1, axis2=2).swapaxes(-1, -2)) # .diagonal will move axis to end

		def print_parameter_scale_loss(scale: npt.NDArray[np.double]) -> None:
			"""
			To print parameters, rescale factor, and loss on sample points to file

			Parameters
			----------
			scale : npt.NDArray[np.double]
				Rescale factor
			"""
			predictors.print(prm_f)
			print("\n", file=prm_f)
			np.savetxt(scl_f, scale)
			print("\n", file=scl_f)
			for i in range(pes.NUM_ELM):
				print(predictors[i].error().item(), file=lss_f)
			print("\n", file=lss_f)

		gc.collect()
		try:
			start_time: int = int(time.time())
			end_time = os.environ.get("SLURM_JOB_END_TIME")
			if end_time is not None:
				end_time = int(end_time) # int | None
			train_pred_draw(0)
			print_parameter_scale_loss(pts.rescale_factor)
			to_stop: bool = False
			iTick: int
			for iTick in range(1, total_ticks):
				# evolve
				for _ in range(output_steps):
					epmca.evolve(mass, dt, predictors.predict)
					pts.evolve(mass, dt, predictors.predict, epmca.purity())
					scale: npt.NDArray[np.double] = pts.rescale_factor
					predictors.update(pts.center, pts.density, pts.num_center, scale)
					predictors.train(ftol=1e-4, xtol=1e-4) # default value of scipy simplex
					print_parameter_scale_loss(scale)
				# update and predict
				train_pred_draw(iTick, iTick % reopt_steps == 0)
				print_parameter_scale_loss(pts.rescale_factor)
				# check stopping criteria, when grid solution is not given
				# use predictors (aia) with old points
				if grid_data is None:
					if np.any(epmca.coordinates()[:pes.DIM] > np.abs(r0[:pes.DIM])):
						to_stop = True
				if end_time is not None:
					current_time: int = int(time.time())
					time_pass: int = current_time - start_time
					time_left: int = end_time - current_time
					if time_left < time_pass // iTick:
						# time left is not enough for next output, kill and rerun the job
						print(f"Time left is {time_left} seconds, not enough for another iteration. Stop evolving after {time_pass} seconds, {iTick} iterations")
						to_stop = True
				if to_stop:
					total_ticks = iTick + 1
					break
				gc.collect()
		except Exception as e:
			traceback.print_exception(e, file=sys.stdout)

	# then plots
	gc.collect()
	plot.plot_average() # averages
	if grid_data is not None: # error
		plot.plot_error(output_interval)
	# parameters, loss and rescale factor
	param_loss_scale_ticks: npt.NDArray[np.double] = np.concatenate([np.arange(i * output_steps, (i + 1) * output_steps + 1) for i in range(iTick)]) * dt # This should be size of (total_ticks - 1) * (output_steps + 1)
	plot.plot_parameters(param_loss_scale_ticks)
	plot.plot_loss_and_rescale_factors(param_loss_scale_ticks)
	# tar figures
	if to_draw:
		if pes.PHASEDIM == 2:
			assert dm_drawer is not None
			if os.path.isdir(dm_drawer.FILENAME_PREFIX):
				os.rename(dm_drawer.FILENAME_PREFIX, dm_drawer.FILENAME_PREFIX + "_" + str(datetime.datetime.now()).replace(" ", "_"))
			subprocess.run(["mkdir", dm_drawer.FILENAME_PREFIX]) # make directory
			subprocess.run(["mv"] + [dm_drawer.picname.format(iTick) for iTick in range(total_ticks)] + [dm_drawer.FILENAME_PREFIX])
			with tarfile.open(dm_drawer.FILENAME_PREFIX + plot.TAR_EXTENSION, "w:gz") as dm_tf:
				dm_tf.add(dm_drawer.FILENAME_PREFIX)
		assert wfn_plotter is not None
		if os.path.isdir(wfn_plotter.FILENAME_PREFIX):
			os.rename(wfn_plotter.FILENAME_PREFIX, wfn_plotter.FILENAME_PREFIX + "_" + str(datetime.datetime.now()).replace(" ", "_"))
		subprocess.run(["mkdir", wfn_plotter.FILENAME_PREFIX]) # make directory
		subprocess.run(["mv"] + [wfn_plotter.picname.format(iTick) for iTick in range(total_ticks)] + [wfn_plotter.FILENAME_PREFIX])
		with tarfile.open(wfn_plotter.FILENAME_PREFIX + plot.TAR_EXTENSION, "w:gz") as wfn_tf:
			wfn_tf.add(wfn_plotter.FILENAME_PREFIX)


if __name__ == "__main__":
	main(*parse_argument())
