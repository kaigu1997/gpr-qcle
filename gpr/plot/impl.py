r"""plot.impl
=========
This module plots all results
"""
import argparse
import collections.abc
import datetime
import typing

import matplotlib
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import constant
import param
import pes

from . import dm, utility, wfn


def plot_average(config: pes.ModelConfig) -> npt.NDArray[np.double]:
	"""
	To plot all averages available

	Parameters
	----------
	config : pes.ModelConfig
		Configuration of the model

	Returns
	-------
	npt.NDArray[np.double]
		Times when output
	"""
	NUM_MOMENTS: typing.Final = config.PHASEDIM * (config.PHASEDIM + 3) // 2
	ave_type_titles: typing.Final[list[str]] = ["Monte Carlo", "Analytical", "Evolving Points Monte Carlo"]
	NUM_AVE_TYPES: typing.Final[int] = len(ave_type_titles)
	plot_titles: typing.Final[list[str]] = ["Population"]\
		+ [utility.dimension_name(iDim, config.DIM) for iDim in config.PHASEDIM_RANGE]\
		+ [("Cov(" + utility.dimension_name(jDim, config.DIM) + ", " + utility.dimension_name(iDim, config.DIM) + ")") if iDim != jDim else ("Var[" + utility.dimension_name(iDim, config.DIM) + "]") for iDim in config.PHASEDIM_RANGE for jDim in range(iDim + 1)]\
		+ ["Energy", "Purity"] # moments of 0th, 1st and 2nd order, and other (energy and purity)
	NUM_TITLE: typing.Final[int] = len(plot_titles)
	fig: matplotlib.figure.Figure = plt.figure(figsize=(constant.FIGSIZE[0] * NUM_TITLE, constant.FIGSIZE[1] * NUM_AVE_TYPES * 2))
	axs = fig.subplots(NUM_AVE_TYPES * 2, NUM_TITLE) # population, <x> and <p>, energy, purity
	assert isinstance(axs, np.ndarray)
	averages_all: typing.Final[npt.NDArray[np.double]] = np.loadtxt(constant.AVERAGE_FILENAME + constant.DATA_EXTENSION)
	ticks: typing.Final[npt.NDArray[np.double]] = averages_all[:, 0]
	averages: typing.Final[npt.NDArray[np.double]] = averages_all[:, 1:].reshape(ticks.size, NUM_AVE_TYPES, (averages_all.shape[1] - 1) // NUM_AVE_TYPES)
	for iRow in range(NUM_AVE_TYPES * 2):
		AVE_TYPE_INDEX: int = iRow // 2
		isOriginal: bool = iRow % 2 == 0
		ppl_sum: float = 1.0 if isOriginal else np.sum(averages[:, AVE_TYPE_INDEX, :config.NUM_PES], -1)
		for iCol in range(NUM_TITLE):
			ax: matplotlib.axes.Axes = axs[iRow, iCol]
			y_label: str = plot_titles[iCol]
			if iCol == 0: # population
				for iPES in config.PES_RANGE:
					ax.plot(ticks, averages[:, AVE_TYPE_INDEX, iPES] / ppl_sum, label="State " + str(iPES))
				ax.plot(ticks, np.sum(averages[:, AVE_TYPE_INDEX, :config.NUM_PES], -1) / ppl_sum, label="Total")
				ax.legend()
			elif iCol <= NUM_MOMENTS: # <x>, <p>, and covariances
				ax.plot(ticks, averages[:, AVE_TYPE_INDEX, iCol - 1 + config.NUM_PES] / ppl_sum)
				y_label += " / a.u."
				if iCol > config.PHASEDIM:
					y_label += r"$^2$"
			elif iCol == NUM_MOMENTS + 1: # energies
				ax.plot(ticks, averages[:, AVE_TYPE_INDEX, config.NUM_PES + NUM_MOMENTS + 1] / ppl_sum, label="Kinetic Energy")
				if AVE_TYPE_INDEX == 0: # only mc has potential and thus total energy
					ax.plot(ticks, averages[:, AVE_TYPE_INDEX, config.NUM_PES + NUM_MOMENTS] / ppl_sum, label="Potential Energy")
					ax.plot(ticks, np.sum(averages[:, AVE_TYPE_INDEX, config.NUM_PES + NUM_MOMENTS:config.NUM_PES + NUM_MOMENTS + 2], -1) / ppl_sum, label="Total Energy")
				ax.legend()
				y_label += " / a.u."
			else: # purity
				for iElement in config.ELEMENT_RANGE:
					ax.plot(ticks, averages[:, AVE_TYPE_INDEX, config.NUM_PES + NUM_MOMENTS + 2 + iElement] / ppl_sum, label=utility.get_RI_label(iElement, config.NUM_PES))
				ax.plot(ticks, np.sum(averages[:, AVE_TYPE_INDEX, config.NUM_PES + NUM_MOMENTS + 2:], -1) / ppl_sum, label="Total")
				ax.legend()
			ax.set_xlabel("Time / a.u.")
			ax.set_ylabel(y_label)
			ax.set_title(ave_type_titles[AVE_TYPE_INDEX] + ("" if isOriginal else " Rescaled") + " Average of " + plot_titles[iCol])
	fig.savefig(constant.AVERAGE_FILENAME + constant.FIGURE_EXTENSION)
	plt.close(fig)
	return ticks


def plot_error(config: pes.ModelConfig, ticks: npt.NDArray[np.double]) -> None:
	r"""To plot the errors: squared error on all grids, its rescaled, and on all sample points

	Parameters
	----------
	config : pes.ModelConfig
		Configuration of the model
	ticks : npt.NDArray[np.double]
		The time since start for each output, in unit of a.u.
	"""
	fig: matplotlib.figure.Figure = plt.figure(figsize=(constant.FIGSIZE[0] * 3, constant.FIGSIZE[1]))
	axs = fig.subplots(1, 3)
	assert isinstance(axs, np.ndarray)
	error: typing.Final[npt.NDArray[np.double]] = np.loadtxt(constant.ERROR_FILENAME + constant.DATA_EXTENSION).reshape(-1, 3, config.NUM_ELM)
	assert ticks.size >= error.shape[0]
	ticks = ticks[:error.shape[0]]
	error_titles: typing.Final[list[str]] = ["Original", "Rescaled", "Evolving"]
	for iPlot in range(axs.size):
		ax: matplotlib.axes.Axes = axs[iPlot]
		for iElement in config.ELEMENT_RANGE:
			ax.semilogy(ticks, error[:, iPlot, iElement], label=utility.get_RI_label(iElement, config.NUM_PES))
		ax.set_xlabel("Time / a.u.")
		ax.set_ylabel("Error")
		ax.legend()
		ax.set_title(error_titles[iPlot] + " Error")
	fig.savefig(constant.ERROR_FILENAME + constant.FIGURE_EXTENSION)
	plt.close(fig)


def plot_parameters(config: pes.ModelConfig, ticks: npt.NDArray[np.double]) -> None:
	r"""To plot parameters

	Parameters
	----------
	config : pes.ModelConfig
		Configuration of the model
	ticks : npt.NDArray[np.double]
		The time since start for each output, in unit of a.u.
	"""
	fig: matplotlib.figure.Figure = plt.figure(figsize=(constant.FIGSIZE[0] * config.PHASEDIM, constant.FIGSIZE[1]))
	axs = fig.subplots(1, config.PHASEDIM)
	assert isinstance(axs, np.ndarray)
	parameters: typing.Final[npt.NDArray[np.double]] = np.loadtxt(constant.PARAMETER_FILENAME + constant.DATA_EXTENSION).reshape(-1, config.NUM_ELM, config.PHASEDIM)[:-1]
	assert ticks.size >= parameters.shape[0]
	parameter_names: typing.Final[list[str]] = [utility.dimension_name(iDim, config.DIM) for iDim in range(config.PHASEDIM)]
	for iDim in range(config.PHASEDIM):
		ax: matplotlib.axes.Axes = axs[iDim]
		for iElement in range(config.NUM_ELM):
			ax.plot(ticks[:parameters.shape[0]], parameters[:, iElement, iDim], label=utility.get_RI_label(iElement, config.NUM_PES))
		ax.set_xlabel("Time / a.u.")
		ax.set_ylabel("Characteristic Length / a.u.")
		ax.set_title("Characteristic Length of " + parameter_names[iDim])
		ax.legend()
	fig.savefig(constant.PARAMETER_FILENAME + constant.FIGURE_EXTENSION)
	plt.close(fig)


def plot_loss_and_rescale_factors(config: pes.ModelConfig, ticks: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
	r"""To plot the difference between prediction and evolution, and the rescale factors of each element

	Parameters
	----------
	config : pes.ModelConfig
		Configuration of the model
	ticks : npt.NDArray[np.double]
		The time since start for each output, in unit of a.u.

	Returns
	-------
	npt.NDArray[np.double], shape of (N_TICKS, NUM_ELM)
		Rescale factors
	"""
	loss_scale_titles: typing.Final[list[str]] = ["Loss on Sample Points", "Rescale Factor"]
	loss_scale_ylabels: typing.Final[list[str]] = ["Loss", "Rescale Factor"]
	fig: matplotlib.figure.Figure = plt.figure(figsize=(constant.FIGSIZE[0] * 2, constant.FIGSIZE[1]))
	axs = fig.subplots(1, 2)
	assert isinstance(axs, np.ndarray)
	loss_scale: typing.Final[npt.NDArray[np.double]] = np.concatenate(
		(
			np.loadtxt(constant.LOSS_FILENAME + constant.DATA_EXTENSION).reshape(-1, config.NUM_ELM, 1),
			np.loadtxt(constant.SCALE_FILENAME + constant.DATA_EXTENSION).reshape(-1, config.NUM_ELM, 1)
		),
		-1
	)[:-1]
	assert ticks.size >= loss_scale.shape[0]
	for iPlot in range(axs.size):
		ax: matplotlib.axes.Axes = axs[iPlot]
		for iElement in range(config.NUM_ELM):
			ax.semilogy(ticks[:loss_scale.shape[0]], loss_scale[:, iElement, iPlot], label=utility.get_RI_label(iElement, config.NUM_PES))
		ax.set_xlabel("Time / a.u.")
		ax.set_ylabel(loss_scale_ylabels[iPlot])
		ax.set_title(loss_scale_titles[iPlot])
		ax.legend()
	fig.savefig(constant.LOSS_FILENAME + "-" + constant.SCALE_FILENAME + constant.FIGURE_EXTENSION)
	plt.close(fig)
	return loss_scale[:, :, 1]


def main(arguments: None | collections.abc.Sequence[str] = None) -> None:
	r"""The plot
	"""
	parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Choice of drawing")
	parser.add_argument("--animation", "--ani", "-a", default=0, type=int, help="Draw animation; if so, provide the interval between frames in ms. Draw frame by frame and compress into tarfile will always be done.")
	parser.add_argument("--scattered", "-s", action="store_true", help="Scatter points on the distribution")
	parser.add_argument("--rescaled", "-r", action="store_true", help="Rescale all elements to same (maximum = 1.0)")
	parser.add_argument("--log-scale", "-l", action="store_true", help="Draw logscale of the PWTDM")
	parser.add_argument("--density-matrix", "--dm", "-d", default=constant.ALL_GRIDS_FILENAME + constant.DATA_EXTENSION, type=str, help="File containing partial Wigner-transformed density matrix from Gaussian process regression")
	parser.add_argument("--marginal", "-m", default=constant.MARGINAL_FILENAME + constant.DATA_EXTENSION, type=str, help="File containing marginal distribution from Gaussian process regression")
	parser.add_argument("--grid-solution", "--grid", "-g", default="", type=str, help="File containing partial Wigner-transformed density matrix from grid solution")
	result: typing.Final[argparse.Namespace] = parser.parse_args(arguments)
	ani_sep: typing.Final = max(int(result.animation), 0)
	draw_rescaled: typing.Final = bool(result.rescaled)
	draw_logscale: typing.Final = bool(result.log_scale)
	draw_scattered: typing.Final = bool(result.scattered)
	grid_solution_filename: typing.Final = str(result.grid_solution)
	dm_data_filename: typing.Final = str(result.density_matrix)
	marginal_filename: typing.Final = str(result.marginal)
	# get quantity, plot average
	quantity: typing.Final = param.Quantity()
	ticks: typing.Final[npt.NDArray[np.double]] = plot_average(quantity.config)
	if grid_solution_filename != "": # error
		plot_error(quantity.config, ticks)
	# parameters, loss and rescale factor
	param_loss_scale_ticks: typing.Final[npt.NDArray[np.double]] = np.concatenate([np.arange(i * quantity.output_ticks, (i + 1) * quantity.output_ticks + 1) for i in range(ticks.size)]) * np.double(quantity.dt) # This should be size of (total_ticks - 1) * (output_steps + 1)
	plot_parameters(quantity.config, param_loss_scale_ticks)
	scale: typing.Final[npt.NDArray[np.double]] = plot_loss_and_rescale_factors(quantity.config, param_loss_scale_ticks)
	# plot from dm: including marginal, or its marginal only
	try:
		if quantity.config.PHASEDIM == dm.DIM_PLOT_DM:
			print("pwtdm: 0", datetime.datetime.now())
			dm_drawer: typing.Final = dm.DMDrawerFromFile(
				quantity,
				draw_rescaled,
				draw_logscale,
				draw_scattered,
				dm_data_filename,
				grid_solution_filename=grid_solution_filename,
				initial_scale=scale[0]
			)
			print("pwtdm:", dm_drawer.frame_index, datetime.datetime.now())
			while dm_drawer(scale[dm_drawer.frame_index]): # pylint: disable=while-used
				print("pwtdm:", dm_drawer.frame_index, datetime.datetime.now())
			if ani_sep != 0: # draw animation
				utility.draw_animation(dm_drawer.picname, dm_drawer.total_ticks, ani_sep)
			# draw frame by frame and combine into a tarfile, removing the pics after tarfile successfully constructed
			utility.tar_files(dm_drawer.picname, dm_drawer.total_ticks)
		print("marginal: 0", datetime.datetime.now())
		dm_marginal_plotter: typing.Final = wfn.DMMarginalPlotterFromFile(
			quantity,
			draw_rescaled,
			marginal_filename,
			grid_solution_filename
		)
		print("marginal:", dm_marginal_plotter.frame_index, datetime.datetime.now())
		while dm_marginal_plotter(): # pylint: disable=while-used
			print("marginal:", dm_marginal_plotter.frame_index, datetime.datetime.now())
		if ani_sep != 0: # draw animation
			utility.draw_animation(dm_marginal_plotter.picname, dm_marginal_plotter.total_ticks, ani_sep)
		# draw frame by frame and combine into a tarfile, removing the pics after tarfile successfully constructed
		utility.tar_files(dm_marginal_plotter.picname, dm_marginal_plotter.total_ticks)
	finally:
		print("Finish plot.", datetime.datetime.now(), flush=True)


if __name__ == "__main__":
	main()
