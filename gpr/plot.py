"""
plot
====
This module plots all results
"""
import argparse
import collections.abc
import datetime
import functools
import math
import os
import subprocess
import sys
import tarfile
import typing

import matplotlib
import matplotlib.animation
import matplotlib.artist
import matplotlib.axes
import matplotlib.colors
import matplotlib.cm
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.transforms
import matplotlib.typing
import numpy as np
import numpy.typing as npt
import PIL.Image

sys.path.append(os.path.dirname(__file__))

import pes
import utility

FIGSIZE: tuple[float, ...] = tuple(matplotlib.rcParams["figure.figsize"])
TIME_TEMPLATE: typing.Literal["Time = {} a.u."] = "Time = {} a.u."
RESCALE_TEMPLATE: typing.Literal["Rescale Factor = {:.6e}"] = "Rescale Factor = {:.6e}"
POINTS_FILENAME: typing.Literal["points"] = "points"
BELONGING_FILENAME: typing.Literal["belonging"] = "belonging"
ALL_GRIDS_FILENAME: typing.Literal["all_grids"] = "all_grids"
MARGINAL_FILENAME: typing.Literal["marginal"] = "marginal"
AVERAGE_FILENAME: typing.Literal["ave"] = "ave"
ERROR_FILENAME: typing.Literal["error"] = "error"
PARAMETER_FILENAME: typing.Literal["parameters"] = "parameters"
SCALE_FILENAME: typing.Literal["scale"] = "scale"
LOSS_FILENAME: typing.Literal["loss"] = "loss"
DATA_EXTENSION: typing.Literal[".txt"] = ".txt"
FIGURE_EXTENSION: typing.Literal[".png"] = ".png"
TAR_EXTENSION: typing.Literal[".tgz"] = ".tgz"


def read_input() -> tuple[npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.double], float, float, float]:
	"""
	To read input

	Returns
	-------
	tuple[npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.double], float, float]
		mass, centers, standard deviation, grid spacing, initial population, initial phase factor, output interval, reoptimization interval and time step
	"""
	with open("input", "r", encoding="UTF-8") as in_f:
		lines: list[str] = in_f.readlines()
		mass: npt.NDArray[np.double] = np.array(lines[3][:-1].split(" "), dtype=np.double)
		assert pes.DIM % mass.size == 0
		mass = np.tile(mass, pes.DIM // mass.size)
		x0: npt.NDArray[np.double] = np.array(lines[5][:-1].split(" "), dtype=np.double)
		assert pes.DIM % x0.size == 0
		x0 = np.tile(x0, pes.DIM // x0.size)
		p0: npt.NDArray[np.double] = np.array(lines[7][:-1].split(" "), dtype=np.double)
		assert pes.DIM % p0.size == 0
		p0 = np.tile(p0, pes.DIM // p0.size)
		sigma_p0: npt.NDArray[np.double] = np.array(lines[9][:-1].split(" "), dtype=np.double)
		assert pes.DIM % sigma_p0.size == 0
		sigma_p0 = np.tile(sigma_p0, pes.DIM // sigma_p0.size)
		dx: npt.NDArray[np.double] = np.array(lines[11][:-1].split(" "), dtype=np.double)
		assert pes.DIM % dx.size == 0
		dx = np.tile(dx, pes.DIM // dx.size)
		init_ppl: npt.NDArray[np.double] = np.array(lines[13][:-1].split(" "), dtype=np.double)
		assert pes.NUM_PES == init_ppl.size
		init_phase: npt.NDArray[np.double] = np.array(lines[15][:-1].split(" "), dtype=np.double)
		assert pes.NUM_PES == init_phase.size
		output_interval: float = float(lines[17])
		reopt_interval: float = float(lines[19])
		dt: float = float(lines[21])
		output_interval = round(output_interval / dt) * dt
		reopt_interval = round(reopt_interval / output_interval) * output_interval
		return mass, np.concatenate((x0, p0)), np.concatenate((pes.HBAR / 2.0 / sigma_p0, sigma_p0)), dx, init_ppl, init_phase, output_interval, reopt_interval, dt


def read_data(filename: str) -> npt.NDArray[np.double]:
	"""
	To read input data

	Parameters
	----------
	filename : str
		The name of the file

	Returns
	-------
	npt.NDArray[np.cdouble], shape of (N_TICKS, NUM_ELM, N_GRIDS...)
		Input data
	"""
	data: npt.NDArray[np.double] = np.loadtxt(filename, np.double)
	ticks: int = data.shape[0] // pes.NUM_ELM
	length: int = int(np.round(np.power(data.shape[1], 1.0 / pes.PHASEDIM)))
	return data.reshape((ticks, pes.NUM_ELM) + (length,) * pes.PHASEDIM)


def get_grids(r0: npt.NDArray[np.double], n_grids: int) -> list[npt.NDArray[np.double]]:
	"""
	To construct grids for x and p

	Parameters
	----------
	r0 : npt.NDArray[np.double], shape of (PHASEDIM,)
		Initial center. Used for determination of the box
	n_grids : int
		The number of grids for each dimension

	Returns
	-------
	list[npt.NDArray[np.double]]
		Grids for positions and momenta
	"""
	xmax: npt.NDArray[np.double] = 2.0 * np.abs(r0[:pes.DIM])
	xmin: npt.NDArray[np.double] = -xmax
	dx: npt.NDArray[np.double] = (xmax - xmin) / (n_grids - 1)
	pmin: npt.NDArray[np.double] = r0[pes.DIM:] - np.pi * pes.HBAR / 2.0 / dx
	pmax: npt.NDArray[np.double] = r0[1] + np.pi * pes.HBAR / 2.0 / dx
	return [arr for arr in np.linspace(xmin, xmax, n_grids, True).T] + [arr for arr in np.linspace(pmin, pmax, n_grids, True).T]


def get_rescale_factor(data: npt.NDArray[np.double] | list[npt.NDArray[np.cdouble]]) -> npt.NDArray[np.double]:
	"""
	To calculate the scale factor from the given data.

	The scale would be 1 / max() in general, or 0 if all are 0

	Parameters
	----------
	data : npt.NDArray[np.double], of shape (NUM_ELM, n_grids, n_grids) | list[npt.NDArray[np.cdouble]], len of NUM_TRIG, each of shape (num_points,)
		The data

	Returns
	-------
	npt.NDArray[np.double], of shape (NUM_ELM,)
		Scale of each element
	"""
	if isinstance(data, list):
		result: npt.NDArray[np.double] = np.empty((pes.NUM_PES, pes.NUM_PES), np.double)
		for iPES in range(pes.NUM_PES):
			for jPES in range(iPES + 1):
				TrilIndex: int = pes.flatten_tril_index[iPES, jPES]
				if iPES == jPES:
					result[iPES, jPES] = 0 if np.all(data[TrilIndex] == 0.0) else 1.0 / np.max(np.abs(data[TrilIndex]))
				else:
					result[iPES, jPES] = 0 if np.all(data[TrilIndex].imag == 0.0) else 1.0 / np.max(np.abs(data[TrilIndex].imag))
					result[jPES, iPES] = 0 if np.all(data[TrilIndex].real == 0.0) else 1.0 / np.max(np.abs(data[TrilIndex].real))
		return result.reshape(-1)
	else:
		with np.errstate(divide="ignore"):
			return np.where(np.any(data.reshape(pes.NUM_ELM, -1) != 0, -1), 1.0 / np.max(np.abs(data.reshape(pes.NUM_ELM, -1)), -1), 0)


def plot_average() -> None:
	"""
	To plot all averages available
	"""
	NUM_MOMENTS: typing.Literal[5] = pes.PHASEDIM * (pes.PHASEDIM + 3) // 2
	ave_type_titles: list[str] = ["Monte Carlo", "Analytical", "Evolving Points Monte Carlo"]
	NUM_AVE_TYPES: int = len(ave_type_titles)
	plot_titles: list[str] = ["Population"]\
		+ [utility.dimension_name(iDim) for iDim in range(pes.PHASEDIM)]\
		+ [("Cov(" + utility.dimension_name(jDim) + ", " + utility.dimension_name(iDim) + ")") if iDim != jDim else ("Var[" + utility.dimension_name(iDim) + "]") for iDim in range(pes.PHASEDIM) for jDim in range(iDim + 1)]\
		+ ["Energy", "Purity"] # moments of 0th, 1st and 2nd order, and other (energy and purity)
	NUM_TITLE: int = len(plot_titles)
	fig: matplotlib.figure.Figure = plt.figure(figsize=(FIGSIZE[0] * NUM_TITLE, FIGSIZE[1] * NUM_AVE_TYPES * 2))
	axs = fig.subplots(NUM_AVE_TYPES * 2, NUM_TITLE) # population, <x> and <p>, energy, purity
	assert isinstance(axs, np.ndarray)
	averages: npt.NDArray[np.double] = np.loadtxt(AVERAGE_FILENAME + DATA_EXTENSION)
	ticks: npt.NDArray[np.double] = averages[:, 0]
	averages = averages[:, 1:].reshape(ticks.size, NUM_AVE_TYPES, (averages.shape[1] - 1) // NUM_AVE_TYPES)
	for iRow in range(NUM_AVE_TYPES * 2):
		AVE_TYPE_INDEX: int = iRow // 2
		isOriginal: bool = iRow % 2 == 0
		ppl_sum: float = 1.0 if isOriginal else np.sum(averages[:, AVE_TYPE_INDEX, :pes.NUM_PES], -1)
		for iCol in range(NUM_TITLE):
			ax: matplotlib.axes.Axes = axs[iRow, iCol]
			y_label: str = plot_titles[iCol]
			if iCol == 0: # population
				for iPES in range(pes.NUM_PES):
					ax.plot(ticks, averages[:, AVE_TYPE_INDEX, iPES] / ppl_sum, label="State " + str(iPES))
				ax.plot(ticks, np.sum(averages[:, AVE_TYPE_INDEX, :pes.NUM_PES], -1) / ppl_sum, label="Total")
				ax.legend()
			elif iCol <= NUM_MOMENTS: # <x>, <p>, and covariances
				ax.plot(ticks, averages[:, AVE_TYPE_INDEX, iCol - 1 + pes.NUM_PES] / ppl_sum)
				y_label += " / a.u."
				if iCol > pes.PHASEDIM:
					y_label += r"$^2$"
			elif iCol == NUM_MOMENTS + 1: # energies
				ax.plot(ticks, averages[:, AVE_TYPE_INDEX, pes.NUM_PES + NUM_MOMENTS + 1] / ppl_sum, label="Kinetic Energy")
				if AVE_TYPE_INDEX == 0: # only mc has potential and thus total energy
					ax.plot(ticks, averages[:, AVE_TYPE_INDEX, pes.NUM_PES + NUM_MOMENTS] / ppl_sum, label="Potential Energy")
					ax.plot(ticks, np.sum(averages[:, AVE_TYPE_INDEX, pes.NUM_PES + NUM_MOMENTS:pes.NUM_PES + NUM_MOMENTS + 2], -1) / ppl_sum, label="Total Energy")
				ax.legend()
				y_label += " / a.u."
			else: # purity
				for iElement in range(pes.NUM_ELM):
					ax.plot(ticks, averages[:, AVE_TYPE_INDEX, pes.NUM_PES + NUM_MOMENTS + 2 + iElement] / ppl_sum, label=utility.get_RI_label(iElement))
				ax.plot(ticks, np.sum(averages[:, AVE_TYPE_INDEX, pes.NUM_PES + NUM_MOMENTS + 2:], -1) / ppl_sum, label="Total")
				ax.legend()
			ax.set_xlabel("Time / a.u.")
			ax.set_ylabel(y_label)
			ax.set_title(ave_type_titles[AVE_TYPE_INDEX] + ("" if isOriginal else " Rescaled") + " Average of " + plot_titles[iCol])
	fig.savefig(AVERAGE_FILENAME + FIGURE_EXTENSION)
	plt.close(fig)


def plot_error(output_interval: float) -> None:
	"""
	To plot the errors: squared error on all grids, its rescaled, and on all sample points

	Parameters
	----------
	output_interval : float
		Interval between outputs, in unit of a.u.
	"""
	fig: matplotlib.figure.Figure = plt.figure(figsize=(FIGSIZE[0] * 3, FIGSIZE[1]))
	axs = fig.subplots(1, 3)
	assert isinstance(axs, np.ndarray)
	error: npt.NDArray[np.double] = np.loadtxt(ERROR_FILENAME + DATA_EXTENSION).reshape(-1, 3, pes.NUM_ELM)
	ticks: npt.NDArray[np.double] = np.arange(error.shape[0]) * output_interval
	error_titles: list[str] = ["Original", "Rescaled", "Evolving"]
	for iPlot in range(axs.size):
		ax: matplotlib.axes.Axes = axs[iPlot]
		for iElement in range(pes.NUM_ELM):
			ax.semilogy(ticks, error[:, iPlot, iElement], label=utility.get_RI_label(iElement))
		ax.set_xlabel("Time / a.u.")
		ax.set_ylabel("Error")
		ax.legend()
		ax.set_title(error_titles[iPlot] + " Error")
	fig.savefig(ERROR_FILENAME + FIGURE_EXTENSION)
	plt.close(fig)


def plot_parameters(ticks: npt.NDArray[np.double]) -> None:
	"""
	To plot parameters

	Parameters
	----------
	ticks : npt.NDArray[np.double]
		The time since start for each output, in unit of a.u.
	"""
	fig: matplotlib.figure.Figure = plt.figure(figsize=(FIGSIZE[0] * pes.PHASEDIM, FIGSIZE[1]))
	axs = fig.subplots(1, pes.PHASEDIM)
	assert isinstance(axs, np.ndarray)
	parameters: npt.NDArray[np.double] = np.loadtxt(PARAMETER_FILENAME + DATA_EXTENSION).reshape(-1, pes.NUM_ELM, pes.PHASEDIM)[:-1]
	assert ticks.size == parameters.shape[0]
	parameter_names: list[str] = [utility.dimension_name(iDim) for iDim in range(pes.PHASEDIM)]
	for iDim in range(pes.PHASEDIM):
		ax: matplotlib.axes.Axes = axs[iDim]
		for iElement in range(pes.NUM_ELM):
			ax.plot(ticks, parameters[:, iElement, iDim], label=utility.get_RI_label(iElement))
		ax.set_xlabel("Time / a.u.")
		ax.set_ylabel("Characteristic Length / a.u.")
		ax.set_title("Characteristic Length of " + parameter_names[iDim])
		ax.legend()
	fig.savefig(PARAMETER_FILENAME + FIGURE_EXTENSION)
	plt.close(fig)


def plot_loss_and_rescale_factors(ticks: npt.NDArray[np.double]) -> None:
	"""
	To plot the difference between prediction and evolution, and the rescale factors of each element

	Parameters
	----------
	ticks : npt.NDArray[np.double]
		_description_
	"""
	loss_scale_titles: list[str] = ["Loss on Sample Points", "Rescale Factor"]
	loss_scale_ylabels: list[str] = ["Loss", "Rescale Factor"]
	fig: matplotlib.figure.Figure = plt.figure(figsize=(FIGSIZE[0] * 2, FIGSIZE[1]))
	axs = fig.subplots(1, 2)
	assert isinstance(axs, np.ndarray)
	loss_scale: npt.NDArray[np.double] = np.concatenate(
		(
			np.loadtxt(LOSS_FILENAME + DATA_EXTENSION).reshape(-1, pes.NUM_ELM, 1),
			np.loadtxt(SCALE_FILENAME + DATA_EXTENSION).reshape(-1, pes.NUM_ELM, 1)
		),
		-1
	)[:-1]
	assert ticks.size == loss_scale.shape[0]
	for iPlot in range(axs.size):
		ax: matplotlib.axes.Axes = axs[iPlot]
		for iElement in range(pes.NUM_ELM):
			ax.semilogy(ticks, loss_scale[:, iElement, iPlot], label=utility.get_RI_label(iElement))
		ax.set_xlabel("Time / a.u.")
		ax.set_ylabel(loss_scale_ylabels[iPlot])
		ax.set_title(loss_scale_titles[iPlot])
		ax.legend()
	fig.savefig(LOSS_FILENAME + "-" + SCALE_FILENAME + FIGURE_EXTENSION)
	plt.close(fig)


class PNLogNorm(matplotlib.colors.Normalize):
	"""
	Log scale for positive/negative values, e.g. (-1, -0.1, -0.01, 0, 0.01, 0.1, 1.0)

	Parameters
	----------
	abs_min : float, optional
		The absolute value minimum of the given dataset, by default sys.float_info.min
	abs_max : float, optional
		The absolute value maximum of the given dataset, by default sys.float_info.max
	clip : bool, optional
		Determines the behavior for mapping values outside the range, by default False

	Methods
	-------
	inverse(value)
		To map a value generally in [0, 1] back to [`-abs_max`, `abs_max`]
	"""
	__slots__ = ("__abs_min", "__abs_max", "__log2_abs_min", "__log2_abs_max", "__log2_diff")

	def __init__(self, abs_min: float = sys.float_info.min, abs_max: float = sys.float_info.max, clip: bool = False):
		assert 0 < abs_min < abs_max
		super().__init__(-abs_max, abs_max, clip)
		self.__abs_min: float = abs_min
		self.__abs_max: float = abs_max
		self.__log2_abs_min: float = math.log2(abs_min)
		self.__log2_abs_max: float = math.log2(abs_max)
		self.__log2_diff: float = self.__log2_abs_max - self.__log2_abs_min

	@property
	def abs_min(self):
		return self.__abs_min

	@property
	def abs_max(self):
		return self.__abs_max

	def __repr__(self) -> str:
		"""
		The official string representation of an object

		Returns
		-------
		str
			The name of the class with min and max
		"""
		return __class__.__name__ + "({}, {})".format(self.__abs_min, self.__abs_max)

	def __call__(self, value: typing.Any, clip: bool | None = None) -> float | npt.NDArray[np.double]:
		"""
		Mapping the given values within [`-abs_max`, `abs_max`] to [0, 1]

		Values outside may map outside of [0, 1] based on `clip` parameter

		Parameters
		----------
		value : typing.Any
			Any value that could be regarded as `np.double` or `np.ndarray` of `np.double`
		clip : bool | None, optional
			Determines the behavior for mapping values outside the range, overwrite object behavior, by default None

		Returns
		-------
		float | npt.NDArray[np.double]
			Mapping result corresponding to value
		"""
		def process_single_value(value: float, clip: bool) -> float:
			"""
			To map a single value.

			`value` > `abs_min` will be mapped to (0.5, `inf`) or (0.5, 1] if clipped

			`value` < `-abs_min` will be mapped to (`-inf`, 0.5) or [0, 0.5) if clipped

			`-abs_min` <= `value` <= `abs_min` will be mapped to 0.5

			Parameters
			----------
			value : float
				A value to be mapped
			clip : bool
				Determines the behavior for mapping values outside the range

			Returns
			-------
			float
				Mapping result to (`-inf`, `inf`) or [0, 1] if clipped
			"""
			if abs(value) <= self.__abs_min:
				return 0.5
			sgn_half: float = 0.5 if value > 0 else -0.5
			if clip and abs(value) >= self.__abs_max:
				return sgn_half + 0.5
			else:
				return (math.log2(abs(value)) - self.__log2_abs_min) / self.__log2_diff * sgn_half + 0.5

		if clip is None:
			clip = self.clip
		result, is_scalar = super().process_value(value)
		if is_scalar == 1:
			return process_single_value(result.data.ravel()[0], clip)
		else:
			return np.ma.array(np.vectorize(process_single_value, otypes=[float])(result.data, clip), mask=result.mask)

	def inverse(self, value: typing.Any) -> float | npt.NDArray[np.double]:
		"""
		To map a value generally in [0, 1] back to [`-abs_max`, `abs_max`]

		Parameters
		----------
		value : typing.Any
			Any value that could be regarded as `np.double` or `np.ndarray` of `np.double`

		Returns
		-------
		float | npt.NDArray[np.double]
			Mapping back result corresponding to value
		"""
		def process_single_value(value: float) -> float:
			"""
			Maps a single value back

			`value` == 0.5 will be mapped back to 0

			`value` > 0.5 will be mapped back to positive values

			`value` < 0.5 will be mapped back to negative values

			Parameters
			----------
			value : float
				A value, generally in [0, 1]

			Returns
			-------
			float
				Generally in [`-abs_max`, `-abs_min`] U {0} U [`abs_min`, `abs_max`]
			"""
			if value == 0.5:
				return 0.0
			return (-1.0 if value < 0.5 else 1.0) * math.exp2(abs(value - 0.5) * 2 * self.__log2_diff + self.__log2_abs_min)

		result, is_scalar = super().process_value(value)
		if is_scalar == 1:
			return process_single_value(result.data.ravel()[0])
		else:
			return np.ma.array(np.vectorize(process_single_value, otypes=[float])(result.data), mask=result.mask)


def get_centered(abs_max: float, cmap: str | matplotlib.colors.Colormap) -> tuple[matplotlib.colors.CenteredNorm, npt.NDArray[np.double], npt.NDArray[np.double]]:
	"""
	To get the `CenteredNorm` from given input, and its corresponding levels and colorbar ticks

	Parameters
	----------
	abs_max : float
		The range of data values
	cmap : str | matplotlib.colors.Colormap
		The colormap, mainly for the RGB quantization levels

	Returns
	-------
	tuple[matplotlib.colors.CenteredNorm, npt.NDArray[np.double], npt.NDArray[np.double]]
		Norm, levels, and ticks
	"""
	abs_max = abs(abs_max)
	if isinstance(cmap, str):
		cmap = matplotlib.colormaps[cmap]
	abs_max_10_level: float = math.pow(10, math.floor(math.log10(abs_max)))
	centered_norm: matplotlib.colors.CenteredNorm = matplotlib.colors.CenteredNorm(0.0, math.ceil(abs_max / abs_max_10_level) * abs_max_10_level, True)
	centered_levels: npt.NDArray[np.double] = np.linspace(-centered_norm.halfrange, centered_norm.halfrange, cmap.N // 8 * 8 + 1, True)
	return centered_norm, centered_levels, centered_levels[::cmap.N // 8]


def get_posneg_log(abs_min: float, abs_max: float, cmap: str | matplotlib.colors.Colormap) -> tuple[PNLogNorm, npt.NDArray[np.double], npt.NDArray[np.double]]:
	"""
	To get the `PNLogNorm` from given input, and its corresponding levels and colorbar ticks

	Parameters
	----------
	abs_min : float
		The absolute value minimum of the given dataset
	abs_max : float
		The absolute value maximum of the given dataset
	cmap : str | matplotlib.colors.Colormap
		The colormap, mainly for the RGB quantization levels

	Returns
	-------
	tuple[PNLogNorm, npt.NDArray[np.double], npt.NDArray[np.double]]
		Norm, levels, and ticks
	"""
	abs_min = abs(abs_min)
	abs_max = abs(abs_max)
	if abs_max == 0:
		abs_max = sys.float_info.min * 2
	if abs_min == 0:
		abs_min = sys.float_info.min
	abs_min = max(abs_min, abs_max * sys.float_info.epsilon)
	lb_log10: int = int(math.floor(math.log10(abs_min)))
	ub_log10: int = int(math.ceil(math.log10(abs_max)))
	lb_log10 += (ub_log10 - lb_log10) % 4
	if isinstance(cmap, str):
		cmap = matplotlib.colormaps[cmap]
	num_ticks: int = cmap.N // 8 * 8 + 3
	pn_log_norm: matplotlib.colors.Normalize = PNLogNorm(math.pow(10, lb_log10), math.pow(10, ub_log10), True)
	pn_log_levels: npt.NDArray[np.double] = np.zeros(num_ticks)
	pn_log_levels[cmap.N // 8 * 4 + 2:] = np.exp2(np.linspace(math.log2(pn_log_norm.abs_min), math.log2(pn_log_norm.abs_max), cmap.N // 8 * 4 + 1, True))
	pn_log_levels[cmap.N // 8 * 4::-1] = -pn_log_levels[cmap.N // 8 * 4 + 2:]
	return pn_log_norm, pn_log_levels, np.concatenate([pn_log_levels[0:cmap.N // 8 * 4:cmap.N // 8], pn_log_levels[cmap.N // 8 * 4 + 1:cmap.N // 8 * 4 + 2], pn_log_levels[cmap.N // 8 * 5 + 2::cmap.N // 8]])


class DensityMatrixDrawer:
	"""
	To draw the density matrix.
	`max_outputs`, `init_dist`, `x_grids` and `p_grids` should be used together.
	`r0` and `dm_data` should be used together.
	`grid_data` is used independently.

	This could only be used if the model is 1D.

	Parameters
	----------
	output_interval : float
		Interval between outputs, in unit of a.u.
	max_outputs : int | None, optional
		Estimation of maximum outputs, by default None
	init_dist : pes.InitialDistribution | None, optional
		Initial density matrix, by default None
	x_grids : npt.NDArray[np.double] | None, optional
		The grids for position, by default None
	p_grids : npt.NDArray[np.double] | None, optional
		The grids for momentum, by default None
	r0 : npt.NDArray[np.double] | None, optional
		Initial positions and momenta, by default None
	dm_data : npt.NDArray[np.double], shape of (N_TICKS, NUM_ELM, ngrids, N_GRIDS) | None, optional
		The partial Wigner-transformed density matrix predicted by GPR, by default None
	grid_data : npt.NDArray[np.double], shape of (N_TICKS, NUM_ELM, ngrids, N_GRIDS) | None, optional
		The partial Wigner-transformed density matrix from grid solution, by default Non
	draw_scattered : bool, optional
		Whether to scatter sample points or not, by default False
	draw_rescaled : bool, optional
		Whether to draw rescaled density or not, by default False

	Attributes
	----------
	__CMAP : typing.Literal["seismic"]
		The colormaps for contourf
	__NTICKS : int
		The number of ticks to draw
	__COLORBAR_NTICKS : typing.Literal[21]
		The number of ticks displayed on colorbar
	FILENAME_PREFIX : typing.Literal["dm"]
		The start of the name of files to save (animation, frame pictures, and tarfile)
	__PICNAME_NO_DIGITS : typing.Literal["dm_{{:0{}}}.png"]
		The template for pictures

	Methods
	-------
	__get_divisors(fig, ax, array_shape)
		To get the omitting factors in plotting
	"""
	__CMAP: typing.Literal["seismic"] = "seismic"
	FILENAME_PREFIX: typing.Literal["dm"] = "dm"
	__PICNAME_NO_DIGITS: typing.Literal["dm_{{:0{}}}.png"] = FILENAME_PREFIX + "_{{:0{}}}" + FIGURE_EXTENSION
	__slots__: tuple = ("__output_interval", "dm_data", "__grid_data", "__draw_scattered", "__draw_rescaled", "__x_grids", "__p_grids", "picname", "__xv", "__pv", "__CENTER_NORM", "__CENTER_LEVEL", "__LOG_NORM", "__LOG_LEVEL", "__fig", "__axs", "__title", "__super_title", "__points", "__belongings", "__row_divisor", "__col_divisor")

	@staticmethod
	def __get_divisors(
		fig: matplotlib.figure.Figure,
		ax: matplotlib.axes.Axes,
		array_shape: tuple[int, ...]
	) -> tuple[int, int]:
		"""
		To get the omitting factors in plotting

		Parameters
		----------
		fig : matplotlib.figure.Figure
			The figure to plot
		ax : matplotlib.axes.Axes
			The axes to draw
		array_shape : tuple[int, ...]
			The shape of array to draw on the axes

		Returns
		-------
		tuple[int, int]
			The divisor of row and column of the array
		"""
		def get_single_divisor(dpi: float, size: int) -> int:
			"""
			To get the divisor of row or column

			Parameters
			----------
			n : int
				An integer

			Returns
			-------
			set[int]
				All the factors of the parameter
			"""
			if dpi > size:
				return 1
			else:
				all_factors: set[int] = set(functools.reduce(list.__add__, ([i, (size - 1) // i] for i in range(1, int(np.sqrt(size - 1)) + 1) if (size - 1) % i == 0)))
				factors_below_lim: set[int] = {i for i in all_factors if i < size / dpi}
				return max(factors_below_lim)

		figsize_in_dpi: npt.NDArray[np.double] = fig.get_size_inches() * fig.dpi
		ax_bbox: matplotlib.transforms.Bbox = ax.get_position()
		return get_single_divisor(figsize_in_dpi[0] * ax_bbox.width, array_shape[0]), get_single_divisor(figsize_in_dpi[1] * ax_bbox.height, array_shape[1])

	def __init__(
		self,
		output_interval: float,
		max_outputs: int | None = None,
		init_dist: pes.InitialDistribution | None = None,
		x_grids: npt.NDArray[np.double] | None = None,
		p_grids: npt.NDArray[np.double] | None = None,
		r0: npt.NDArray[np.double] | None = None,
		dm_data: npt.NDArray[np.double] | None = None,
		grid_data: npt.NDArray[np.double] | None = None,
		draw_scattered: bool = False,
		draw_rescaled: bool = False
	):
		# directly save from parameters
		self.__output_interval: float = output_interval
		self.dm_data: npt.NDArray[np.double] | None = dm_data
		self.__grid_data: npt.NDArray[np.double] | None = grid_data # grid_data.shape[0] may not be dm_data.shape[0]
		self.__draw_scattered: bool = draw_scattered
		self.__draw_rescaled: bool = draw_rescaled
		# data, grids, and filename
		self.__x_grids: npt.NDArray[np.double]
		self.__p_grids: npt.NDArray[np.double]
		num_digits_of_ticks: int
		if max_outputs is not None and init_dist is not None and x_grids is not None and p_grids is not None:
			# parameter from main
			if self.__grid_data is not None:
				num_digits_of_ticks = int(np.ceil(np.log10(self.__grid_data.shape[0])))
			else:
				num_digits_of_ticks = int(np.ceil(np.log10(max_outputs + 1)))
			self.__x_grids = x_grids
			self.__p_grids = p_grids
		else:
			# read from files
			assert r0 is not None and self.dm_data is not None and r0.shape == (2,)
			num_digits_of_ticks = int(np.ceil(np.log10(self.dm_data.shape[0])))
			self.__x_grids, self.__p_grids = get_grids(r0, self.dm_data.shape[-1])
		self.picname: str = __class__.__PICNAME_NO_DIGITS.format(num_digits_of_ticks)
		self.__xv: npt.NDArray[np.double]
		self.__pv: npt.NDArray[np.double]
		self.__xv, self.__pv = np.meshgrid(self.__x_grids, self.__p_grids)
		# levels, norms and ticks
		LOG_MAX: float
		LOG_MIN: float
		CENTER_MAX: float
		if self.__draw_rescaled:
			LOG_MAX = 1.0
			CENTER_MAX = LOG_MAX
		elif init_dist is not None:
			LOG_MAX = 1.5 * np.max(np.abs(init_dist.factors))
			CENTER_MAX = LOG_MAX
		else:
			assert self.dm_data is not None
			LOG_MAX = np.max(np.abs(self.dm_data))
			CENTER_MAX = 1.1 * LOG_MAX
		if self.dm_data is not None:
			LOG_MIN = np.min(np.abs(self.dm_data[self.dm_data != 0])) / np.max(np.abs(self.dm_data))
		else:
			LOG_MIN = LOG_MAX * sys.float_info.epsilon
		self.__CENTER_NORM: matplotlib.colors.CenteredNorm
		self.__CENTER_LEVEL: npt.NDArray[np.double]
		CENTER_TICKS: npt.NDArray[np.double]
		self.__CENTER_NORM, self.__CENTER_LEVEL, CENTER_TICKS = get_centered(CENTER_MAX, __class__.__CMAP)
		self.__LOG_NORM: PNLogNorm
		self.__LOG_LEVEL: npt.NDArray[np.double]
		LOG_TICKS: npt.NDArray[np.double]
		self.__LOG_NORM, self.__LOG_LEVEL, LOG_TICKS = get_posneg_log(LOG_MIN, LOG_MAX, __class__.__CMAP)
		# figure and axes, titles, and sample points (if available)
		self.__title: list[str] = ["Rescaled " if self.__draw_rescaled else ""]
		self.__super_title: str = self.__title[0] + "Partial Wigner-Transformed Density Matrix"
		self.__title.insert(0, self.__title[0] + "Log Scale ")
		self.__points: npt.NDArray[np.double] | None = None
		self.__belongings: npt. NDArray[np.int_] | None = None
		if draw_scattered:
			if self.dm_data is not None:
				# read from file; otherwise pass from main
				self.__points = np.loadtxt(POINTS_FILENAME + DATA_EXTENSION)
				self.__points = self.__points.reshape(self.__points.shape[0] // pes.PHASEDIM, pes.PHASEDIM, self.__points.shape[1]) # N_TICKS * PHASEDIM * NUM_PTS
				self.__belongings = np.loadtxt(BELONGING_FILENAME + DATA_EXTENSION).astype(np.int_) # N_TICKS * NUM_PTS
			self.__title.append(self.__title[0] + "Scattered ")
		if self.__grid_data is not None:
			self.__title.append(self.__title[0] + "Exact ")
			self.__title.append(self.__title[0] + "Difference of ")
		nrows: int = len(self.__title)
		self.__fig: matplotlib.figure.Figure = plt.figure(figsize=(FIGSIZE[0] * pes.NUM_ELM, FIGSIZE[1] * nrows))
		axs = self.__fig.subplots(nrows, pes.NUM_ELM, squeeze=False)
		assert isinstance(axs, np.ndarray)
		self.__axs: np.ndarray[collections.abc.Sequence[collections.abc.Sequence[matplotlib.axes.Axes]], np.dtype[np.object_]] = axs
		for iRow in range(nrows):
			for iElement in range(pes.NUM_ELM):
				ax: matplotlib.axes.Axes = self.__axs[iRow, iElement]
				ax.set_xlabel("x")
				ax.set_ylabel("p")
				ax.set_title(self.__title[iRow] + utility.get_RI_label(iElement))
				ax.contourf(
					self.__xv,
					self.__pv,
					np.zeros_like(self.__xv),
					self.__LOG_LEVEL if iRow == 0 else self.__CENTER_LEVEL,
					cmap=__class__.__CMAP,
					norm=self.__LOG_NORM if iRow == 0 else self.__CENTER_NORM
				)
		self.__fig.colorbar(
			matplotlib.cm.ScalarMappable(self.__LOG_NORM, __class__.__CMAP),
			ax=self.__axs[0].tolist(),
			ticks=LOG_TICKS,
			format="%+.1e"
		)
		self.__fig.colorbar(
			matplotlib.cm.ScalarMappable(self.__CENTER_NORM, __class__.__CMAP),
			ax=self.__axs[1:].ravel().tolist(),
			ticks=CENTER_TICKS
		)
		self.__fig.suptitle(self.__super_title)
		self.__row_divisor: int
		self.__col_divisor: int
		self.__row_divisor, self.__col_divisor = __class__.__get_divisors(self.__fig, self.__axs[0, 0], (self.__x_grids.size, self.__p_grids.size))
		self.__xv = self.__xv[::self.__row_divisor, ::self.__col_divisor]
		self.__pv = self.__pv[::self.__row_divisor, ::self.__col_divisor]

	def __call__(
		self,
		frame_index: int,
		data: npt.NDArray[np.double] | None = None,
		points: list[npt.NDArray[np.double]] | None = None,
		num_points: int | npt.NDArray[np.int_] | None = None,
		scale: npt.NDArray[np.double] | None = None
	) -> None:
		"""
		To draw a frame and save the picture

		Parameters
		----------
		frame_index : int
			The index of the frame.
			Product with `self.__output_interval` gives the duration since beginning
		data : npt.NDArray[np.double], shape of (NUM_ELM, N_GRIDS, N_GRIDS) | None, optional
			The data of the frame, by default None (and self.dm_data will be used)
		points : list[npt.NDArray[np.double]] | None, len of NUM_TRIG, each of shape (NUM_PTS, PHASEDIM), optional
			The points to scatter. Only used if `self.__draw_scattered` is True.
			By default None (`self.__points` and `self.__belongings` will be used instead.)
		num_points : int | npt.NDArray[np.int_], shape of (NUM_TRIG,) | None, optional
			The number of points located at the front of all points that is used as the subset,
			by default None (and `self.__points` and `self.__belongings` will be used instead.)
		scale : npt.NDArray[np.double], shape of (NUM_ELM,)
			The rescale factor, by default None (and `get_rescale_factor` will be used instead)
		"""
		def draw_an_axs(
			ax: matplotlib.axes.Axes,
			data: npt.NDArray[np.double],
			use_logscale: bool = False,
			title: str | None = None,
			central_points: npt.NDArray[np.double] | None = None,
			extra_points: npt.NDArray[np.double] | None = None
		) -> None:
			"""
			To draw the contourf of an Axes and add title if applicable

			Parameters
			----------
			ax : matplotlib.axes.Axes
				The Axes on which to draw
			data : npt.NDArray[np.double]
				The data for the contourf
			use_logscale : bool, optional
				Whether to plot with log norm or not, by default False
			title : str | None, optional
				The title of the Axe, by default None
			central_points : npt.NDArray[np.double], shape of (NUM_PES, NUM_PTS) | None, optional
				The central points to scatter on the element, by default None
			extra_points : npt.NDArray[np.double], shape of (NUM_PES, NUM_PTS * EXTRA_RATIO) | None, optional
				The extra points to scatter on the element, by default None
			"""
			ax.contourf(
				self.__xv,
				self.__pv,
				data[::self.__col_divisor, ::self.__row_divisor].T,
				self.__LOG_LEVEL if use_logscale else self.__CENTER_LEVEL,
				cmap=__class__.__CMAP,
				norm=self.__LOG_NORM if use_logscale else self.__CENTER_NORM
			)
			if title is not None:
				ax.set_title(title)
			# scatter the central points on top
			if extra_points is not None:
				ax.scatter(
					np.clip(extra_points[0], self.__x_grids[0], self.__x_grids[-1]),
					np.clip(extra_points[1], self.__p_grids[0], self.__p_grids[-1]),
					0.1,
					"green"
				)
			if central_points is not None:
				ax.scatter(
					np.clip(central_points[0], self.__x_grids[0], self.__x_grids[-1]),
					np.clip(central_points[1], self.__p_grids[0], self.__p_grids[-1]),
					0.5,
					"black"
				)

		# calculate rescale factors
		rescale_factors: npt.NDArray[np.double]
		if self.__draw_rescaled:
			if scale is not None:
				rescale_factors = scale
			else:
				assert self.dm_data is not None
				rescale_factors = get_rescale_factor(self.dm_data[frame_index])
		else:
			rescale_factors = np.ones(pes.NUM_ELM)
		# get data
		pred_data: npt.NDArray[np.double]
		if data is not None:
			pred_data = data
		else:
			assert self.dm_data is not None
			pred_data = self.dm_data[frame_index]
		for iElement in range(pes.NUM_ELM): # contourfs for predicted data
			draw_an_axs(
				self.__axs[0, iElement],
				pred_data[iElement].T * rescale_factors[iElement],
				True,
				self.__title[0] + utility.get_RI_label(iElement) + ("\nRescaled Factor = {:.6e}".format(rescale_factors[iElement]) if self.__draw_rescaled else "")
			)
			draw_an_axs(
				self.__axs[1, iElement],
				pred_data[iElement].T * rescale_factors[iElement],
				False,
				self.__title[1] + utility.get_RI_label(iElement) + ("\nRescaled Factor = {:.6e}".format(rescale_factors[iElement]) if self.__draw_rescaled else "")
			)
		row_index: int = 2
		if self.__draw_scattered: # scatter points, no title change
			for iElement in range(pes.NUM_ELM):
				if isinstance(num_points, int): # all elements have same number of points
					num_points = np.full(pes.NUM_TRIG, num_points, np.int_)
				if num_points is not None and points is not None: # from main
					TrilIndex: int = pes.flatten_tril_index[iElement // pes.NUM_PES, iElement % pes.NUM_PES]
					draw_an_axs(
						self.__axs[row_index, iElement],
						pred_data[iElement].T * rescale_factors[iElement],
						central_points=points[TrilIndex][:num_points[TrilIndex]].T,
						extra_points=points[TrilIndex][num_points[TrilIndex]:].T,
					)
				else: # from file
					assert self.__points is not None and self.__belongings is not None
					TrilElementIndex: int = iElement if iElement // pes.NUM_PES >= iElement % pes.NUM_PES else iElement % pes.NUM_PES * pes.NUM_PES + iElement // pes.NUM_PES
					if frame_index < self.__points.shape[0] or frame_index < self.__belongings.shape[0]:
						draw_an_axs(
							self.__axs[row_index, iElement],
							pred_data[iElement].T * rescale_factors[iElement],
							central_points=self.__points[frame_index][:, self.__belongings[frame_index] == TrilElementIndex],
							extra_points=self.__points[frame_index][:, self.__belongings[frame_index] == TrilElementIndex + pes.NUM_ELM],
						) # mixing of basic and advanced slicing leads to error
					else: # have points, but unavailable due to output
						draw_an_axs(self.__axs[row_index, iElement], pred_data[iElement].T * rescale_factors[iElement])
			row_index += 1
		if self.__grid_data is not None:
			if frame_index < self.__grid_data.shape[0]:
				for iElement in range(pes.NUM_ELM): # contourfs for grid data, no title change
					draw_an_axs(self.__axs[row_index, iElement], self.__grid_data[frame_index, iElement].T * rescale_factors[iElement])
				row_index += 1
				for iElement in range(pes.NUM_ELM): # pred - grid
					ax: matplotlib.axes.Axes = self.__axs[row_index, iElement]
					diff: npt.NDArray[np.double] = pred_data[iElement] - self.__grid_data[frame_index, iElement]
					max_idx: int = int(np.argmax(np.abs(diff).reshape(-1)))
					ax.scatter(self.__x_grids[max_idx // self.__x_grids.size], self.__p_grids[max_idx % self.__x_grids.size], 15, "black")
					draw_an_axs(
						ax,
						diff.T * rescale_factors[iElement],
						title="{}\n{}Max Abs diff = {:.6e}".format(
							self.__title[row_index] + utility.get_RI_label(iElement),
							"Rescaled " if self.__draw_rescaled else "",
							np.max(np.abs(diff))
						)
					)
				row_index += 1
			else:
				# unable to compare, set invisible
				for iElement in range(pes.NUM_ELM):
					self.__axs[row_index, iElement].set_visible(False)
				row_index += 1
				for iElement in range(pes.NUM_ELM):
					self.__axs[row_index, iElement].set_visible(False)
				row_index += 1
		self.__fig.suptitle(self.__super_title + "\n" + TIME_TEMPLATE.format(frame_index * self.__output_interval))
		self.__fig.savefig(self.picname.format(frame_index))


class WavefunctionPlotter:
	"""
	To plot wavefunctions.
	`max_outputs`, `init_dist`, `x_grids` and `p_grids` should be used together.
	`r0` and `wfn_data` should be used with together.

	Parameters
	----------
	output_interval : float
		Interval between outputs, in unit of a.u.
	max_outputs : int | None, optional
		Estimation of maximum outputs, by default None
	init_dist : pes.InitialDistribution | None, optional
		Initial density matrix, by default None
	grids_each_dim : list[npt.NDArray[np.double]] | None, optional
		The grids for positions and momenta, by default None
	r0 : npt.NDArray[np.double] | None, optional
		Initial positions and momenta, by default None
	marginal_data : npt.NDArray[np.double], shape of (N_TICKS, PHASEDIM, NUM_PES, N_GRIDS) | None, optional
		The marginal distribution in configuration space and momentum space, by default None
	draw_rescaled : bool, optional
		Whether to draw rescaled density or not, by default False

	Attributes
	----------
	__WFN_COLORS : list[matplotlib.typing.ColorType]
		Colors of wavefunction plots of each potential energy surface
	__LABEL_TEMPLATE : typing.Literal["Surface {}"]
		Template for labels of each line
	FILENAME_PREFIX : typing.Literal["wfn"]
		The start of the name of files to save (animation, frame pictures, and tarfile)
	__PICNAME_NO_DIGITS : typing.Literal["wfn_{{:0{}}}.png"]
		The template for pictures
	"""
	__WFN_COLORS: list[matplotlib.typing.ColorType]
	if pes.NUM_PES < 10:
		__WFN_COLORS = matplotlib.color_sequences["Set1"][:pes.NUM_PES]
	else:
		cmap: npt.NDArray[np.double] = matplotlib.colormaps["gist_rainbow"](np.linspace(0.0, 1.0, pes.NUM_PES, True))
		__WFN_COLORS = list(zip(cmap[:, 0], cmap[:, 1], cmap[:, 2]))
	__LABEL_TEMPLATE: typing.Literal["Surface {}"] = "Surface {}"
	FILENAME_PREFIX: typing.Literal["wfn"] = "wfn"
	__PICNAME_NO_DIGITS: typing.Literal["wfn_{{:0{}}}.png"] = FILENAME_PREFIX + "_{{:0{}}}.png"
	__slots__: tuple = ("__output_interval", "wfn_sqnm", "__draw_rescaled", "__title", "__grids", "picname", "__fig", "__axs")

	def __init__(
		self,
		output_interval: float,
		max_outputs: int | None = None,
		init_dist: pes.InitialDistribution | None = None,
		grids_each_dim: list[npt.NDArray[np.double]] | None = None,
		r0: npt.NDArray[np.double] | None = None,
		marginal_data: npt.NDArray[np.double] | None = None,
		draw_rescaled: bool = False
	):
		# directly save from parameters
		self.__output_interval: float = output_interval
		self.wfn_sqnm: npt.NDArray[np.double] | None = marginal_data
		self.__draw_rescaled: bool = draw_rescaled
		self.__title: str = ("Rescaled " if self.__draw_rescaled else "") + "Marginal on each surfaces"
		# data, grids, and filename
		num_digits_of_ticks: int
		self.__grids: list[npt.NDArray[np.double]]
		if max_outputs is not None and grids_each_dim is not None:
			# parameter from main
			num_digits_of_ticks: int = int(np.ceil(np.log10(max_outputs + 1)))
			self.__grids = grids_each_dim
		else:
			# read from files
			assert r0 is not None and self.wfn_sqnm is not None
			num_digits_of_ticks: int = int(np.ceil(np.log10(self.wfn_sqnm.shape[0])))
			self.__grids = get_grids(r0, self.wfn_sqnm.shape[-1])
		self.picname: str = __class__.__PICNAME_NO_DIGITS.format(num_digits_of_ticks)
		# figure and axes
		self.__fig: matplotlib.figure.Figure = plt.figure(figsize=(FIGSIZE[0] * 2, FIGSIZE[1] * pes.DIM))
		axs = self.__fig.subplots(pes.DIM, 2, squeeze=False) # guaranteen it is matrix in case pes.DIM == 1
		assert isinstance(axs, np.ndarray)
		self.__axs: np.ndarray[collections.abc.Sequence[matplotlib.axes.Axes], np.dtype[np.object_]] = axs
		self.__fig.suptitle(self.__title)
		labels: list[str] = [utility.dimension_name(iDim) for iDim in range(pes.PHASEDIM)]
		titles: list[str] = ["Position", "Momentum"]
		for iDim in range(pes.PHASEDIM):
			ax: matplotlib.axes.Axes = self.__axs[iDim % pes.DIM, iDim // pes.DIM]
			ax.set_xlabel("{} / a.u.".format(labels[iDim]))
			ax.set_ylabel("Population")
			ax.set_xbound(self.__grids[iDim][0], self.__grids[iDim][-1])
			max_y: float
			if self.__draw_rescaled:
				max_y = 1.0
			elif init_dist is not None:
				max_y = 1.5 * np.max(np.abs(init_dist.weight_phase.diagonal() / np.sqrt(2.0 * np.pi) / init_dist.sigma_r0[iDim]))
			else:
				assert self.wfn_sqnm is not None
				max_y = 1.1 * float(np.max(self.wfn_sqnm[:, iDim]))
			ax.set_ybound(0.0, 1.5 * max_y)
			plot_title: str = "Marginal on "
			if pes.DIM != 1:
				plot_title += "Dimension {} of ".format(iDim % pes.DIM)
			plot_title += titles[iDim // pes.DIM]
			ax.set_title(plot_title)

	def __call__(
		self,
		frame_index: int,
		marginal: npt.NDArray[np.double] | None = None
	) -> None:
		"""
		To draw a frame and save as png

		Parameters
		----------
		frame_index : int
			The index of the frame.
			Product with `self.__output_interval` gives the duration since beginning
		marginal : npt.NDArray[np.double], shape of (PHASEDIM, NUM_PES, N_GRIDS) | None, optional
			The squared norm of wavefunction, by default None (and self.wfn_sqnm will be used)

		Returns
		-------
		collections.abc.Iterable[matplotlib.artist.Artist]
			figure and axe
		"""
		data_to_plot: npt.NDArray[np.double]
		if marginal is not None:
			data_to_plot = marginal
		else:
			assert self.wfn_sqnm is not None
			data_to_plot = self.wfn_sqnm[frame_index]
		for iDim in range(pes.PHASEDIM):
			ax: matplotlib.axes.Axes = self.__axs[iDim % pes.DIM, iDim // pes.DIM]
			if ax.legend_:
				ax.legend_.remove()
			for line in ax.lines:
				line.remove()
			# draw newdata
			for iPES in range(pes.NUM_PES):
				rescale_factor: float
				label: str = __class__.__LABEL_TEMPLATE.format(iPES + 1)
				if self.__draw_rescaled:
					rescale_factor = 1.0 / np.max(np.abs(data_to_plot[iDim, iPES]))
					label += "\n" + RESCALE_TEMPLATE.format(rescale_factor)
				else:
					rescale_factor = 1.0
				ax.plot(
					self.__grids[iDim],
					data_to_plot[iDim, iPES] * rescale_factor,
					color=__class__.__WFN_COLORS[iPES],
					lw=2,
					label=label
				)
			ax.legend()
		self.__fig.suptitle(self.__title + "\n" + TIME_TEMPLATE.format(frame_index * self.__output_interval))
		self.__fig.savefig(self.picname.format(frame_index))


def main() -> None:
	_, r0, _, _, _, _, output_interval, _, _ = read_input()
	parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Choice of drawing")
	parser.add_argument("--animation", "--ani", "-a", default=0, type=int, help="Draw animation; if so, provide the interval between frames in ms. Draw frame by frame and compress into tarfile will always be done.")
	parser.add_argument("--scattered", "-s", action="store_true", help="Scatter points on the distribution")
	parser.add_argument("--rescaled", "-r", action="store_true", help="Rescale all elements to same (maximum = 1.0)")
	parser.add_argument("--wavefunction-rescaled", "--wr", action="store_true", help="Rescale wavefunction as well (default to be the same as --rescaled)")
	parser.add_argument("--density-matrix", "--dm", "-d", default=ALL_GRIDS_FILENAME + DATA_EXTENSION, type=str, help="File containing partial Wigner-transformed density matrix from Gaussian process regression")
	parser.add_argument("--marginal", "-m", default=MARGINAL_FILENAME + DATA_EXTENSION, type=str, help="File containing marginal distribution from Gaussian process regression")
	parser.add_argument("--grid-solution", "--grid", "-g", default="", type=str, help="File containing partial Wigner-transformed density matrix from grid solution")
	result: dict[str, typing.Any] = vars(parser.parse_args())
	try:
		grid_solution: npt.NDArray[np.double] | None = None
		if result["grid_solution"] != "":
			grid_solution = read_data(result["grid_solution"])
		dm_drawer: DensityMatrixDrawer = DensityMatrixDrawer(
			output_interval,
			r0=r0,
			dm_data=read_data(result["density_matrix"]),
			grid_data=grid_solution,
			draw_scattered=result["scattered"],
			draw_rescaled=result["rescaled"]
		)
		assert dm_drawer.dm_data is not None
		dm_total_ticks: int = dm_drawer.dm_data.shape[0]
		for iframe in range(dm_total_ticks):
			dm_drawer(iframe)
		if result["animation"] != 0: # draw an animation
			with PIL.Image.open(dm_drawer.picname.format(0)) as first_img:
				first_img.save(
					DensityMatrixDrawer.FILENAME_PREFIX + ".gif",
					save_all=True,
					append_images=[PIL.Image.open(dm_drawer.picname.format(iframe)) for iframe in range(1, dm_total_ticks)],
					loop=0,
					duration=result["animation"]
				)
		# draw frame by frame and combine into a tarfile
		if os.path.isdir(dm_drawer.FILENAME_PREFIX):
			os.rename(dm_drawer.FILENAME_PREFIX, dm_drawer.FILENAME_PREFIX + "_" + str(datetime.datetime.now()).replace(" ", "_"))
		subprocess.run(["mkdir", dm_drawer.FILENAME_PREFIX]) # make directory
		subprocess.run(["mv"] + [dm_drawer.picname.format(iTick) for iTick in range(dm_total_ticks)] + [dm_drawer.FILENAME_PREFIX])
		with tarfile.open(dm_drawer.FILENAME_PREFIX + TAR_EXTENSION, "w:gz") as dm_tf:
			dm_tf.add(dm_drawer.FILENAME_PREFIX)
	finally:
		print("Finish dm.", datetime.datetime.now(), flush=True)
	try:
		marginal_data: npt.NDArray[np.double] = np.loadtxt(result["marginal"])
		marginal_data = marginal_data.reshape(
			marginal_data.shape[0] // (pes.PHASEDIM * pes.NUM_ELM),
			pes.PHASEDIM,
			pes.NUM_PES,
			pes.NUM_PES,
			marginal_data.shape[1]
		) # (N_TICKS, PHASEDIM, NUM_PES, NUM_PES, N_GRIDS)
		wfn_plotter: WavefunctionPlotter = WavefunctionPlotter(
			output_interval,
			r0=r0,
			marginal_data=marginal_data.diagonal(axis1=2, axis2=3).swapaxes(-1, -2), # .diagonal will move axis to end
			draw_rescaled=result["wavefunction_rescaled"]
		)
		assert wfn_plotter.wfn_sqnm is not None
		wfn_total_ticks: int = wfn_plotter.wfn_sqnm.shape[0]
		for iframe in range(wfn_total_ticks):
			wfn_plotter(iframe)
		if result["animation"] != 0: # draw an animation
			with PIL.Image.open(wfn_plotter.picname.format(0)) as first_img:
				first_img.save(
					WavefunctionPlotter.FILENAME_PREFIX + ".gif",
					save_all=True,
					append_images=[PIL.Image.open(wfn_plotter.picname.format(iframe)) for iframe in range(1, wfn_total_ticks)],
					loop=0,
					duration=result["animation"]
				)
		# draw frame by frame and combine into a tarfile
		if os.path.isdir(wfn_plotter.FILENAME_PREFIX):
			os.rename(wfn_plotter.FILENAME_PREFIX, wfn_plotter.FILENAME_PREFIX + "_" + str(datetime.datetime.now()).replace(" ", "_"))
		subprocess.run(["mkdir", wfn_plotter.FILENAME_PREFIX]) # make directory
		subprocess.run(["mv"] + [wfn_plotter.picname.format(iTick) for iTick in range(wfn_total_ticks)] + [wfn_plotter.FILENAME_PREFIX])
		with tarfile.open(wfn_plotter.FILENAME_PREFIX + TAR_EXTENSION, "w:gz") as wfn_tf:
			wfn_tf.add(wfn_plotter.FILENAME_PREFIX)
	finally:
		print("Finish wfn.", datetime.datetime.now(), flush=True)


if __name__ == "__main__":
	main()
