#!/usr/bin/env python3

"""
main
====

The main module.
"""

import collections.abc
import io
import os
import tarfile
import typing

import matplotlib.animation
import matplotlib.artist
import matplotlib.axes
import matplotlib.colors
import matplotlib.cm
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import numpy.typing as npt
import scipy.interpolate

import evolve
import expectation
import gp
import pes
import sample

FIGSIZE: tuple[float, float] = (6.4, 4.8)
LIMIT: float = 1.0
CMAP: typing.Literal["seismic"] = "seismic"
LOCATOR: matplotlib.ticker.Locator = matplotlib.ticker.MaxNLocator(nbins=21)
NORM: matplotlib.colors.Normalize = matplotlib.colors.CenteredNorm(0.0, LIMIT, True)
NUM_PTS: typing.Literal[256] = 256
NUM_XTR_RATIO: typing.Literal[50] = 50
NUM_MC_PTS: typing.Literal[1_000_000] = 1_000_000


def read_input() -> tuple[npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.double], float, float]:
	"""
	To read input

	Returns
	-------
	tuple[npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.double], float, float]
		mass, center, standard deviation, output interval and time step
	"""
	with open("input", 'r', encoding="UTF-8") as in_f:
		lines: list[str] = in_f.readlines()
		mass: npt.NDArray[np.double] = np.array(lines[3][:-1].split(' '), dtype=np.double)
		assert pes.DIM % mass.size == 0
		mass = np.tile(mass, pes.DIM // mass.size)
		x0: npt.NDArray[np.double] = np.array(lines[5][:-1].split(' '), dtype=np.double)
		assert pes.DIM % x0.size == 0
		x0 = np.tile(x0, pes.DIM // x0.size)
		p0: npt.NDArray[np.double] = np.array(lines[7][:-1].split(' '), dtype=np.double)
		assert pes.DIM % p0.size == 0
		p0 = np.tile(p0, pes.DIM // p0.size)
		sigma_p0: npt.NDArray[np.double] = np.array(lines[9][:-1].split(' '), dtype=np.double)
		assert pes.DIM % sigma_p0.size == 0
		sigma_p0 = np.tile(sigma_p0, pes.DIM // sigma_p0.size)
		output_time: float = float(lines[13])
		dt: float = float(lines[15])
		return mass, np.concatenate((x0, p0)), np.concatenate((pes.HBAR / 2.0 / sigma_p0, sigma_p0)), output_time, dt


def read_data() -> npt.NDArray[np.cdouble]:
	"""
	To read input data

	Returns
	-------
	npt.NDArray[np.cdouble]
		Input data
	"""
	data: npt.NDArray[np.double] = np.loadtxt("pwtdm.txt", np.double)
	ticks: int = data.shape[0] // (pes.NUM_ELM * 2)
	length: int = int(np.round(np.sqrt(data.shape[1])))
	data = data.reshape((ticks, pes.NUM_ELM, 2, length, length))
	result: npt.NDArray[np.cdouble] = np.empty((ticks, pes.NUM_ELM, length, length), np.cdouble)
	result.real = data[:, :, 0]
	result.imag = data[:, :, 1]
	return result


def get_scale(data: npt.NDArray[np.double] | list[npt.NDArray[np.cdouble]]) -> npt.NDArray[np.double]:
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


def get_independent_parts(arr: npt.NDArray[np.cdouble]) -> npt.NDArray[np.double]:
	"""
	To extract the independent components of density supervector

	It takes the real part of upper triangular (including diagonal), and imaginary part of strictly-lower triangular

	Parameters
	----------
	arr : npt.NDArray[np.cdouble], shape of (NUM_ELM, ...)
		The density of all elements

	Returns
	-------
	npt.NDArray[np.double]
		Independent components, a real array
	"""
	return arr.view(np.double).reshape(arr.shape + (2,))[np.arange(pes.NUM_ELM), ..., np.tril(np.ones((pes.NUM_PES, pes.NUM_PES), np.int_), -1).reshape(pes.NUM_ELM)]


def plot_region(arr: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
	"""
	To extract the region to plot.

	To draw the contour more efficiently, only 1/4 of the grids are used.

	Parameters
	----------
	arr : npt.NDArray[np.double], shape of (N, N)
		A 2D array for contour plotting

	Returns
	-------
	npt.NDArray[np.double]
		_description_
	"""
	return arr[::4, ::4]


def format_array(arr_name: str, arr: npt.NDArray | float | int | complex) -> str:
	"""
	To print the flatten array as well as raw number

	Parameters
	----------
	arr_name : str
		The name of the array or variable
	arr : npt.NDArray | float | int
		The array or raw number

	Returns
	-------
	str
		_description_
	"""
	if isinstance(arr, np.ndarray):
		return (arr_name + " =" + " {}" * arr.size).format(*arr.ravel())
	elif isinstance(arr, complex):
		return arr_name + " = {} + {}i".format(arr.real, arr.imag)
	else:
		return arr_name + " = " + str(arr)


def dimension_name(DimIndex: int) -> str:
	"""
	Name corresponding to the dimension in phase space

	Parameters
	----------
	DimIndex : int
		Index of the dimension, range in [0, PHASEDIM)

	Returns
	-------
	str
		The name of the dimension
	"""
	assert 0 <= DimIndex < pes.PHASEDIM
	if pes.DIM == 1:
		if DimIndex == 0:
			return "x"
		else:
			return "p"
	else:
		if DimIndex < pes.DIM:
			return r"$x_{}$".format(DimIndex)
		else:
			return r"$p_{}$".format(DimIndex - pes.DIM)


def main() -> None:
	"""
	The main routine

	Raises
	------
	NotImplementedError
		In case an unimplemented branch is reached
	"""
	title: list[str] = ["Scattered ", "", "Exact ", "Difference of "]
	NUM_PLOTS: int = len(title)
	mass: npt.NDArray[np.double]
	r0: npt.NDArray[np.double]
	sigma_r0: npt.NDArray[np.double]
	output_interval: float
	dt: float
	mass, r0, sigma_r0, output_interval, dt = read_input()
	output_steps: int = int(round(output_interval / dt))
	print(format_array("Mass", mass), format_array("Initial center", r0), format_array("Initial deviation", sigma_r0), format_array("Time step", dt), format_array("Steps between output", output_steps), sep='\n')
	# read input
	data: npt.NDArray[np.cdouble] = read_data()
	total_ticks: int = data.shape[0]
	n_grids: int = data.shape[-1]
	x_grids: npt.NDArray[np.double] = 2.0 * np.abs(r0[0]) * np.linspace(-1.0, 1.0, n_grids, True, dtype=np.double)
	dx: np.double = (x_grids[-1] - x_grids[0]) / (n_grids - 1)
	p_grids: npt.NDArray[np.double] = r0[1] + np.pi / 2.0 / dx * np.linspace(-1.0, 1.0, n_grids, True, dtype=np.double)
	xv: npt.NDArray[np.double]
	pv: npt.NDArray[np.double]
	xv, pv = np.meshgrid(x_grids, p_grids)
	grids: npt.NDArray[np.double] = np.array((xv, pv), dtype=np.double).T.reshape(-1, pes.PHASEDIM)
	xv, pv = plot_region(xv), plot_region(pv)
	# the regressor
	predictors: gp.GPRPredictors = gp.GPRPredictors()
	# drawer
	fig: matplotlib.figure.Figure
	axs: np.ndarray[collections.abc.Sequence[collections.abc.Sequence[matplotlib.axes.Axes]], np.dtype[np.object_]]
	fig, axs = plt.subplots(NUM_PLOTS, pes.NUM_ELM, figsize=(FIGSIZE[0] * pes.NUM_ELM, FIGSIZE[1] * NUM_PLOTS))
	# sampling. Initial sample from gaussian directly
	gp_pts: list[npt.NDArray[np.double]] = [arr for arr in np.tile(sample.normal_sample(NUM_PTS, r0, sigma_r0)[np.newaxis], (pes.NUM_TRIG, 1 + NUM_XTR_RATIO, 1))]
	num_points: npt.NDArray[np.int_] = np.full(pes.NUM_TRIG, NUM_PTS)
	sample.sample_extra_points(num_points, gp_pts)
	sample.sample_central_points(num_points, gp_pts)
	sample.sample_extra_points(num_points, gp_pts)
	gp_density: list[npt.NDArray[np.cdouble]] = [np.empty(pts.shape[0], np.cdouble) for pts in gp_pts]
	for iPES in range(pes.NUM_PES):
		for jPES in range(iPES + 1):
			TrilIndex: int = pes.flatten_tril_index[iPES, jPES]
			ElementIndex: int = iPES * pes.NUM_PES + jPES
			interpolator_re: scipy.interpolate.RegularGridInterpolator = scipy.interpolate.RegularGridInterpolator((x_grids, p_grids), data[0, ElementIndex].real, "cubic", False, 0.0)
			gp_density[TrilIndex].real = interpolator_re(gp_pts[TrilIndex])
			if iPES != jPES:
				interpolator_im: scipy.interpolate.RegularGridInterpolator = scipy.interpolate.RegularGridInterpolator((x_grids, p_grids), data[0, ElementIndex].imag, "cubic", False, 0.0)
				gp_density[TrilIndex].imag = interpolator_im(gp_pts[TrilIndex])
			else:
				gp_density[TrilIndex].imag = 0
	# prepare for surface-hopping part
	sh_pts: npt.NDArray[np.double] = np.concatenate([arr[:NUM_PTS] for arr in gp_pts], 0)
	sh_belonging_idx: npt.NDArray[np.int_] = np.repeat(pes.tril_element_indices, NUM_PTS, 0)
	sh_density: npt.NDArray[np.cdouble] = np.concatenate([den[:NUM_PTS] for den in gp_density], 0)
	# average evaluators
	mca: expectation.MonteCarloAverage = expectation.MonteCarloAverage(NUM_MC_PTS)
	aia: expectation.AnalyticalAverager = expectation.AnalyticalAverager(predictors)
	# files for output
	ERR_FNAME: typing.Literal["error"] = "error"
	SCL_FNAME: typing.Literal["scale"] = "scale"
	PRM_FNAME: typing.Literal["parameters"] = "parameters"
	LSS_FNAME: typing.Literal["loss"] = "loss"
	AVE_FNAME: typing.Literal["ave"] = "ave"
	pts_f: io.TextIOWrapper
	den_f: io.TextIOWrapper
	all_f: io.TextIOWrapper
	err_f: io.TextIOWrapper
	scl_f: io.TextIOWrapper
	prm_f: io.TextIOWrapper
	lss_f: io.TextIOWrapper
	with open("points.txt", 'w', encoding="UTF-8") as pts_f,\
		open("belonging.txt", 'w', encoding="UTF-8") as bln_f,\
		open("density.txt", 'w', encoding="UTF-8") as den_f,\
		open("all_grids.txt", 'w', encoding="UTF-8") as all_f,\
		open(ERR_FNAME + ".txt", 'w', encoding="UTF-8") as err_f,\
		open(AVE_FNAME + ".txt", 'w', encoding="UTF-8") as ave_f,\
		open(SCL_FNAME + ".txt", 'w', encoding="UTF-8") as scl_f,\
		open(PRM_FNAME + ".txt", 'w', encoding="UTF-8") as prm_f,\
		open(LSS_FNAME + ".txt", 'w', encoding="UTF-8") as lss_f:
		def train_pred_draw(iTick: int) -> None:
			"""
			To train the parameter, doing prediction on all grids, and draw it

			Parameters
			----------
			iTick : int
				Current time tick, to access grid data and for plotting

			Raises
			------
			NotImplementedError
				In case an unimplemented branch is reached
			"""
			# get scale
			scale: npt.NDArray[np.double] = get_scale(gp_density)
			print("Tick {}, scales = {}".format(iTick, scale))
			# interpolate the data
			y_all: list[npt.NDArray[np.cdouble]] = [np.array([], dtype=np.cdouble) for _ in range(pes.NUM_TRIG)]
			evolving_errors: npt.NDArray[np.double] = np.empty((pes.NUM_PES, pes.NUM_PES), np.double)
			for iPES in range(pes.NUM_PES):
				for jPES in range(iPES + 1):
					TrilIndex: int = pes.flatten_tril_index[iPES, jPES]
					ElementIndex: int = iPES * pes.NUM_PES + jPES
					interpolator_re: scipy.interpolate.RegularGridInterpolator = scipy.interpolate.RegularGridInterpolator((x_grids, p_grids), data[iTick, ElementIndex].real, "cubic", False, 0.0)
					y_all[TrilIndex] = interpolator_re(gp_pts[TrilIndex]).astype(np.cdouble)
					if iPES == jPES:
						evolving_errors[iPES, jPES] = np.sum((y_all[TrilIndex].real - gp_density[TrilIndex].real) ** 2) / num_points[TrilIndex]
					else:
						interpolator_im: scipy.interpolate.RegularGridInterpolator = scipy.interpolate.RegularGridInterpolator((x_grids, p_grids), data[iTick, ElementIndex].imag, "cubic", False, 0.0)
						y_all[TrilIndex].imag = interpolator_im(gp_pts[TrilIndex])
						evolving_errors[jPES, iPES] = np.sum((y_all[TrilIndex].real - gp_density[TrilIndex].real) ** 2) / num_points[TrilIndex]
						evolving_errors[iPES, jPES] = np.sum((y_all[TrilIndex].imag - gp_density[TrilIndex].imag) ** 2) / num_points[TrilIndex]

			evolving_errors = evolving_errors.reshape(-1)
			# fit
			predictors.update(gp_pts, gp_density, num_points, scale)
			predictors.train()
			# predict
			pred: npt.NDArray[np.cdouble] = np.empty((pes.NUM_ELM, n_grids, n_grids), np.cdouble)
			for iPES in range(pes.NUM_PES):
				for jPES in range(iPES + 1):
					ElementIndex: int = iPES * pes.NUM_PES + jPES
					pred[ElementIndex] = predictors.predict(grids, ElementIndex).reshape(n_grids, n_grids)
					if iPES != jPES:
						pred[jPES * pes.NUM_PES + iPES] = np.conj(pred[ElementIndex])
			real_pred: npt.NDArray[np.double] = get_independent_parts(pred)
			real_data: npt.NDArray[np.double] = get_independent_parts(data[iTick])
			diff: npt.NDArray[np.double] = real_pred - real_data
			original_errors: npt.NDArray[np.double] = np.sum(diff ** 2, (-2, -1))
			rescaled_errors: npt.NDArray[np.double] = original_errors * scale ** 2

			for iPlot in range(NUM_PLOTS):
				contour_data: npt.NDArray[np.double]
				match iPlot:
					case 0 | 1:
						contour_data = real_pred
					case 2:
						contour_data = real_data
					case 3:
						contour_data = diff
					case _:
						raise NotImplementedError("Unexpected Plot for Contour")
				ct_data: npt.NDArray[np.double] = contour_data * scale[:, np.newaxis, np.newaxis]
				for iElement in range(pes.NUM_ELM):
					TrilIndex: int = pes.flatten_tril_index[iElement // pes.NUM_PES, iElement % pes.NUM_PES]
					ax: matplotlib.axes.Axes = axs[iPlot, iElement]
					ax.contourf(
						xv,
						pv,
						plot_region(ct_data[iElement].T),
						cmap=CMAP,
						norm=NORM,
						vmin=-LIMIT,
						vmax=LIMIT,
						locator=LOCATOR)
					if iPlot == 0:
						ax.scatter(
							np.clip(gp_pts[TrilIndex][num_points[TrilIndex]:, 0], x_grids[0], x_grids[-1]),
							np.clip(gp_pts[TrilIndex][num_points[TrilIndex]:, 1], p_grids[0], p_grids[-1]),
							0.1,
							"green"
						)
						ax.scatter(
							np.clip(gp_pts[TrilIndex][:num_points[TrilIndex], 0], x_grids[0], x_grids[-1]),
							np.clip(gp_pts[TrilIndex][:num_points[TrilIndex], 1], p_grids[0], p_grids[-1]),
							0.5,
							"black"
						)
						ax.set_title("Rescaled {}\nRescale Factor = {:.6e}".format(title[0] + gp.get_RI_label(iElement), scale[iElement]))
					elif iPlot == 1:
						ax.set_title("Rescaled Error = {:.6e}\nRescaled Predict Error = {:.6e}".format(rescaled_errors[iElement], evolving_errors[iElement]))
					elif iPlot == 3:
						max_idx: np.intp = np.argmax(np.abs(diff[iElement]).reshape(-1))
						ax.scatter(x_grids[max_idx // n_grids], p_grids[max_idx % n_grids], 15, "black")
						ax.set_title("Rescaled {}\nRescaled Max Abs diff = {:.6e}".format(title[iPlot] + gp.get_RI_label(iElement), np.abs(diff[iElement])[max_idx // n_grids, max_idx % n_grids]))
			fig.suptitle("Tick = {}\nTotal Rescaled Error = {}\nTotal Rescaled Predict Error = {}".format(iTick, np.sum(rescaled_errors), np.sum(evolving_errors)))
			fig.savefig("Tick = {:03}.png".format(iTick))
			# print to file
			np.savetxt(pts_f, np.concatenate(gp_pts).T, footer='\n', comments="")
			np.savetxt(bln_f, np.concatenate([np.full(gp_pts[i].shape[0], pes.tril_element_indices[i], np.int_) for i in range(pes.NUM_TRIG)])[np.newaxis], footer='\n', comments="")
			np.savetxt(den_f, np.concatenate(gp_density).view(np.double).reshape(-1, 2).T)
			np.savetxt(den_f, np.concatenate(y_all).view(np.double).reshape(-1, 2).T)
			np.savetxt(den_f, np.concatenate([predictors.predict(gp_pts[i], pes.tril_element_indices[i]) for i in range(pes.NUM_TRIG)]).view(np.double).reshape(-1, 2).T, footer='\n', comments="")
			print('\n', end='\n', file=den_f)
			np.savetxt(all_f, get_independent_parts(pred).reshape(pes.NUM_ELM, n_grids ** 2), footer='\n', comments="")
			np.savetxt(err_f, (original_errors, rescaled_errors, evolving_errors), footer='\n', comments="")
			# calculate averages
			print(iTick * output_interval, end=' ', file=ave_f)
			mca.update_pts(gp_pts, predictors.predict)
			aver: expectation.Averager
			for aver in [mca, aia]:
				print(*aver.population(), *aver.coordinates(), *aver.covariance()[np.tril_indices(pes.PHASEDIM)], aver.potential(), aver.kinetic(mass), *aver.purity().reshape(-1), end=' ', file=ave_f)
			print("", file=ave_f, flush=True)

		def init() -> typing.Iterable[matplotlib.artist.Artist | typing.Iterable[matplotlib.artist.Artist]]:
			"""
			Initializer of the animation

			Returns
			-------
			typing.Iterable[matplotlib.artist.Artist | typing.Iterable[matplotlib.artist.Artist]]
				figure and axes
			"""
			for iElement in range(pes.NUM_ELM):
				for iPlot in range(NUM_PLOTS):
					ax: matplotlib.axes.Axes = axs[iPlot, iElement]
					ax.set_xlabel('x')
					ax.set_ylabel('p')
					ax.set_title("Rescaled " + title[iPlot] + gp.get_RI_label(iElement))
			fig.colorbar(matplotlib.cm.ScalarMappable(cmap=CMAP, norm=NORM), ax=axs.ravel().tolist())
			train_pred_draw(0)
			predictors.print(prm_f)
			print('\n', file=prm_f)
			np.savetxt(scl_f, get_scale(gp_density))
			print('\n', file=scl_f)
			for i in range(pes.NUM_ELM):
				print(predictors[i].error().item(), file=lss_f)
			print('\n', file=lss_f)
			return fig, axs

		def draw(iTick: int) -> typing.Iterable[matplotlib.artist.Artist | typing.Iterable[matplotlib.artist.Artist]]:
			"""
			Implementation of each frame

			Parameters
			----------
			iTick : int
				Current time tick, to access grid data and for plotting

			Returns
			-------
			typing.Iterable[matplotlib.artist.Artist | typing.Iterable[matplotlib.artist.Artist]]
				figure and axes
			"""
			nonlocal gp_pts, gp_density, num_points
			# evolve
			for _ in range(output_steps):
				evolve.evolve(gp_pts, gp_density, mass, dt, predictors.predict)
				evolve.sh_evolve(sh_pts, sh_density, sh_belonging_idx, mass, dt, predictors.predict)
				scale: npt.NDArray[np.double] = get_scale(gp_density)
				predictors.update(gp_pts, gp_density, num_points, scale)
				predictors.print(prm_f)
				print('\n', file=prm_f)
				np.savetxt(scl_f, scale)
				print('\n', file=scl_f)
				for i in range(pes.NUM_ELM):
					print(predictors[i].error().item(), file=lss_f)
				print('\n', file=lss_f)
			# update and predict
			train_pred_draw(iTick)
			# exchange with SH
			num_points = np.unique(sh_belonging_idx, return_counts=True)[1]
			gp_pts = [np.tile(sh_pts[sh_belonging_idx == iTril], (1 + NUM_XTR_RATIO, 1)) for iTril in pes.tril_element_indices]
			gp_density = [np.tile(sh_density[sh_belonging_idx == iTril], 1 + NUM_XTR_RATIO) for iTril in pes.tril_element_indices]
			# sample extra points and predict them
			sample.sample_extra_points(num_points, gp_pts)
			for iPES in range(pes.NUM_PES):
				for jPES in range(iPES + 1):
					TrilIndex: int = pes.flatten_tril_index[iPES, jPES]
					gp_density[TrilIndex][num_points[TrilIndex]:] = predictors.predict(gp_pts[TrilIndex][num_points[TrilIndex]:], iPES * pes.NUM_PES + jPES)
			predictors.update(gp_pts, gp_density, num_points, get_scale(gp_density))
			predictors.train()
			predictors.print(prm_f)
			print('\n', file=prm_f)
			np.savetxt(scl_f, get_scale(gp_density))
			print('\n', file=scl_f)
			for i in range(pes.NUM_ELM):
				print(predictors[i].error().item(), file=lss_f)
			print('\n', file=lss_f)
			return fig, axs

		# draw animation
		matplotlib.animation.FuncAnimation(fig, draw, range(1, total_ticks), init).save("gpr.gif", "imagemagick")

	# then plots
	plt.close(fig)
	# error
	error_average_ticks: npt.NDArray[np.double] = np.arange(total_ticks) * output_interval
	fig, axs = plt.subplots(1, 3, figsize=(FIGSIZE[0] * 3, FIGSIZE[1]))
	error: npt.NDArray[np.double] = np.loadtxt(ERR_FNAME + ".txt").reshape(-1, 3, pes.NUM_ELM)
	error_titles: list[str] = ["Original", "Rescaled", "Evolving"]
	for iPlot in range(axs.size):
		ax: matplotlib.axes.Axes = axs[iPlot]
		for iElement in range(pes.NUM_ELM):
			ax.semilogy(error_average_ticks, error[:, iPlot, iElement], label=gp.get_RI_label(iElement))
		ax.set_xlabel("Time / a.u.")
		ax.set_ylabel("Error")
		ax.legend()
		ax.set_title(error_titles[iPlot] + " Error")
	fig.savefig(ERR_FNAME + ".png")
	plt.close(fig)
	# averages
	ave_type_titles: list[str] = ["Monte Carlo", "Analytical"]
	plot_titles = ["Population"] + [dimension_name(iDim) for iDim in range(pes.PHASEDIM)] + [("Cov(" + dimension_name(jDim) + ", " + dimension_name(iDim) + ")") if iDim != jDim else ("Var[" + dimension_name(iDim) + "]") for iDim in range(pes.PHASEDIM) for jDim in range(iDim + 1)] + ["Energy", "Purity"]
	fig, axs = plt.subplots(len(ave_type_titles), len(plot_titles), figsize=(FIGSIZE[0] * len(plot_titles), FIGSIZE[1] * len(ave_type_titles))) # population, <x> and <p>, energy, purity
	averages: npt.NDArray[np.double] = np.loadtxt(AVE_FNAME + ".txt")
	assert averages.shape[0] == total_ticks
	averages = averages[:, 1:].reshape(total_ticks, 2, (averages.shape[1] - 1) // 2)
	for iRow in range(axs.shape[0]):
		for iCol in range(axs.shape[1]):
			ax: matplotlib.axes.Axes = axs[iRow, iCol]
			y_label: str = plot_titles[iCol]
			if iCol == 0: # population
				for iPES in range(pes.NUM_PES):
					ax.plot(error_average_ticks, averages[:, iRow, iPES], label="State " + str(iPES))
				ax.plot(error_average_ticks, np.sum(averages[:, iRow, :pes.NUM_PES], -1), label="Total")
				ax.legend()
			elif iCol <= pes.PHASEDIM * (pes.PHASEDIM + 3) // 2: # <x>, <p>, and covariances
				ax.plot(error_average_ticks, averages[:, iRow, iCol - 1 + pes.NUM_PES])
				y_label += " / a.u."
				if iCol > pes.PHASEDIM:
					y_label += r"$^2$"
			elif iCol == pes.PHASEDIM * (pes.PHASEDIM + 3) // 2 + 1: # energies
				ax.plot(error_average_ticks, averages[:, iRow, pes.NUM_PES + pes.PHASEDIM * (pes.PHASEDIM + 3) // 2 + 1], label="Kinetic Energy")
				if iRow == 0: # only mc has potential and thus total energy
					ax.plot(error_average_ticks, averages[:, iRow, pes.NUM_PES + pes.PHASEDIM * (pes.PHASEDIM + 3) // 2], label="Potential Energy")
					ax.plot(error_average_ticks, np.sum(averages[:, iRow, pes.NUM_PES + pes.PHASEDIM * (pes.PHASEDIM + 3) // 2:pes.NUM_PES + pes.PHASEDIM * (pes.PHASEDIM + 3) // 2 + 2], -1), label="Total Energy")
				ax.legend()
				y_label += " / a.u."
			else: # purity
				for iElement in range(pes.NUM_ELM):
					ax.plot(error_average_ticks, averages[:, iRow, pes.NUM_PES + pes.PHASEDIM * (pes.PHASEDIM + 3) // 2 + 2 + iElement], label=gp.get_RI_label(iElement))
				ax.plot(error_average_ticks, np.sum(averages[:, iRow, pes.NUM_PES + pes.PHASEDIM * (pes.PHASEDIM + 3) // 2 + 2:], -1), label="Total")
				ax.legend()
			ax.set_xlabel("Time / a.u.")
			ax.set_ylabel(y_label)
			ax.set_title(ave_type_titles[iRow] + " Average of " + plot_titles[iCol])
	fig.savefig(AVE_FNAME + ".png")
	plt.close(fig)
	# parameters
	param_loss_scale_ticks: npt.NDArray[np.double] = np.concatenate([np.arange(i * output_steps, (i + 1) * output_steps + 1) for i in range(total_ticks - 1)]) * dt # This should be size of (total_ticks - 1) * (output_steps + 1)
	fig, axs = plt.subplots(1, pes.PHASEDIM, figsize=(FIGSIZE[0] * pes.PHASEDIM, FIGSIZE[1]))
	parameters: npt.NDArray[np.double] = np.loadtxt(PRM_FNAME + ".txt").reshape(-1, pes.NUM_ELM, pes.PHASEDIM)[:-1]
	assert param_loss_scale_ticks.size == parameters.shape[0]
	parameter_names: list[str] = [dimension_name(iDim) for iDim in range(pes.PHASEDIM)]
	for iDim in range(pes.PHASEDIM):
		ax: matplotlib.axes.Axes = axs[iDim]
		for iElement in range(pes.NUM_ELM):
			ax.plot(param_loss_scale_ticks, np.sqrt(parameters[:, iElement, iDim]), label=gp.get_RI_label(iElement))
		ax.set_xlabel("Time / a.u.")
		ax.set_ylabel("Characteristic Length / a.u.")
		ax.set_title("Characteristic Length of " + parameter_names[iDim])
		ax.legend()
	fig.savefig(PRM_FNAME + ".png")
	plt.close(fig)
	# loss and rescale factor
	loss_scale: npt.NDArray[np.double] = np.concatenate((np.loadtxt(LSS_FNAME + ".txt").reshape(-1, pes.NUM_ELM, 1), np.loadtxt(SCL_FNAME + ".txt").reshape(-1, pes.NUM_ELM, 1)), -1)[:-1]
	assert param_loss_scale_ticks.size == loss_scale.shape[0]
	loss_scale_titles: list[str] = ["Loss on Sample Points", "Rescale Factor"]
	fig, axs = plt.subplots(1, 2, figsize=(FIGSIZE[0] * 2, FIGSIZE[1]))
	for iPlot in range(axs.size):
		ax: matplotlib.axes.Axes = axs[iPlot]
		for iElement in range(pes.NUM_ELM):
				ax.semilogy(param_loss_scale_ticks, loss_scale[:, iElement, iPlot], label=gp.get_RI_label(iElement))
		ax.set_xlabel("Time / a.u.")
		ax.set_ylabel("Characteristic Length / a.u." if iPlot < pes.PHASEDIM else "")
		ax.set_title(loss_scale_titles[iPlot])
		ax.legend()
	fig.savefig(LSS_FNAME + "-" + SCL_FNAME + ".png")
	plt.close(fig)
	# tar figures
	with tarfile.open("ticks.tar.gz", "w:gz") as tf:
		for iTick in range(total_ticks):
			name: str = "Tick = {:03}.png".format(iTick)
			tf.add(name)
			os.remove(name)


if __name__ == "__main__":
	main()
