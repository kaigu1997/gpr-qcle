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


def read_input() -> tuple[npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.double], float, float]:
	"""
	To read input

	Returns
	-------
	tuple[npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.double], float, float]
		mass, center, standard deviation, output interval and time step
	"""
	with open('input', 'r', encoding='UTF-8') as in_f:
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
	data: npt.NDArray[np.double] = np.loadtxt('pwtdm.txt', np.double)
	ticks: int = data.shape[0] // (pes.NUM_ELM * 2)
	length: int = int(np.round(np.sqrt(data.shape[1])))
	data = data.reshape((ticks, pes.NUM_ELM, 2, length, length))
	result: npt.NDArray[np.cdouble] = np.empty((ticks, pes.NUM_ELM, length, length), np.cdouble)
	result.real = data[:, :, 0]
	result.imag = data[:, :, 1]
	return result


def get_scale(data: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
	"""
	To calculate the scale factor from the given data.

	The scale would be 1 / max() in general, or 0 if all are 0

	Parameters
	----------
	data : npt.NDArray[np.double], of shape (NUM_ELM, ...)
		The data

	Returns
	-------
	npt.NDArray[np.double]
		Scale of each element
	"""
	with np.errstate(divide='ignore'):
		return np.where(np.any(data.reshape(pes.NUM_ELM, -1) != 0, -1), 1.0 / np.max(np.abs(data.reshape(pes.NUM_ELM, -1)), -1), 0)


def real_part(arr: npt.NDArray[np.cdouble]) -> npt.NDArray[np.double]:
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


def main() -> None:
	"""
	The main routine

	Raises
	------
	NotImplementedError
		In case an unimplemented branch is reached
	"""
	main.FIGSIZE = (6.4, 4.8)
	main.LIMIT = 1.0
	main.CMAP = 'seismic'
	main.LOCATOR = matplotlib.ticker.MaxNLocator(nbins=21)
	main.NORM = matplotlib.colors.CenteredNorm(0.0, main.LIMIT, True)
	main.NUM_PTS = 256
	main.NUM_XTR_RATIO = 50
	title: list[str] = ['Scattered ', '', 'Exact ', 'Difference of ']
	NUM_PLOTS: int = len(title)
	mass: npt.NDArray[np.double]
	r0: npt.NDArray[np.double]
	sigma_r0: npt.NDArray[np.double]
	output_interval: float
	dt: float
	mass, r0, sigma_r0, output_interval, dt = read_input()
	output_steps: int = int(round(output_interval / dt))
	print("Mass = {}\nInitial center = {}\nInitial deviation = {}\nTime step = {}\nSteps between output = {}\n".format(mass, r0, sigma_r0, dt, output_steps))
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
	rescaled_errors: npt.NDArray[np.double] = np.empty((total_ticks, pes.NUM_ELM))
	original_errors: npt.NDArray[np.double] = np.empty((total_ticks, pes.NUM_ELM))
	evolving_errors: npt.NDArray[np.double] = np.empty((total_ticks, pes.NUM_ELM))
	# drawer
	fig: matplotlib.figure.Figure
	axs: np.ndarray[collections.abc.Sequence[collections.abc.Sequence[matplotlib.axes.Axes]], np.dtype[np.object_]]
	fig, axs = plt.subplots(NUM_PLOTS, pes.NUM_PES ** 2, figsize=(main.FIGSIZE[0] * (pes.NUM_PES ** 2), main.FIGSIZE[1] * NUM_PLOTS))
	# sampling. Initial sample from gaussian directly
	all_pts: npt.NDArray[np.double] = np.tile(sample.normal_sample(main.NUM_PTS, r0, sigma_r0)[np.newaxis], (pes.NUM_ELM, 1 + main.NUM_XTR_RATIO, 1))
	sample.sample_extra_points(main.NUM_PTS, all_pts)
	sample.sample_central_points(main.NUM_PTS, all_pts)
	sample.sample_extra_points(main.NUM_PTS, all_pts)
	all_density: npt.NDArray[np.cdouble] = np.zeros(all_pts.shape[:-1], np.cdouble)
	for iPES in range(pes.NUM_PES):
		for jPES in range(iPES + 1):
			ElementIndex: int = iPES * pes.NUM_PES + jPES
			interpolator_re: scipy.interpolate.RegularGridInterpolator = scipy.interpolate.RegularGridInterpolator((x_grids, p_grids), data[0, ElementIndex].real, 'cubic', False, 0.0)
			all_density.real[ElementIndex] = interpolator_re(all_pts[ElementIndex])
			if iPES != jPES:
				interpolator_im: scipy.interpolate.RegularGridInterpolator = scipy.interpolate.RegularGridInterpolator((x_grids, p_grids), data[0, ElementIndex].imag, 'cubic', False, 0.0)
				all_density.imag[ElementIndex] = interpolator_im(all_pts[ElementIndex])
				all_density[jPES * pes.NUM_PES + iPES] = np.conj(all_density[ElementIndex])
	# average evaluators
	mca: expectation.MonteCarloAverage = expectation.MonteCarloAverage(1_000_000)
	aia: expectation.AnalyticalAverager = expectation.AnalyticalAverager(predictors)
	# files for output
	pts_f: io.TextIOWrapper
	den_f: io.TextIOWrapper
	all_f: io.TextIOWrapper
	err_f: io.TextIOWrapper
	scl_f: io.TextIOWrapper
	prm_f: io.TextIOWrapper
	lss_f: io.TextIOWrapper
	with open('points.txt', 'w', encoding='UTF-8') as pts_f,\
		open('density.txt', 'w', encoding='UTF-8') as den_f,\
		open('all_grids.txt', 'w', encoding='UTF-8') as all_f,\
		open('error.txt', 'w', encoding='UTF-8') as err_f,\
		open('scale.txt', 'w', encoding='UTF-8') as scl_f,\
		open('parameters.txt', 'w', encoding='UTF-8') as prm_f,\
		open('loss.txt', 'w', encoding='UTF-8') as lss_f,\
		open('ave.txt', 'w', encoding='UTF-8') as ave_f:
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
			scale: npt.NDArray[np.double] = get_scale(real_part(data[iTick]))
			print("Tick {}, scales = {}".format(iTick, scale))
			# interpolate the data
			y_all: npt.NDArray[np.cdouble] = np.empty(all_density.shape, np.cdouble)
			for iPES in range(pes.NUM_PES):
				for jPES in range(iPES + 1):
					ElementIndex: int = iPES * pes.NUM_PES + jPES
					interpolator_re: scipy.interpolate.RegularGridInterpolator = scipy.interpolate.RegularGridInterpolator((x_grids, p_grids), data[iTick, ElementIndex].real, 'cubic', False, 0.0)
					y_all.real[ElementIndex] = interpolator_re(all_pts[ElementIndex])
					if iPES != jPES:
						interpolator_im: scipy.interpolate.RegularGridInterpolator = scipy.interpolate.RegularGridInterpolator((x_grids, p_grids), data[iTick, ElementIndex].imag, 'cubic', False, 0.0)
						y_all.imag[ElementIndex] = interpolator_im(all_pts[ElementIndex])
						y_all[jPES * pes.NUM_PES + iPES] = np.conj(y_all[ElementIndex])
					else:
						y_all.imag = 0
			evolving_errors[iTick] = np.sum((real_part(y_all) - real_part(all_density)) ** 2, -1)
			# fit
			predictors.update(all_pts, all_density, main.NUM_PTS, scale)
			predictors.train()
			# predict
			pred: npt.NDArray[np.cdouble] = np.empty((pes.NUM_ELM, n_grids, n_grids), np.cdouble)
			for iPES in range(pes.NUM_PES):
				for jPES in range(iPES + 1):
					ElementIndex: int = iPES * pes.NUM_PES + jPES
					pred[ElementIndex] = predictors.predict(grids, ElementIndex).reshape(n_grids, n_grids)
					if iPES != jPES:
						pred[jPES * pes.NUM_PES + iPES] = np.conj(pred[ElementIndex])
			real_pred: npt.NDArray[np.double] = real_part(pred)
			real_data: npt.NDArray[np.double] = real_part(data[iTick])
			diff: npt.NDArray[np.double] = real_pred - real_data
			original_errors[iTick] = np.sum(diff ** 2, (-2, -1))
			rescaled_errors[iTick] = original_errors[iTick] * scale ** 2

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
					ax: matplotlib.axes.Axes = axs[iPlot, iElement]
					ax.contourf(
						xv,
						pv,
						plot_region(ct_data[iElement].T),
						cmap=main.CMAP,
						norm=main.NORM,
						vmin=-main.LIMIT,
						vmax=main.LIMIT,
						locator=main.LOCATOR)
					if iPlot == 0:
						ax.scatter(
							np.clip(all_pts[iElement, main.NUM_PTS:, 0], x_grids[0], x_grids[-1]),
							np.clip(all_pts[iElement, main.NUM_PTS:, 1], p_grids[0], p_grids[-1]),
							0.1,
							'green')
						ax.scatter(
							np.clip(all_pts[iElement, :main.NUM_PTS, 0], x_grids[0], x_grids[-1]),
							np.clip(all_pts[iElement, :main.NUM_PTS, 1], p_grids[0], p_grids[-1]),
							0.5,
							'black')
						ax.set_title('Rescaled {}\nRescale Factor = {}'.format(title[0] + gp.get_RI_label(iElement // pes.NUM_PES, iElement % pes.NUM_PES), scale[iElement]))
					elif iPlot == 1:
						ax.set_title('Rescaled Error = {}\nRescaled Predict Error = {}'.format(rescaled_errors[iTick, iElement], evolving_errors[iTick, iElement]))
					elif iPlot == 3:
						max_idx: np.intp = np.argmax(np.abs(diff[iElement]).reshape(-1))
						ax.scatter(x_grids[max_idx // n_grids], p_grids[max_idx % n_grids], 15, 'black')
						ax.set_title('Rescaled {}\nRescaled Max Abs diff = {}'.format(title[iPlot] + gp.get_RI_label(iElement // pes.NUM_PES, iElement % pes.NUM_PES), np.abs(diff[iElement])[max_idx // n_grids, max_idx % n_grids]))
			fig.suptitle('Tick = {}\nTotal Rescaled Error = {}\nTotal Rescaled Predict Error = {}'.format(iTick, np.sum(rescaled_errors[iTick]), np.sum(evolving_errors[iTick])))
			fig.savefig('Tick = {:03}.png'.format(iTick))
			# print to file
			np.savetxt(pts_f, all_pts.reshape(pes.NUM_ELM, main.NUM_PTS * (main.NUM_XTR_RATIO + 1) * pes.PHASEDIM), footer='\n', comments='')
			np.savetxt(den_f, real_part(all_density))
			np.savetxt(den_f, real_part(y_all))
			# all_density = y_all
			for iPES in range(pes.NUM_PES):
				for jPES in range(pes.NUM_PES):
					if iPES == jPES:
						np.savetxt(den_f, predictors.predict(all_pts[iPES * pes.NUM_PES + jPES], iPES * pes.NUM_PES + jPES).real.reshape(1, -1))
					elif iPES < jPES:
						np.savetxt(den_f, predictors.predict(all_pts[jPES * pes.NUM_PES + iPES], jPES * pes.NUM_PES + iPES).real.reshape(1, -1))
					else:
						np.savetxt(den_f, predictors.predict(all_pts[iPES * pes.NUM_PES + jPES], iPES * pes.NUM_PES + jPES).imag.reshape(1, -1))
			print('\n', end='\n', file=den_f)
			np.savetxt(all_f, real_part(pred).reshape(pes.NUM_ELM, n_grids ** 2), footer='\n', comments='')
			np.savetxt(err_f, (original_errors[iTick], rescaled_errors[iTick], evolving_errors[iTick]), footer='\n', comments='')
			# calculate averages
			print(iTick * output_interval, end=" ", file=ave_f)
			mca.update_pts(all_pts[:, :main.NUM_PTS], predictors.predict)
			aver: expectation.Averager
			for aver in [mca, aia]:
				print(*aver.population(), *aver.coordinates(), aver.potential(), aver.kinetic(mass), *aver.purity().reshape(-1), end=" ", file=ave_f)
			print("", file=ave_f, flush=True)

		def init() -> typing.Iterable[matplotlib.artist.Artist | typing.Iterable[matplotlib.artist.Artist]]:
			"""
			Initializer of the animation

			Returns
			-------
			typing.Iterable[matplotlib.artist.Artist | typing.Iterable[matplotlib.artist.Artist]]
				figure and axes
			"""
			for iPES in range(pes.NUM_PES):
				for jPES in range(pes.NUM_PES):
					ElementIndex: int = iPES * pes.NUM_PES + jPES
					for iPlot in range(NUM_PLOTS):
						ax: matplotlib.axes.Axes = axs[iPlot, ElementIndex]
						ax.set_xlabel('x')
						ax.set_ylabel('p')
						ax.set_title('Rescaled ' + title[iPlot] + gp.get_RI_label(iPES, jPES))
			fig.colorbar(matplotlib.cm.ScalarMappable(cmap=main.CMAP, norm=main.NORM), ax=axs.ravel().tolist())
			train_pred_draw(0)
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
			# evolve
			for _ in range(output_steps):
				predictors.print(prm_f)
				print('\n', file=prm_f)
				evolve.evolve(all_pts, all_density, mass, dt, predictors.predict)
				scale: npt.NDArray[np.double] = get_scale(real_part(all_density))
				np.savetxt(scl_f, scale)
				print('\n', file=scl_f)
				predictors.update(all_pts, all_density, main.NUM_PTS, scale)
				for i in range(pes.NUM_ELM):
					print(predictors[i].error().item(), file=lss_f)
				print('\n', file=lss_f)
			train_pred_draw(iTick)
			DoResample: bool = False #iTick % 50 == 0
			if DoResample:
				sample.sample_central_points(main.NUM_PTS, all_pts, all_density)
			sample.sample_extra_points(main.NUM_PTS, all_pts, predictors)
			pred_starting_idx: int = main.NUM_PTS
			for iPES in range(pes.NUM_PES):
				for jPES in range(iPES + 1):
					ElementIndex: int = iPES * pes.NUM_PES + jPES
					all_density[ElementIndex, pred_starting_idx:] = predictors.predict(all_pts[ElementIndex, pred_starting_idx:], ElementIndex)
					if iPES != jPES:
						all_density[jPES * pes.NUM_PES + iPES, pred_starting_idx:] = np.conj(all_density[ElementIndex, pred_starting_idx:])
			predictors.update(all_pts, all_density, main.NUM_PTS, get_scale(real_part(all_density)))
			return fig, axs

		# draw animation
		matplotlib.animation.FuncAnimation(fig, draw, range(1, total_ticks), init).save('gpr.gif', 'imagemagick')

	# then plots
	# error
	plt.close(fig)
	fig, axs = plt.subplots(1, 3, figsize=(main.FIGSIZE[0] * 3, main.FIGSIZE[1]))
	error_titles: list[str] = ['Original', 'Rescaled', 'Evolving']
	for iPlot in range(3):
		err: npt.NDArray[np.double]
		match iPlot:
			case 0:
				err = original_errors
			case 1:
				err = rescaled_errors
			case 2:
				err = evolving_errors
			case _:
				raise NotImplementedError("Unexpected Plot for Errors")
		ax: matplotlib.axes.Axes = axs[iPlot]
		for iPES in range(pes.NUM_PES):
			for jPES in range(pes.NUM_PES):
				ElementIndex: int = iPES * pes.NUM_PES + jPES
				ax.semilogy(np.arange(total_ticks), err[:, ElementIndex], label=gp.get_RI_label(iPES, jPES))
		ax.semilogy(np.arange(total_ticks), np.sum(err, axis=1), label='Total Error')
		ax.set_xlabel('Ticks')
		ax.set_ylabel('Error')
		ax.legend()
		ax.set_title(error_titles[iPlot] + ' Error')
	fig.savefig('error.png')
	# param
	fig, axs = plt.subplots(1, pes.PHASEDIM, figsize=(main.FIGSIZE[0] * pes.PHASEDIM, main.FIGSIZE[1]))
	param: npt.NDArray[np.double] = np.loadtxt('parameters.txt').reshape(-1, pes.NUM_ELM, pes.PHASEDIM)
	param_total_ticks = param.shape[0] # This should be (total_ticks - 1) * output_steps
	for iDim in range(pes.PHASEDIM):
		ax: matplotlib.axes.Axes = axs[iDim]
		for iPES in range(pes.NUM_PES):
			for jPES in range(pes.NUM_PES):
				ElementIndex: int = iPES * pes.NUM_PES + jPES
				ax.plot(np.arange(param_total_ticks) * dt, param[:, ElementIndex, iDim], label=gp.get_RI_label(iPES, jPES))
			ax.set_xlabel('Time')
			ax.set_title(('x' if iDim < pes.DIM else 'p') + str(iDim % pes.DIM))
			ax.legend()
	fig.savefig('param.png')
	plt.close(fig)
	# tar figures
	with tarfile.open('ticks.tar.gz', 'w:gz') as tf:
		for iTick in range(total_ticks):
			name: str = 'Tick = {:03}.png'.format(iTick)
			tf.add(name)
			os.remove(name)


if __name__ == "__main__":
	main()
