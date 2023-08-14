#!/usr/bin/env python3
import evolve
import gp
import pes
import sample

import collections.abc
import io
import matplotlib.animation
import matplotlib.axes
import matplotlib.colors
import matplotlib.cm
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import numpy.typing as npt
import os
import scipy.interpolate
import tarfile
import typing

FIGSIZE: tuple[float, float] = (6.4, 4.8)
LIMIT: float = 1.0
CMAP: str = 'seismic'
LOCATOR: matplotlib.ticker.MaxNLocator = matplotlib.ticker.MaxNLocator(nbins=21)
NORM: matplotlib.colors.CenteredNorm = matplotlib.colors.CenteredNorm(0.0, LIMIT, True)
MASS: npt.NDArray[np.double] = np.full(pes.DIM, 2000.0, np.double)
DT: float = 0.1


def read_input() -> npt.NDArray[np.cdouble]:
	data: npt.NDArray[np.double] = np.loadtxt('pwtdm.txt')
	ticks: int = data.shape[0] // (pes.NUM_ELM * 2)
	length: int = int(np.round(np.sqrt(data.shape[1])))
	data = data.reshape((ticks, pes.NUM_ELM, 2, length, length))
	result: npt.NDArray[np.cdouble] = np.empty((ticks, pes.NUM_ELM, length, length), np.cdouble)
	result.real = data[:, :, 0]
	result.imag = data[:, :, 1]
	return result


def get_RI_label(row: int, col: int) -> str:
	if row == col:
		return r'$\rho_{%d,%d}$' % (row, col)
	elif row < col:
		return r'$\Re\rho_{%d,%d}$' % (col, row)
	else:
		return r'$\Im\rho_{%d,%d}$' % (row, col)


def get_scale(data: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
	with np.errstate(divide='ignore'):
		return np.where(np.any(data.reshape(pes.NUM_ELM, -1) != 0, -1), 1.0 / np.max(np.abs(data.reshape(pes.NUM_ELM, -1)), -1), 0)


def real_part(arr: npt.NDArray[np.cdouble]) -> npt.NDArray[np.double]:
	return arr.view(np.double).reshape(arr.shape + (2,))[np.arange(pes.NUM_ELM), ..., np.tril(np.ones((pes.NUM_PES, pes.NUM_PES), np.int_), -1).reshape(pes.NUM_ELM)]


def plot_region(arr: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
	return arr[::4, ::4]


if __name__ == "__main__":
	NUM_PTS: typing.Literal[200] = 200
	NUM_XTR_RATIO: typing.Literal[50] = 50
	OUTPUT_INTERVAL: float = 5.0
	title: list[str] = ['Scattered ', '', 'Exact ', 'Difference of ']
	NUM_PLOTS: int = len(title)
	r0: npt.NDArray[np.double] = np.array([-8.0, 14.112], np.double)
	sigma_r0: npt.NDArray[np.double] = np.array([10.0 / r0[1], r0[1] / 20.0], np.double)
	# read input
	data: npt.NDArray[np.cdouble] = read_input()
	total_ticks: int = data.shape[0]
	n_grids: int = data.shape[-1]
	x_grids: npt.NDArray[np.double] = np.linspace(-16.0, 16.0, n_grids, True, dtype=np.double)
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
	fig, axs = plt.subplots(NUM_PLOTS, pes.NUM_PES ** 2, figsize=(FIGSIZE[0] * (pes.NUM_PES ** 2), FIGSIZE[1] * NUM_PLOTS))
	# sampling. Initial sample from gaussian directly
	all_pts: npt.NDArray[np.double] = np.tile(sample.init_sample(NUM_PTS, r0, sigma_r0), (1 + NUM_XTR_RATIO, 1))
	sample.sample_extra_points(all_pts, NUM_PTS)
	all_density: npt.NDArray[np.cdouble] = np.zeros(all_pts.shape[:-1], np.cdouble)
	all_density.real[0] = np.exp(-np.sum(np.square((all_pts[0] - r0) / sigma_r0), axis=-1) / 2.0) / (2.0 * np.pi * np.prod(sigma_r0))
	element_populated: npt.NDArray[np.bool_] = np.zeros(pes.NUM_ELM, np.bool_)
	element_populated[0] = True
	# files for output
	pts_f: io.TextIOWrapper = open('points.txt', 'w')
	den_f: io.TextIOWrapper = open('density.txt', 'w')
	all_f: io.TextIOWrapper = open('all_grids.txt', 'w')
	err_f: io.TextIOWrapper = open('error.txt', 'w')
	scl_f: io.TextIOWrapper = open('scale.txt', 'w')
	prm_f: io.TextIOWrapper = open('parameters.txt', 'w')

	def train_pred_draw(iTick: int):
		global predictors, rescaled_errors, original_errors, evolving_errors, fig, axs, all_pts, all_density, pts_f, den_f, all_f, err_f
		# get scale
		scale: npt.NDArray[np.double] = get_scale(real_part(data[iTick]))
		print(iTick, scale)
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
		evolving_errors[iTick] = np.sum((real_part(y_all) - real_part(all_density)) ** 2, -1)
		# fit
		predictors.update(all_pts, y_all, NUM_PTS, scale)
		predictors.train()
		# predict
		pred: npt.NDArray[np.cdouble] = np.empty((pes.NUM_ELM, n_grids, n_grids), np.cdouble)
		for iPES in range(pes.NUM_PES):
			for jPES in range(iPES + 1):
				ElementIndex: int = iPES * pes.NUM_PES + jPES
				pred[ElementIndex] = predictors.predict(grids, ElementIndex).reshape(n_grids, n_grids)
				if iPES != jPES:
					pred[jPES * pes.NUM_PES + iPES] = np.conj(pred[ElementIndex])
		original_errors[iTick] = np.sum(real_part(pred - data[iTick]) ** 2, (-2, -1))
		rescaled_errors[iTick] = np.sum((real_part(pred - data[iTick]) * scale[:, np.newaxis, np.newaxis]) ** 2, (-2, -1))

		for iPlot in range(NUM_PLOTS):
			def contour_data() -> npt.NDArray[np.cdouble]:
				match iPlot:
					case 0 | 1:
						return pred
					case 2:
						return data[iTick]
					case 3:
						return pred - data[iTick]
					case _:
						raise NotImplemented("Unexpected Plot for Contour")
			ct_data: npt.NDArray[np.double] = real_part(contour_data()) * scale[:, np.newaxis, np.newaxis]
			for iElement in range(pes.NUM_ELM):
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
						np.clip(all_pts[iElement, NUM_PTS:, 0], x_grids[0], x_grids[-1]),
						np.clip(all_pts[iElement, NUM_PTS:, 1], p_grids[0], p_grids[-1]),
						0.1,
						'green')
					ax.scatter(
						np.clip(all_pts[iElement, :NUM_PTS, 0], x_grids[0], x_grids[-1]),
						np.clip(all_pts[iElement, :NUM_PTS, 1], p_grids[0], p_grids[-1]),
						0.5,
						'black')
					ax.set_title('Rescaled {}\nRescale Factor = {}'.format(title[0] + get_RI_label(iElement // pes.NUM_PES, iElement % pes.NUM_PES), scale[iElement]))
				elif iPlot == 1:
					ax.set_title('Rescaled Error = {}\nRescaled Predict Error = {}'.format(rescaled_errors[iTick, iElement], evolving_errors[iTick, iElement]))
		fig.suptitle('Tick = {}\nTotal Rescaled Error = {}\nTotal Rescaled Predict Error = {}'.format(iTick, np.sum(rescaled_errors[iTick]), np.sum(evolving_errors[iTick])))
		fig.savefig('Tick = {:03}.png'.format(iTick))
		# print to file
		np.savetxt(pts_f, all_pts.reshape(pes.NUM_ELM, NUM_PTS * (NUM_XTR_RATIO + 1) * pes.PHASEDIM), footer='\n', comments='')
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
		np.savetxt(scl_f, scale)
		print('\n', file=scl_f)
		predictors.print(prm_f)
		print('\n', file=prm_f)

	def init():
		global all_pts, all_density, fig, axs
		for iPES in range(pes.NUM_PES):
			for jPES in range(pes.NUM_PES):
				ElementIndex: int = iPES * pes.NUM_PES + jPES
				for iPlot in range(NUM_PLOTS):
					ax: matplotlib.axes.Axes = axs[iPlot, ElementIndex]
					ax.set_xlabel('x')
					ax.set_ylabel('p')
					ax.set_title('Rescaled ' + title[iPlot] + get_RI_label(iPES, jPES))
		fig.colorbar(matplotlib.cm.ScalarMappable(cmap=CMAP, norm=NORM), ax=axs.ravel().tolist())
		train_pred_draw(0)
		return fig, axs,

	def draw(iTick: int):
		global all_pts, all_density, element_populated, fig, axs, scl_f, prm_f
		# evolve
		for i in range(int(round(OUTPUT_INTERVAL / DT))):
			evolve.evolve(all_pts, all_density, MASS, DT, predictors.predict)
			scale: npt.NDArray[np.double] = get_scale(real_part(all_density))
			predictors.update(all_pts, all_density, NUM_PTS, scale)
			element_currently_populated: npt.NDArray[np.bool_] = np.any(all_density != 0.0, -1)
			if np.any(element_populated != element_currently_populated):
				predictors.train()
				element_populated[...] = element_currently_populated
			np.savetxt(scl_f, scale)
			print('\n', file=scl_f)
			predictors.print(prm_f)
			print('\n', file=prm_f)
		train_pred_draw(iTick)
		if iTick % 1 == 0:
			sample.resample(all_pts, all_density, NUM_PTS, predictors)
		sample.sample_extra_points(all_pts, NUM_PTS, predictors)
		for iPES in range(pes.NUM_PES):
			for jPES in range(iPES + 1):
				ElementIndex: int = iPES * pes.NUM_PES + jPES
				all_density[ElementIndex, NUM_PTS:] = predictors.predict(all_pts[ElementIndex, NUM_PTS:], ElementIndex)
				if iPES != jPES:
					all_density[jPES * pes.NUM_PES + iPES, NUM_PTS:] = np.conj(all_density[ElementIndex, NUM_PTS:])
		return fig, axs,

	# draw animation
	matplotlib.animation.FuncAnimation(fig, draw, range(1, total_ticks), init).save('gpr.gif', 'imagemagick')
	pts_f.close()
	den_f.close()
	all_f.close()
	err_f.close()
	scl_f.close()
	prm_f.close()
	# then plot errors

	def get_error(iPlot: int) -> npt.NDArray[np.double]:
		match iPlot:
			case 0:
				return original_errors
			case 1:
				return rescaled_errors
			case 2:
				return evolving_errors
			case _:
				raise ValueError("Unexpected Plot for Errors")

	plt.close(fig)
	fig, axs = plt.subplots(1, 3, figsize=(FIGSIZE[0] * 3, FIGSIZE[1]))
	error_titles: list[str] = ['Original', 'Rescaled', 'Evolving']
	for iPlot in range(3):
		ax: matplotlib.axes.Axes = axs[iPlot]
		for iPES in range(pes.NUM_PES):
			for jPES in range(pes.NUM_PES):
				ElementIndex: int = iPES * pes.NUM_PES + jPES
				ax.semilogy(np.arange(total_ticks), get_error(iPlot)[:, ElementIndex], label=get_RI_label(iPES, jPES))
		ax.semilogy(np.arange(total_ticks), np.sum(get_error(iPlot), axis=1), label='Total Error')
		ax.set_xlabel('Ticks')
		ax.set_ylabel('Error')
		ax.legend()
		ax.set_title(error_titles[iPlot] + ' Error')
	fig.savefig('error.png')
	plt.close(fig)
	with tarfile.open('ticks.tar.gz', 'w:gz') as tf:
		for iTick in range(total_ticks):
			name: str = 'Tick = {:03}.png'.format(iTick)
			tf.add(name)
			os.remove(name)
