#!/usr/bin/env python3
import evolve
import gp
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
import torch
import typing

FIGSIZE: tuple[float, float] = (6.4, 4.8)
LIMIT: float = 1.0
CMAP: str = 'seismic'
LOCATOR: matplotlib.ticker.MaxNLocator = matplotlib.ticker.MaxNLocator(nbins=21)
NORM: matplotlib.colors.CenteredNorm = matplotlib.colors.CenteredNorm(0.0, LIMIT, True)


def read_input() -> npt.NDArray[np.double]:
	data: npt.NDArray[np.double] = np.loadtxt('pwtdm.txt')
	ticks: int = data.shape[0] // (evolve.NUM_ELM * 2)
	length: int = int(np.round(np.sqrt(data.shape[1])))
	data = data.reshape((ticks, evolve.NUM_ELM, 2, length, length))
	result: npt.NDArray[np.double] = np.empty((ticks, evolve.NUM_ELM, length, length))
	for iPES in range(evolve.NUM_PES):
		for jPES in range(evolve.NUM_PES):
			result[:, iPES * evolve.NUM_PES + jPES, :, :] = data[:, iPES * evolve.NUM_PES + jPES, 0 if iPES <= jPES else 1, :, :]
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
		return np.where(np.any(data != 0, (1, 2)), 1.0 / np.max(np.abs(data), (1, 2)), 0)


def plot_region(arr: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
	return arr[::4, ::4]


if __name__ == "__main__":
	NUM_PTS: typing.Literal[200] = 200
	NUM_XTR_RATIO: typing.Literal[10] = 10
	OUTPUT_INTERVAL: float = 0.1
	title: list[str] = ['Scattered ', '', 'Exact ', 'Difference of ']
	NUM_PLOTS: int = len(title)
	r0: npt.NDArray[np.double] = np.array([-1.6, 14.112], np.double)
	sigma_r0: npt.NDArray[np.double] = np.array([10.0 / r0[1], r0[1] / 20.0], np.double)
	# read input
	data: npt.NDArray[np.double] = read_input()
	total_ticks: int = data.shape[0]
	n_grids: int = data.shape[-1]
	x_grids: npt.NDArray[np.double] = np.linspace(-16.0, 16.0, n_grids, True, dtype=np.double)
	dx: np.double = (x_grids[-1] - x_grids[0]) / (n_grids - 1)
	p_grids: npt.NDArray[np.double] = r0[1] + np.pi / 2.0 / dx * np.linspace(-1.0, 1.0, n_grids, True, dtype=np.double)
	grids_min: npt.NDArray[np.double] = np.array([x_grids[0], p_grids[0]], dtype=np.double)
	grids_max: npt.NDArray[np.double] = np.array([x_grids[-1], p_grids[-1]], dtype=np.double)
	xv: npt.NDArray[np.double]
	pv: npt.NDArray[np.double]
	xv, pv = np.meshgrid(x_grids, p_grids)
	grids: npt.NDArray[np.double] = np.array((xv, pv), dtype=np.double).T.reshape(-1, evolve.PHASEDIM)
	xv, pv = plot_region(xv), plot_region(pv)
	# the regressor
	rescaled_errors: npt.NDArray[np.double] = np.empty((total_ticks, evolve.NUM_ELM))
	original_errors: npt.NDArray[np.double] = np.empty((total_ticks, evolve.NUM_ELM))
	evolving_errors: npt.NDArray[np.double] = np.empty((total_ticks, evolve.NUM_ELM))
	# drawer
	fig: matplotlib.figure.Figure
	axs: np.ndarray[collections.abc.Sequence[collections.abc.Sequence[matplotlib.axes.Axes]], np.dtype[np.object_]]
	fig, axs = plt.subplots(NUM_PLOTS, evolve.NUM_PES ** 2, figsize=(FIGSIZE[0] * (evolve.NUM_PES ** 2), FIGSIZE[1] * NUM_PLOTS))
	# sampling. Initial sample from gaussian directly
	all_pts: npt.NDArray[np.double] = np.clip(sample.sample_extra_points(sample.init_sample(NUM_PTS, r0, sigma_r0), NUM_XTR_RATIO), grids_min, grids_max)
	all_density: npt.NDArray[np.double] = np.zeros(all_pts.shape[:-1])
	all_density[0] = np.exp(-np.sum(np.square((all_pts[0] - r0) / sigma_r0), axis=-1) / 2.0) / (2.0 * np.pi * np.prod(sigma_r0))
	# files for output
	pts_f: io.TextIOWrapper = open('points.txt', 'w')
	den_f: io.TextIOWrapper = open('density.txt', 'w')
	all_f: io.TextIOWrapper = open('all_grids.txt', 'w')
	err_f: io.TextIOWrapper = open('error.txt', 'w')

	def train_pred_draw(iTick: int):
		global rescaled_errors, original_errors, evolving_errors, fig, axs, all_pts, all_density, pts_f, den_f, all_f, err_f
		# get scale
		scale: npt.NDArray[np.double] = get_scale(data[iTick])
		print(iTick, scale)
		# interpolate the data
		y_all: npt.NDArray[np.double] = np.empty(all_density.shape, np.double)
		for iElement in range(evolve.NUM_ELM):
			interpolator: scipy.interpolate.RegularGridInterpolator = scipy.interpolate.RegularGridInterpolator((x_grids, p_grids), data[iTick, iElement], 'cubic')
			y_all[iElement] = interpolator(all_pts[iElement]) * scale[iElement]
		evolving_errors[iTick] = np.sum(np.square(y_all - all_density * scale[:, np.newaxis]), axis=-1)
		# fit
		gp.update(all_pts, y_all, NUM_PTS)
		gp.train()
		# predict
		pred: npt.NDArray[np.double] = np.empty((evolve.NUM_ELM, n_grids, n_grids))
		for iPES in range(evolve.NUM_PES):
			for jPES in range(evolve.NUM_PES):
				ElementIndex: int = iPES * evolve.NUM_PES + jPES
				pred[ElementIndex] = gp.predict(grids, ElementIndex).reshape(n_grids, n_grids)
				# pred /= np.where(scale != 0, scale, 1.0)
				original_errors[iTick, ElementIndex] = np.sum((pred[ElementIndex] / (1.0 if scale[ElementIndex] == 0.0 else scale[ElementIndex]) - data[iTick, ElementIndex]) ** 2)
				rescaled_errors[iTick, ElementIndex] = np.sum((pred[ElementIndex] - data[iTick, ElementIndex] * scale[ElementIndex]) ** 2)

				def contour_data(iPlot: int) -> npt.NDArray[np.double]:
					match iPlot:
						case 0 | 1:
							return pred[ElementIndex]
						case 2:
							return data[iTick, ElementIndex] * scale[ElementIndex]
						case 3:
							return pred[ElementIndex] - data[iTick, ElementIndex] * scale[ElementIndex]
						case _:
							raise ValueError("Unexpected Plot for Contour")

				for iPlot in range(NUM_PLOTS):
					ax: matplotlib.axes.Axes = axs[iPlot, ElementIndex]
					ax.contourf(
						xv,
						pv,
						plot_region(contour_data(iPlot).T),
						cmap=CMAP,
						norm=NORM,
						vmin=-LIMIT,
						vmax=LIMIT,
						locator=LOCATOR)
					if iPlot == 0:
						ax.scatter(all_pts[ElementIndex, :NUM_PTS, 0], all_pts[ElementIndex, :NUM_PTS, 1], s=0.5, c='black')
						ax.set_title('Rescaled {}\nRescale Factor = {}'.format(title[0] + get_RI_label(iPES, jPES), scale[ElementIndex]))
					elif iPlot == 1:
						ax.set_title('Rescaled Error = {}\nRescaled Predict Error = {}'.format(rescaled_errors[iTick, ElementIndex], evolving_errors[iTick, ElementIndex]))
		fig.suptitle('Tick = {}\nTotal Rescaled Error = {}\nTotal Rescaled Predict Error = {}'.format(iTick, np.sum(rescaled_errors[iTick]), np.sum(evolving_errors[iTick])))
		fig.savefig('Tick = {:03}.png'.format(iTick))
		# print to file
		np.savetxt(pts_f, all_pts.reshape(evolve.NUM_ELM, NUM_PTS * (NUM_XTR_RATIO + 1) * evolve.PHASEDIM), footer='\n', comments='')
		np.savetxt(den_f, all_density)
		all_density = y_all / np.where(scale != 0.0, scale, 1.0)[:, np.newaxis] # update the un-scaled density for prediction
		np.savetxt(den_f, all_density)
		gp.update(all_pts, all_density, NUM_PTS)
		np.savetxt(den_f, [gp.predict(all_pts[iElement], iElement) for iElement in range(evolve.NUM_ELM)], footer='\n', comments='')
		np.savetxt(all_f, pred.reshape(evolve.NUM_ELM, n_grids ** 2), footer='\n', comments='')
		np.savetxt(err_f, (original_errors[iTick], rescaled_errors[iTick], evolving_errors[iTick]), footer='\n', comments='')

	def init():
		global fig, axs
		for iPES in range(evolve.NUM_PES):
			for jPES in range(evolve.NUM_PES):
				ElementIndex: int = iPES * evolve.NUM_PES + jPES
				for iPlot in range(NUM_PLOTS):
					ax: matplotlib.axes.Axes = axs[iPlot, ElementIndex]
					ax.set_xlabel('x')
					ax.set_ylabel('p')
					ax.set_title('Rescaled ' + title[iPlot] + get_RI_label(iPES, jPES))
		fig.colorbar(matplotlib.cm.ScalarMappable(cmap=CMAP, norm=NORM), ax=axs.ravel().tolist())
		train_pred_draw(0)
		return fig, axs,

	def draw(iTick: int):
		global all_pts, all_density, fig, axs
		# evolve
		evolve.evolve(all_pts, all_density, NUM_PTS, OUTPUT_INTERVAL)
		train_pred_draw(iTick)
		return fig, axs,

	# draw animation
	matplotlib.animation.FuncAnimation(fig, draw, range(1, total_ticks), init).save('gpr.gif', 'imagemagick')
	pts_f.close()
	den_f.close()
	all_f.close()
	err_f.close()
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
		for iPES in range(evolve.NUM_PES):
			for jPES in range(evolve.NUM_PES):
				ElementIndex: int = iPES * evolve.NUM_PES + jPES
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
