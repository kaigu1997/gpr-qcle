#!/usr/bin/env python3
"""
main
====

The main module
"""
import collections.abc
import time
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
import pes

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


def get_RI_label(row: int, col: int) -> str:
	"""
	To get the label of the element

	Parameters
	----------
	row : int
		The row index
	col : int
		The column index

	Returns
	-------
	str
		The corresponding name
	"""
	if row == col:
		return r'$\rho_{%d,%d}$' % (row, col)
	elif row < col:
		return r'$\Re\rho_{%d,%d}$' % (col, row)
	else:
		return r'$\Im\rho_{%d,%d}$' % (row, col)


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
	The main driver

	Raises
	------
	ValueError
		_description_
	"""
	# read input
	mass: npt.NDArray[np.double]
	r0: npt.NDArray[np.double]
	sigma_r0: npt.NDArray[np.double]
	output_interval: float
	dt: float
	mass, r0, sigma_r0, output_interval, dt = read_input()
	output_steps: int = int(round(output_interval / dt))
	print("Mass = {}\nInitial center = {}\nInitial deviation = {}\nTime step = {}\nSteps between output = {}\n".format(mass, r0, sigma_r0, dt, output_steps))
	# read data
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
	exact_pts: npt.NDArray[np.double] = grids.reshape(-1, pes.PHASEDIM)
	np.savetxt('points.txt', exact_pts)
	all_density: npt.NDArray[np.cdouble] = data[0].reshape(pes.NUM_ELM, -1)
	interpolators: list[tuple[scipy.interpolate.RegularGridInterpolator, scipy.interpolate.RegularGridInterpolator]] = [(scipy.interpolate.RegularGridInterpolator((x_grids, p_grids), all_density[iElement].reshape(n_grids, n_grids).real, 'cubic', False, 0.0), scipy.interpolate.RegularGridInterpolator((x_grids, p_grids), all_density[iElement].reshape(n_grids, n_grids).imag, 'cubic', False, 0.0)) for iElement in range(pes.NUM_ELM)]
	# plot
	rescaled_errors: npt.NDArray[np.double] = np.empty((total_ticks, pes.NUM_ELM))
	original_errors: npt.NDArray[np.double] = np.empty((total_ticks, pes.NUM_ELM))
	titles: list[str] = ['Absolute', 'Relative']
	NUM_PLOTS: int = len(titles)
	COLORBAR_LB: float = 1e-16
	FIGSIZE: tuple[float, float] = (6.4, 4.8)
	CMAP: str = 'seismic'
	NORM: matplotlib.colors.Normalize = matplotlib.colors.LogNorm(COLORBAR_LB, 1e4, True)
	LOCATOR: matplotlib.ticker.Locator = matplotlib.ticker.LogLocator(numticks=21)
	fig: matplotlib.figure.Figure
	axs: np.ndarray[collections.abc.Sequence[collections.abc.Sequence[matplotlib.axes.Axes]], np.dtype[np.object_]]
	fig, axs = plt.subplots(NUM_PLOTS, pes.NUM_ELM, figsize=(FIGSIZE[0] * pes.NUM_ELM, FIGSIZE[1] * NUM_PLOTS))

	def pred(r: npt.NDArray[np.double], i: int) -> npt.NDArray[np.cdouble]:
		"""
		To predict

		Parameters
		----------
		r : npt.NDArray[np.double]
			The coordinates
		i : int
			The index of element

		Returns
		-------
		npt.NDArray[np.cdouble]
			Density of the element at the coordinates
		"""
		if i // pes.NUM_PES == i % pes.NUM_PES:
			return interpolators[i][0](r).astype(np.cdouble)
		else:
			return interpolators[i][0](r) + 1.0j * interpolators[i][1](r)

	with open('density.txt', 'w', encoding='UTF-8') as den_f,\
		open('error.txt', 'w', encoding='UTF-8') as err_f:
		def pred_draw(iTick: int) -> None:
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
			# predict
			original_errors[iTick] = np.sum(real_part(all_density - data[iTick].reshape(pes.NUM_ELM, -1)) ** 2, (-2, -1))
			rescaled_errors[iTick] = np.sum((real_part(all_density - data[iTick].reshape(pes.NUM_ELM, -1)) * scale[:, np.newaxis, np.newaxis]) ** 2, (-2, -1))

			for iPlot in range(NUM_PLOTS):
				contour_data: npt.NDArray[np.double]
				match iPlot:
					case 0:
						contour_data = np.abs(real_part(all_density - data[iTick].reshape(pes.NUM_ELM, -1)))
					case 1:
						with np.errstate(divide='ignore'):
							contour_data = np.abs(real_part(all_density) / np.where(real_part(data[iTick].reshape(pes.NUM_ELM, -1)) == 0.0, 1.0, real_part(data[iTick].reshape(pes.NUM_ELM, -1))) - 1.0)
					case _:
						raise NotImplementedError("Unexpected Plot for Contour")
				max_idx: npt.NDArray[np.int64] = np.argmax(contour_data, axis=1)
				for iElement in range(pes.NUM_ELM):
					ax: matplotlib.axes.Axes = axs[iPlot, iElement]
					ax.contourf(
						xv,
						pv,
						plot_region(np.maximum(contour_data[iElement].reshape(n_grids, n_grids).T, COLORBAR_LB)),
						cmap=CMAP,
						norm=NORM,
						locator=LOCATOR)
					ax.set_title(
						'{} Error of {}\nMaximum = {}'.format(
							titles[iPlot],
							get_RI_label(iElement // pes.NUM_PES, iElement % pes.NUM_PES),
							np.max(contour_data[iElement])
						)
					)
					ax.scatter(x_grids[max_idx[iElement] // n_grids], p_grids[max_idx[iElement] % n_grids], 15, 'black')
			fig.suptitle('Tick = {}\nTotal Rescaled Error = {}'.format(iTick, np.sum(rescaled_errors[iTick])))
			fig.savefig('Tick = {:03}.png'.format(iTick))
			# print to file
			np.savetxt(den_f, real_part(all_density))
			np.savetxt(err_f, (original_errors[iTick], rescaled_errors[iTick]), footer='\n', comments='')
			print(iTick, time.asctime(), flush=True)

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
						ax.set_title('Rescaled ' + titles[iPlot] + get_RI_label(iPES, jPES))
			fig.colorbar(matplotlib.cm.ScalarMappable(cmap=CMAP, norm=NORM), ax=axs.ravel().tolist())
			pred_draw(0)
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
			for i in range(output_steps):
				print(iTick, i, time.asctime(), flush=True)
				for iPES in range(pes.NUM_PES):
					for jPES in range(iPES + 1):
						ElementIndex: int = iPES * pes.NUM_PES + jPES
						all_density[ElementIndex] = evolve.evolve_density_non_adiabatically(
							None,
							exact_pts[:, :pes.DIM],
							exact_pts[:, pes.DIM:],
							mass,
							# evolve.is_coupling(
							# 	exact_pts[:, :pes.DIM],
							# 	exact_pts[:, pes.DIM:],
							# 	mass,
							# 	dt
							# ),
							np.ones(exact_pts.shape[:-1] + (1,), np.bool_),
							dt,
							pred,
							iPES,
							jPES
						)
						if iPES != jPES:
							all_density[jPES * pes.NUM_PES + iPES] = np.conj(all_density[ElementIndex])
				for iElement in range(pes.NUM_ELM):
					interpolators[iElement][0].values = all_density[iElement].reshape(n_grids, n_grids).real
					interpolators[iElement][1].values = all_density[iElement].reshape(n_grids, n_grids).imag
			pred_draw(iTick)
			return fig, axs

		# draw animation
		matplotlib.animation.FuncAnimation(fig, draw, range(1, total_ticks), init).save('diff.gif', 'imagemagick')


if __name__ == "__main__":
	main()
