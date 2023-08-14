#!/usr/bin/env python3
import evolve

import collections.abc
import matplotlib.animation
import matplotlib.axes
import matplotlib.colors
import matplotlib.cm
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import numpy.typing as npt
import scipy.interpolate
import typing

FIGSIZE: tuple[float, float] = (6.4, 4.8)
CMAP: str = 'seismic'
LOCATOR: matplotlib.ticker.Locator = matplotlib.ticker.LogLocator(numticks=21)


def read_input() -> npt.NDArray[np.cdouble]:
	data: npt.NDArray[np.double] = np.loadtxt('pwtdm.txt', np.double)
	ticks: int = data.shape[0] // (evolve.NUM_ELM * 2)
	length: int = int(np.round(np.sqrt(data.shape[1])))
	data = data.reshape((ticks, evolve.NUM_ELM, 2, length, length))
	result: npt.NDArray[np.cdouble] = np.empty((ticks, evolve.NUM_ELM, length, length), np.cdouble)
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


if __name__ == "__main__":
	# read input
	data: npt.NDArray[np.cdouble] = read_input()
	r0: npt.NDArray[np.double] = np.array([-1.6, 14.112], np.double)
	total_ticks: int = data.shape[0]
	n_grids: int = data.shape[-1]
	x_grids: npt.NDArray[np.double] = np.linspace(-16.0, 16.0, n_grids, True, dtype=np.double)
	dx: np.double = (x_grids[-1] - x_grids[0]) / (n_grids - 1)
	p_grids: npt.NDArray[np.double] = r0[1] + np.pi / 2.0 / dx * np.linspace(-1.0, 1.0, n_grids, True, dtype=np.double)
	grids: npt.NDArray[np.double] = np.array(np.meshgrid(x_grids, p_grids), dtype=np.double).T
	all_pts: npt.NDArray[np.double] = grids[n_grids // 4:n_grids // 4 * 3 + 1:8, n_grids // 4:n_grids // 4 * 3 + 1:8].reshape(-1, evolve.PHASEDIM)
	exact_density: npt.NDArray[np.cdouble] = np.reshape(np.moveaxis(data[1, :, n_grids // 4:n_grids // 4 * 3 + 1:8, n_grids // 4:n_grids // 4 * 3 + 1:8], 0, -1), (-1, evolve.NUM_ELM))
	all_density: npt.NDArray[np.cdouble] = np.empty(exact_density.shape, np.cdouble)
	np.savetxt('points.txt', all_pts)
	interpolators: list[tuple[scipy.interpolate.RegularGridInterpolator, scipy.interpolate.RegularGridInterpolator]] = [(scipy.interpolate.RegularGridInterpolator((x_grids, p_grids), data[0, iElement].real, 'cubic'), scipy.interpolate.RegularGridInterpolator((x_grids, p_grids), data[0, iElement].imag, 'cubic')) for iElement in range(evolve.NUM_ELM)]

	def pred(r: npt.NDArray[np.double], RowIndex: int, ColIndex: int) -> np.cdouble:
		ElementIndex: int = RowIndex * evolve.NUM_PES + ColIndex
		if RowIndex == ColIndex:
			return interpolators[ElementIndex][0](r)[0].astype(np.cdouble)
		else:
			return interpolators[ElementIndex][0](r)[0] + 1.0j * interpolators[ElementIndex][1](r)[0]

	for point, density in zip(all_pts, all_density):
		for iPES in range(evolve.NUM_PES):
			for jPES in range(iPES + 1):
				z: np.cdouble = evolve.non_adiabatic_evolve_predict(point[0], point[1], None, pred, iPES, jPES)
				density[iPES * evolve.NUM_PES + jPES] = z
				if iPES != jPES:
					density[jPES * evolve.NUM_PES + iPES] = np.conj(z)
	# files for output
	real_view: typing.Callable[[npt.NDArray[np.cdouble]], npt.NDArray[np.double]] = lambda arr: arr.view(np.double).reshape(arr.shape + (2,))
	with open('density.txt', 'w') as den_f:
		np.savetxt(den_f, np.moveaxis(real_view(all_density).reshape(-1, 2 * evolve.NUM_ELM), -1, 0), footer='\n', comments='')
		np.savetxt(den_f, np.moveaxis(real_view(exact_density).reshape(-1, 2 * evolve.NUM_ELM), -1, 0))
	titles: list[str] = ['Absolute', 'Relative']
	NUM_PLOTS: int = len(titles)
	COLORBAR_LB: float = 1e-16
	fig: matplotlib.figure.Figure
	axs: np.ndarray[collections.abc.Sequence[collections.abc.Sequence[matplotlib.axes.Axes]], np.dtype[np.object_]]
	fig, axs = plt.subplots(NUM_PLOTS, evolve.NUM_ELM, figsize=(FIGSIZE[0] * evolve.NUM_ELM, FIGSIZE[1] * NUM_PLOTS))
	for iPlot in range(NUM_PLOTS):
		diff: npt.NDArray[np.double]
		match iPlot:
			case 0:
				diff = np.abs(real_view(all_density - exact_density)[:, [0, 2, 2, 3], [0, 0, 1, 0]])
			case 1:
				diff = np.abs((real_view(all_density) / real_view(exact_density))[:, [0, 2, 2, 3], [0, 0, 1, 0]] - 1.0)
			case _:
				raise ValueError('Unexpected Plot Number')
		norm: matplotlib.colors.Normalize = matplotlib.colors.LogNorm(np.power(10, np.floor(np.log10(np.maximum(np.min(diff), COLORBAR_LB)))), np.power(10, np.ceil(np.log10(np.max(diff)))), True)
		max_idx: npt.NDArray[np.int64] = np.argmax(diff, 0)
		for iElement in range(evolve.NUM_ELM):
			ax: matplotlib.axes.Axes = axs[iPlot, iElement]
			ax.contourf(
				x_grids[n_grids // 4:n_grids // 4 * 3 + 1:8],
				p_grids[n_grids // 4:n_grids // 4 * 3 + 1:8],
				np.maximum(diff[:, iElement].reshape(n_grids // 16 + 1, n_grids // 16 + 1).T, COLORBAR_LB),
				cmap=CMAP,
				norm=norm,
				locator=LOCATOR)
			ax.set_title(
				'{} Error of {}\nMaximum = {}'.format(
					titles[iPlot],
					get_RI_label(iElement / evolve.NUM_PES, iElement % evolve.NUM_PES),
					np.max(diff[:, iElement])
				)
			)
			ax.scatter(x_grids[n_grids // 4:n_grids // 4 * 3 + 1:8][max_idx[iElement] // (n_grids // 16 + 1)], p_grids[n_grids // 4:n_grids // 4 * 3 + 1:8][max_idx[iElement] % (n_grids // 16 + 1)], 15, 'black')
		fig.colorbar(matplotlib.cm.ScalarMappable(cmap=CMAP, norm=norm), ax=axs[iPlot].ravel().tolist())
	fig.savefig('diff.png')
