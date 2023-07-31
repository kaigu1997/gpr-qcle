#!/usr/bin/env python3
import evolve
import pes

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
import sys
import time
import typing

FIGSIZE: tuple[float, float] = (6.4, 4.8)
CMAP: str = 'seismic'
LOCATOR: matplotlib.ticker.Locator = matplotlib.ticker.LogLocator(numticks=21)

mass: npt.NDArray[np.double] = np.full((pes.DIM,), 2000.0, np.double)
dt: float = 0.1


def read_input() -> npt.NDArray[np.cdouble]:
	data: npt.NDArray[np.double] = np.loadtxt('pwtdm.txt', np.double)
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
	exact_pts: npt.NDArray[np.double] = grids[n_grids // 4:n_grids // 4 * 3 + 1:8, n_grids // 4:n_grids // 4 * 3 + 1:8].reshape(-1, pes.PHASEDIM)
	exact_density: npt.NDArray[np.cdouble] = np.reshape(data[1, :, n_grids // 4:n_grids // 4 * 3 + 1:8, n_grids // 4:n_grids // 4 * 3 + 1:8], (pes.NUM_ELM, -1))
	all_density: npt.NDArray[np.cdouble] = np.empty(exact_density.shape, np.cdouble)
	# 2 cases: evolve to these places
	interpolators: list[tuple[scipy.interpolate.RegularGridInterpolator, scipy.interpolate.RegularGridInterpolator]] = [(scipy.interpolate.RegularGridInterpolator((x_grids, p_grids), data[0, iElement].real, 'cubic'), scipy.interpolate.RegularGridInterpolator((x_grids, p_grids), data[0, iElement].imag, 'cubic')) for iElement in range(pes.NUM_ELM)]

	def pred(r: npt.NDArray[np.double], i: int) -> npt.NDArray[np.cdouble]:
		if i // pes.NUM_PES == i % pes.NUM_PES:
			return interpolators[i][0](r).astype(np.cdouble)
		else:
			return interpolators[i][0](r) + 1.0j * interpolators[i][1](r)
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
	real_part: typing.Callable[[npt.NDArray[np.cdouble]], npt.NDArray[np.double]] = lambda arr: arr.view(np.double).reshape(arr.shape + (2,))[np.arange(pes.NUM_ELM), ..., np.tril(np.ones((pes.NUM_PES, pes.NUM_PES), np.int_), -1).reshape(pes.NUM_ELM)]
	with open('density.txt', 'w') as den_f, open('points.txt', 'w') as pts_f:
		np.savetxt(pts_f, exact_pts)
		np.savetxt(den_f, np.moveaxis(real_part(all_density), 0, -1).reshape(-1, pes.NUM_ELM), footer='\n', comments='')
		np.savetxt(den_f, np.moveaxis(real_part(exact_density), 0, -1).reshape(-1, pes.NUM_ELM))
	titles: list[str] = ['Absolute', 'Relative']
	NUM_PLOTS: int = len(titles)
	COLORBAR_LB: float = 1e-16
	fig: matplotlib.figure.Figure
	axs: np.ndarray[collections.abc.Sequence[collections.abc.Sequence[matplotlib.axes.Axes]], np.dtype[np.object_]]
	fig, axs = plt.subplots(NUM_PLOTS, pes.NUM_ELM, figsize=(FIGSIZE[0] * pes.NUM_ELM, FIGSIZE[1] * NUM_PLOTS))
	for iPlot in range(NUM_PLOTS):
		diff: npt.NDArray[np.double]
		match iPlot:
			case 0:
				diff = np.abs(real_part(all_density - exact_density))
			case 1:
				diff = np.abs(real_part(all_density) / real_part(exact_density) - 1.0)
			case _:
				raise ValueError('Unexpected Plot Number')
		norm: matplotlib.colors.Normalize = matplotlib.colors.LogNorm(np.power(10, np.floor(np.log10(np.maximum(np.min(diff), COLORBAR_LB)))), np.power(10, np.ceil(np.log10(np.max(diff)))), True)
		max_idx: npt.NDArray[np.int64] = np.argmax(diff, axis=1)
		for iElement in range(pes.NUM_ELM):
			ax: matplotlib.axes.Axes = axs[iPlot, iElement]
			ax.contourf(
				x_grids[n_grids // 4:n_grids // 4 * 3 + 1:8],
				p_grids[n_grids // 4:n_grids // 4 * 3 + 1:8],
				np.maximum(diff[iElement].reshape(n_grids // 16 + 1, n_grids // 16 + 1).T, COLORBAR_LB),
				cmap=CMAP,
				norm=norm,
				locator=LOCATOR)
			ax.set_title(
				'{} Error of {}\nMaximum = {}'.format(
					titles[iPlot],
					get_RI_label(iElement // pes.NUM_PES, iElement % pes.NUM_PES),
					np.max(diff[iElement])
				)
			)
			ax.scatter(x_grids[n_grids // 4:n_grids // 4 * 3 + 1:8][max_idx[iElement] // (n_grids // 16 + 1)], p_grids[n_grids // 4:n_grids // 4 * 3 + 1:8][max_idx[iElement] % (n_grids // 16 + 1)], 15, 'black')
		fig.colorbar(matplotlib.cm.ScalarMappable(cmap=CMAP, norm=norm), ax=axs[iPlot].ravel().tolist())
	fig.savefig('diff.png')
