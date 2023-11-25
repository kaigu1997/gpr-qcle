"""
_summary_
"""
import collections.abc
import typing

import matplotlib.axes
import matplotlib.colors
import matplotlib.cm
import matplotlib.figure
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import evolve
import main
import pes


def main_func() -> None:
	"""
	The main routine
	"""
	FIGSIZE: tuple[float, float] = (6.4, 4.8)
	CMAP: typing.Literal['gist_rainbow'] = 'gist_rainbow'
	NORM: matplotlib.colors.Normalize = matplotlib.colors.Normalize(0.0, 1.0, True)
	mass: npt.NDArray[np.double]
	r0: npt.NDArray[np.double]
	sigma_r0: npt.NDArray[np.double]
	output_interval: float
	dt: float
	mass, r0, sigma_r0, output_interval, dt = main.read_input()
	output_steps: int = int(round(output_interval / dt))
	# read input
	data: npt.NDArray[np.cdouble] = main.read_data() # t * NUM_ELM * grid * grid
	total_ticks: int = data.shape[0]
	n_grids: int = data.shape[-1]
	x_grids: npt.NDArray[np.double] = 2.0 * np.abs(r0[0]) * np.linspace(-1.0, 1.0, n_grids, True, dtype=np.double)
	dx: np.double = (x_grids[-1] - x_grids[0]) / (n_grids - 1)
	p_grids: npt.NDArray[np.double] = r0[1] + np.pi / 2.0 / dx * np.linspace(-1.0, 1.0, n_grids, True, dtype=np.double)
	# find the max points
	max_indices: npt.NDArray[np.int_] = np.argmax(np.abs(data).reshape(total_ticks * pes.NUM_ELM, n_grids ** 2), -1).reshape(total_ticks, pes.NUM_ELM)
	pts_current: npt.NDArray[np.double] = np.empty((pes.NUM_ELM, total_ticks, pes.PHASEDIM))
	pts_current[:, :, 0] = x_grids[max_indices.T // n_grids]
	pts_current[:, :, 1] = p_grids[max_indices.T % n_grids]
	pts_backprop: npt.NDArray[np.double] = np.copy(pts_current)
	iTick: int
	for iTick in range(total_ticks - 1, -1, -1): # going (total_ticks, 0]
		# evolve back
		if iTick != total_ticks - 1:
			iElement: int
			for iElement in range(pes.NUM_ELM):
				for _ in range(output_steps):
					pts_backprop[iElement, iTick + 1:, :pes.DIM], pts_backprop[iElement, iTick + 1:, pes.DIM:] = evolve.evolve_coordinates_adiabatically(
						pts_backprop[iElement, iTick + 1:, :pes.DIM],
						pts_backprop[iElement, iTick + 1:, pes.DIM:],
						mass,
						dt,
						evolve.Direction.Backward,
						iElement // pes.NUM_PES,
						iElement % pes.NUM_PES
					)
	# save the data
	with open("backtrace-pts_backprop.txt", 'w', encoding="UTF-8") as f:
		for iElement in range(pes.NUM_ELM):
			np.savetxt(f, pts_backprop[iElement].T)
			print('\n', file=f)
	# plot
	fig: matplotlib.figure.Figure
	axs: np.ndarray[collections.abc.Sequence[collections.abc.Sequence[matplotlib.axes.Axes]], np.dtype[np.object_]]
	fig, axs = plt.subplots(2, 4, figsize=(FIGSIZE[0] * pes.NUM_ELM, FIGSIZE[1] * 2))
	for iPlot in range(2):
		for iElement in range(pes.NUM_ELM):
			ax: matplotlib.axes.Axes = axs[iPlot, iElement]
			if iPlot == 0:
				ax.add_artist(matplotlib.patches.Ellipse((r0[0], r0[1]), 5.0 * sigma_r0[0], 5.0 * sigma_r0[1], edgecolor='black', facecolor='white'))
				ax.add_artist(matplotlib.patches.Ellipse((r0[0], r0[1]), 3.0 * sigma_r0[0], 3.0 * sigma_r0[1], edgecolor='black', facecolor='white'))
				ax.add_artist(matplotlib.patches.Ellipse((r0[0], r0[1]), sigma_r0[0], sigma_r0[1], edgecolor='black', facecolor='white'))
				ax.scatter(pts_backprop[iElement, :, 0], pts_backprop[iElement, :, 1], 3, np.arange(total_ticks, dtype=np.double) / total_ticks, cmap=CMAP, norm=NORM)
				ax.set_title("Initial Coordinates of Maximum of " + main.get_RI_label(iElement // pes.NUM_PES, iElement % pes.NUM_PES))
			else:
				ax.scatter(pts_current[iElement, :, 0], pts_current[iElement, :, 1], 3, np.arange(total_ticks, dtype=np.double) / total_ticks, cmap=CMAP, norm=NORM)
				ax.set_title("Maximum of " + main.get_RI_label(iElement // pes.NUM_PES, iElement % pes.NUM_PES))
			ax.set_xlim(x_grids[0], x_grids[-1])
			ax.set_ylim(p_grids[0], p_grids[-1])
			ax.set_xlabel('x / a.u.')
			ax.set_ylabel('p / a.u.')
	fig.savefig('History-of-Max-Grid.png')


if __name__ == "__main__":
	main_func()
