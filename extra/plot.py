import matplotlib.animation
import matplotlib.axes
import matplotlib.colors
import matplotlib.cm
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import numpy.typing as npt
import typing

FIGSIZE: tuple[float, float] = (6.4, 4.8)
LIMIT: float = 1.0
CMAP: str = 'seismic'
LOCATOR: matplotlib.ticker.MaxNLocator = matplotlib.ticker.MaxNLocator(nbins=21)
NORM: matplotlib.colors.CenteredNorm = matplotlib.colors.CenteredNorm(0.0, LIMIT, True)
NUM_PES: typing.Literal[2] = 2
NUM_ELM: typing.Literal[4] = 4


def get_RI_label(row: int, col: int) -> str:
	if row == col:
		return r'$\rho_{%d,%d}$' % (row, col)
	elif row < col:
		return r'$\Re\rho_{%d,%d}$' % (col, row)
	else:
		return r'$\Im\rho_{%d,%d}$' % (row, col)


def data_preprocess(data: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
	scale: npt.NDArray[np.double] = np.zeros((NUM_ELM), np.double)
	# extract independent variables, then rescale to the same magnitude
	for idx in range(NUM_ELM):
		iPES: int
		jPES: int
		iPES, jPES = idx // NUM_PES, idx % NUM_PES
		if np.any(data[idx, 0 if iPES <= jPES else 1, :, :] != 0):
			scale[idx] = 1.0 / np.max(np.abs(data[idx, 0 if iPES <= jPES else 1, :, :]))
	return scale


if __name__ == '__main__':
	data: npt.NDArray = np.loadtxt('pwtdm.txt')
	ticks: int = data.shape[0] // (NUM_ELM * 2)
	n_grids: int = int(np.round(np.sqrt(data.shape[-1])))
	data = data.reshape((ticks, NUM_ELM, 2, n_grids, n_grids))
	r0: npt.NDArray[np.double] = np.array([-8.0, 14.112], np.double)
	dx: float = 4.0 * abs(r0[0]) / (n_grids - 1)
	x_grids: npt.NDArray[np.double] = 2.0 * abs(r0[0]) * np.linspace(-1.0, 1.0, n_grids, dtype=np.double)
	p_grids: npt.NDArray[np.double] = r0[1] + np.pi / 2.0 / dx * np.linspace(-1.0, 1.0, n_grids, dtype=np.double)
	xv, pv = np.meshgrid(x_grids, p_grids)
	# plot
	fig, axs = plt.subplots(NUM_PES, NUM_PES, figsize=(FIGSIZE[0] * NUM_PES, FIGSIZE[1] * NUM_PES))
	for iPES in range(NUM_PES):
		for jPES in range(NUM_PES):
			ax: matplotlib.axes.Axes = axs[iPES, jPES]
			ax.set_xlabel('x')
			ax.set_ylabel('p')
			ax.set_title(get_RI_label(iPES, jPES))
			ax.contourf(xv, pv, np.zeros(xv.shape), cmap=CMAP, norm=NORM, vmin=-LIMIT, vmax=LIMIT, locator=LOCATOR)
	fig.colorbar(matplotlib.cm.ScalarMappable(cmap=CMAP, norm=NORM), ax=axs.ravel().tolist())

	def draw(i):
		scale = data_preprocess(data[i])
		global fig, axs
		for iPES in range(NUM_PES):
			for jPES in range(NUM_PES):
				idx: int = iPES * NUM_PES + jPES
				axs[iPES, jPES].contourf(xv, pv, data[i, idx, 0 if iPES <= jPES else 1, :, :].T * scale[idx], cmap=CMAP, norm=NORM, vmin=-LIMIT, vmax=LIMIT, locator=LOCATOR)
		fig.suptitle('Tick = {}\nTime = {}'.format(i, i * 5.0))
		return fig, axs

	matplotlib.animation.FuncAnimation(fig, draw, ticks).save('rescaled.gif')
