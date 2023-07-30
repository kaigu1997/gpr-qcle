import numpy as np
import numpy.typing as npt
import matplotlib.axes
import matplotlib.colors
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.ticker
import typing

FIGSIZE: tuple[float, float] = (6.4, 4.8)
LIMIT: float = 1.0
CMAP: str = 'seismic'
LOCATOR: matplotlib.ticker.MaxNLocator = matplotlib.ticker.MaxNLocator(nbins=21)
NORM: matplotlib.colors.CenteredNorm = matplotlib.colors.CenteredNorm(0.0, LIMIT, True)
length: typing.Literal[641] = 641
factors: npt.NDArray[np.int64] = np.array([i for i in range(1, length) if (length - 1) % i == 0])
data: npt.NDArray[np.double] = np.loadtxt('pwtdm.txt')[233 * 8 + 3].reshape(length, length)
data /= np.max(np.abs(data))
fig: matplotlib.figure.Figure
axs: np.ndarray
fig, axs = plt.subplots(factors.size, 1, figsize=(FIGSIZE[0], FIGSIZE[1] * factors.size))
for i in range(factors.size):
	ax: matplotlib.axes.Axes = axs[i]
	ax.contourf(
		data[::factors[i], ::factors[i]].T,
		cmap=CMAP,
		norm=NORM,
		vmin=-LIMIT,
		vmax=LIMIT,
		locator=LOCATOR)
	ax.set_title('{} grids'.format((length - 1) // factors[i] + 1))
fig.savefig('diff_grids.png')
