r"""phase_factor
============
To show the accumulated adiabatic phase factor of coherence during evolution
"""
import os
import tarfile
import typing

import matplotlib.axes
import matplotlib.cm
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import evolve
import main
import pes


def main_func() -> None:
	ELM_IDX: typing.Literal[2] = 2
	ROW_IDX: typing.Literal[1]
	COL_IDX: typing.Literal[0]
	ROW_IDX, COL_IDX = ELM_IDX // pes.NUM_PES, ELM_IDX % pes.NUM_PES
	title: list[list[str]] = [[col_str + row_str for col_str in [main.get_RI_label(COL_IDX, ROW_IDX), main.get_RI_label(ROW_IDX, COL_IDX), "$|\\rho_{{{},{}}}|$".format(ROW_IDX, COL_IDX)]] for row_str in ["", " After Phase Factor Extraction"]]
	NUM_ROWS: int = len(title)
	NUM_COLS: int = len(title[0])
	mass: npt.NDArray[np.double]
	r0: npt.NDArray[np.double]
	sigma_r0: npt.NDArray[np.double]
	output_interval: float
	dt: float
	mass, r0, sigma_r0, output_interval, dt = main.read_input()
	output_steps: int = int(round(output_interval / dt))
	print("Mass = {}\nInitial center = {}\nInitial deviation = {}\nTime step = {}\nSteps between output = {}\n".format(mass, r0, sigma_r0, dt, output_steps))
	# read input
	data: npt.NDArray[np.cdouble] = main.read_data()[:, ELM_IDX, :, :] # only rho_10 is needed
	total_ticks: int = data.shape[0]
	n_grids: int = data.shape[-1]
	x_grids: npt.NDArray[np.double] = 2.0 * np.abs(r0[0]) * np.linspace(-1.0, 1.0, n_grids, True, dtype=np.double)
	dx: np.double = (x_grids[-1] - x_grids[0]) / (n_grids - 1)
	p_grids: npt.NDArray[np.double] = r0[1] + np.pi / 2.0 / dx * np.linspace(-1.0, 1.0, n_grids, True, dtype=np.double)
	xv: npt.NDArray[np.double]
	pv: npt.NDArray[np.double]
	xv, pv = np.meshgrid(x_grids, p_grids)
	# for evolving
	grids: npt.NDArray[np.double] = np.array((xv, pv), dtype=np.double).T.reshape(-1, pes.PHASEDIM)
	xv, pv = main.plot_region(xv), main.plot_region(pv)
	adiabatic_phase_factor_10: npt.NDArray[np.double] = np.zeros(grids.shape[:-1])
	# for plot
	fig: matplotlib.figure.Figure
	axs: np.ndarray
	fig, axs = plt.subplots(NUM_ROWS, NUM_COLS, figsize=(main.FIGSIZE[0] * NUM_COLS, main.FIGSIZE[1] * NUM_ROWS))
	for iPlot in range(NUM_ROWS):
		for jPlot in range(NUM_COLS):
			ax: matplotlib.axes.Axes = axs[iPlot, jPlot]
			ax.set_xlabel('x')
			ax.set_ylabel('p')
			ax.set_title("Rescaled " + title[iPlot][jPlot])
	fig.suptitle("Rescaled $\\rho_{1,0}$")
	fig.colorbar(matplotlib.cm.ScalarMappable(cmap=main.CMAP, norm=main.NORM), ax=axs.ravel().tolist())
	SCL_FNAME: typing.Literal["scales"] = "scales"
	with open(SCL_FNAME + ".txt", "w", encoding="UTF-8") as scl_f:
		for iTick in range(total_ticks):
			# plot
			data_now: list[npt.NDArray[np.cdouble]] = [data[iTick], data[iTick] / np.exp(1.0j * adiabatic_phase_factor_10).reshape(n_grids, n_grids)]
			for iPlot, data_to_plot in enumerate(data_now):
				for jPlot, data_reim in enumerate([data_to_plot.real, data_to_plot.imag, np.sqrt(data_to_plot.real ** 2 + data_to_plot.imag ** 2)]):
					ax: matplotlib.axes.Axes = axs[iPlot, jPlot]
					scale: float = 0.0 if np.all(data_reim == 0.0) else 1.0 / np.max(np.abs(data_reim))
					ax.contourf(xv, pv, scale * main.plot_region(data_reim).T, cmap=main.CMAP, norm=main.NORM, vmin=-main.LIMIT, vmax=main.LIMIT, locator=main.LOCATOR)
					ax.set_title("Rescaled " + title[iPlot][jPlot] + "\nRescale Factor = {:.6e}".format(scale))
					print(scale, file=scl_f)
			print("\n", file=scl_f)
			fig.suptitle("Tick = {}\nRescaled $\\rho_{{1,0}}$".format(iTick))
			fig.savefig("{:03}.png".format(iTick))
			# evolve
			for _ in range(output_steps):
				x0, p0 = grids[..., :pes.DIM], grids[..., pes.DIM:]
				x2, p1 = evolve.evolve_coordinates_adiabatically(x0, p0, mass, dt / 2.0, evolve.Direction.Backward, 1, 0)
				x4, p2 = evolve.evolve_coordinates_adiabatically(x2, p1, mass, dt / 2.0, evolve.Direction.Backward, 1, 0)
				E0 = pes.adiabatic_potential(x0)
				E2 = pes.adiabatic_potential(x2)
				E4 = pes.adiabatic_potential(x4)
				adiabatic_phase_factor_10 += -evolve.Direction.Forward.value * dt / 4.0 / pes.HBAR * (E0[..., 1] - E0[..., 0] + 2 * (E2[..., 1] - E2[..., 0]) + E4[..., 1] - E4[..., 0])
				grids[..., :pes.DIM] = x4
				grids[..., pes.DIM:] = p2

	plt.close(fig)
	# plot scale
	fig, ax = plt.subplots(figsize=main.FIGSIZE)
	scales: npt.NDArray[np.double] = np.loadtxt(SCL_FNAME + ".txt").reshape(total_ticks, NUM_ROWS, NUM_COLS)
	for iPlot in range(NUM_ROWS):
		for jPlot in range(NUM_COLS):
			ax.semilogy(np.arange(total_ticks) * output_interval, scales[:, iPlot, jPlot], label=title[iPlot][jPlot])
	ax.set_xlabel("t / a.u.")
	ax.set_ylabel("Rescale Factor")
	ax.legend()
	fig.suptitle("Rescale Factor")
	fig.savefig(SCL_FNAME + ".png")
	plt.close(fig)
	# tar figures
	with tarfile.open("ticks.tar.gz", "w:gz") as tf:
		for iTick in range(total_ticks):
			name: str = "{:03}.png".format(iTick)
			tf.add(name)
			os.remove(name)

if __name__ == "__main__":
	main_func()
