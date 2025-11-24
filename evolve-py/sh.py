r"""sh
==
The main routine for surface hopping
"""
import collections.abc
import datetime
import io
import os
import tarfile

import matplotlib.animation
import matplotlib.artist
import matplotlib.axes
import matplotlib.cm
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import evolve
import main
import pes
import sample


def main_sh() -> None:
	r"""The main routine for surface hopping
	"""
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
	xv: npt.NDArray[np.double]
	pv: npt.NDArray[np.double]
	xv, pv = np.meshgrid(x_grids, p_grids)
	xv, pv = main.plot_region(xv), main.plot_region(pv)
	# initial sample
	pts: npt.NDArray[np.double] = np.tile(sample.normal_sample(main.NUM_PTS, r0, sigma_r0)[np.newaxis], (pes.NUM_ELM, 1 + main.NUM_XTR_RATIO, 1))
	sample.sample_extra_points(main.NUM_PTS, pts)
	sample.sample_central_points(main.NUM_PTS, pts)
	pts = pts[pes.tril_element_indices, :main.NUM_PTS].reshape(pes.NUM_TRIG * main.NUM_PTS, pes.PHASEDIM)
	belonging_idx: npt.NDArray[np.int_] = np.repeat(pes.tril_element_indices, main.NUM_PTS, 0)
	# possible destinations of fssh
	dest_row_idx: npt.NDArray[np.int_] = np.concatenate(np.broadcast_arrays(pes.tril_row_indices[:, np.newaxis], np.arange(pes.NUM_PES)), -1) # i, k
	dest_col_idx: npt.NDArray[np.int_] = np.concatenate(np.broadcast_arrays(np.arange(pes.NUM_PES), pes.tril_col_indices[:, np.newaxis]), -1) # k, j
	coup_row_idx: npt.NDArray[np.int_] = dest_col_idx[:, np.concatenate((np.arange(pes.NUM_PES, 2 * pes.NUM_PES), np.arange(pes.NUM_PES)))] # j, k
	coup_col_idx: npt.NDArray[np.int_] = dest_row_idx[:, np.concatenate((np.arange(pes.NUM_PES, 2 * pes.NUM_PES), np.arange(pes.NUM_PES)))] # k, i
	filters: npt.NDArray[np.bool_] = np.logical_or(dest_row_idx != pes.tril_row_indices[:, np.newaxis], dest_col_idx != pes.tril_col_indices[:, np.newaxis]) # remove identical
	dest_row_idx = dest_row_idx[filters].reshape(pes.tril_element_indices.size, 2 * pes.NUM_PES - 2)
	dest_col_idx = dest_col_idx[filters].reshape(pes.tril_element_indices.size, 2 * pes.NUM_PES - 2)
	coup_row_idx = coup_row_idx[filters].reshape(pes.tril_element_indices.size, 2 * pes.NUM_PES - 2)
	coup_col_idx = coup_col_idx[filters].reshape(pes.tril_element_indices.size, 2 * pes.NUM_PES - 2)
	dest_row_idx, dest_col_idx = np.maximum(dest_row_idx, dest_col_idx), np.minimum(dest_row_idx, dest_col_idx) # order of coupling does not matter
	dest_idx: npt.NDArray[np.int_] = dest_row_idx * pes.NUM_PES + dest_col_idx
	# files for output
	pts_f: io.TextIOWrapper
	bln_f: io.TextIOWrapper
	with open("sh-points.txt", 'w', encoding='UTF-8') as pts_f,\
		open("sh-belonging.txt", 'w', encoding='UTF-8') as bln_f:
		print("Start FSSH", datetime.datetime.now())
		for iTick in range(total_ticks):
			for _ in range(output_steps):
				# save current indices
				current_indices: npt.NDArray[np.bool_] = belonging_idx == pes.tril_element_indices[:, np.newaxis]
				for iTrig in range(pes.NUM_TRIG):
					ElementIndex: int = pes.tril_element_indices[iTrig]
					num_pts: int = int(np.count_nonzero(current_indices[iTrig]))
					if num_pts > 0: # only have elements
						# evolve
						pts[current_indices[iTrig], :pes.DIM], pts[current_indices[iTrig], pes.DIM:] = evolve.evolve_coordinates_adiabatically(
							pts[current_indices[iTrig], :pes.DIM],
							pts[current_indices[iTrig], pes.DIM:],
							mass,
							dt,
							evolve.Direction.Forward,
							ElementIndex // pes.NUM_PES,
							ElementIndex % pes.NUM_PES
						)
						# hopping
						# choose the one to jump to
						idx_of_dest_idx: npt.NDArray[np.int_] = sample.np_rng.integers(2 * pes.NUM_PES - 2, size=num_pts) # n
						velocity: npt.NDArray[np.double] = pts[current_indices[iTrig], pes.DIM:] / mass # n * D
						coupling: npt.NDArray[np.double] = pes.adiabatic_coupling(pts[current_indices[iTrig], :pes.DIM])[np.arange(num_pts), :, coup_row_idx[iTrig, idx_of_dest_idx], coup_col_idx[iTrig, idx_of_dest_idx]] # n * D
						transition_rate: npt.NDArray[np.double] = np.abs(np.sum(velocity * coupling, -1) * dt) # n
						transition_prob: npt.NDArray[np.double] = transition_rate / (1.0 + transition_rate) # n
						# energy conservation
						potential: npt.NDArray[np.double] = pes.adiabatic_potential(pts[current_indices[iTrig], :pes.DIM]) # n * N
						momentum_rescale_factor_sq: npt.NDArray[np.double] = 1.0\
							+ np.where(
								idx_of_dest_idx < pes.NUM_PES - 1,
								potential[..., ElementIndex % pes.NUM_PES] - potential[np.arange(num_pts), dest_col_idx[iTrig, idx_of_dest_idx]], # diff on column, ij->ik
								potential[..., ElementIndex // pes.NUM_PES] - potential[np.arange(num_pts), dest_row_idx[iTrig, idx_of_dest_idx]] # diff on row, ij->kj
							)\
							/ np.sum(pts[current_indices[iTrig], pes.DIM:] ** 2 / mass, -1) # n
						# judgment
						transition: npt.NDArray[np.bool_] = np.logical_and(sample.np_rng.random(num_pts) < transition_prob, momentum_rescale_factor_sq >= 0.0) # n
						# chaneg index and momentum
						belonging_idx[current_indices[iTrig]] = np.where(transition, dest_idx[iTrig, idx_of_dest_idx], ElementIndex)
						pts[current_indices[iTrig], pes.DIM:] *= np.sqrt(np.where(transition, momentum_rescale_factor_sq, 1.0))[:, np.newaxis]
			# print to file
			np.savetxt(pts_f, pts.T, footer='\n', comments='')
			np.savetxt(bln_f, belonging_idx[np.newaxis], footer='\n', comments='')
		print("End FSSH", datetime.datetime.now())

	# drawer
	fig: matplotlib.figure.Figure
	axs: np.ndarray
	fig, axs = plt.subplots(2, pes.NUM_ELM, figsize=(main.FIGSIZE[0] * pes.NUM_ELM, main.FIGSIZE[1] * 2))
	for iElement in range(pes.NUM_ELM):
		for iPlot in range(2):
			ax: matplotlib.axes.Axes = axs[iPlot, iElement]
			ax.set_xlabel('x / a.u.')
			ax.set_ylabel('p / a.u.')
			if iPlot == 0:
				ax.set_title(main.get_RI_label(iElement // pes.NUM_PES, iElement % pes.NUM_PES))
	fig.colorbar(matplotlib.cm.ScalarMappable(cmap=main.CMAP, norm=main.NORM), ax=axs.ravel().tolist())
	all_pts: npt.NDArray[np.double] = np.loadtxt("sh-points.txt").reshape(total_ticks, pes.PHASEDIM, main.NUM_PTS * pes.NUM_TRIG)
	all_belonging: npt.NDArray[np.int_] = np.loadtxt("sh-belonging.txt").astype(np.int_)

	def draw(iTick: int) -> collections.abc.Iterable[matplotlib.artist.Artist | collections.abc.Iterable[matplotlib.artist.Artist]]:
		print("{}/{}".format(iTick, total_ticks), datetime.datetime.now())
		ct_data: npt.NDArray[np.double] = main.real_part(data[iTick])
		ct_data *= main.get_scale(ct_data)[:, np.newaxis, np.newaxis]
		for iElement in range(pes.NUM_ELM):
			for iPlot in range(2):
				ax: matplotlib.axes.Axes = axs[iPlot, iElement]
				ax.contourf(
					xv,
					pv,
					main.plot_region(ct_data[iElement].T),
					cmap=main.CMAP,
					norm=main.NORM,
					vmin=-main.LIMIT,
					vmax=main.LIMIT,
					locator=main.LOCATOR
				)
				if iPlot == 1:
					ax.scatter(
						np.clip(all_pts[iTick, 0, all_belonging[iTick] == max(iElement // pes.NUM_PES, iElement % pes.NUM_PES) * pes.NUM_PES + min(iElement // pes.NUM_PES, iElement % pes.NUM_PES)], x_grids[0], x_grids[-1]),
						np.clip(all_pts[iTick, 1, all_belonging[iTick] == max(iElement // pes.NUM_PES, iElement % pes.NUM_PES) * pes.NUM_PES + min(iElement // pes.NUM_PES, iElement % pes.NUM_PES)], p_grids[0], p_grids[-1]),
						0.5,
						'black'
					)
					ax.set_title("{} Points".format(np.count_nonzero(all_belonging[iTick] == max(iElement // pes.NUM_PES, iElement % pes.NUM_PES) * pes.NUM_PES + min(iElement // pes.NUM_PES, iElement % pes.NUM_PES))))
		fig.savefig('Tick = {:03}.png'.format(iTick))
		return fig, axs

	# draw animation
	matplotlib.animation.FuncAnimation(fig, draw, total_ticks).save('gpr.gif', 'imagemagick') # pyright: ignore[reportArgumentType]

	with tarfile.open('ticks.tar.gz', 'w:gz') as tf:
		for iTick in range(total_ticks):
			name: str = 'Tick = {:03}.png'.format(iTick)
			tf.add(name)
			os.remove(name)


if __name__ == "__main__":
	main_sh()
