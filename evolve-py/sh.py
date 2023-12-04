"""
_summary_
"""
import collections.abc
import datetime
import io
import os
import tarfile
import typing

import matplotlib.animation
import matplotlib.artist
import matplotlib.axes
import matplotlib.cm
import matplotlib.colors
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import numpy.typing as npt

import evolve
import main
import pes
import sample

def main_func() -> None:
	"""
	The main routine
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
	coup_row_idx: npt.NDArray[np.int_] = np.repeat(np.tile(np.arange(pes.NUM_PES), 2)[np.newaxis], pes.NUM_TRIG, 0) # k, k
	coup_col_idx: npt.NDArray[np.int_] = np.repeat(np.concatenate((pes.tril_col_indices[:, np.newaxis], pes.tril_row_indices[:, np.newaxis]), -1), pes.NUM_PES, -1) # j, i
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
				x0: npt.NDArray[np.double] = pts[:, :pes.DIM] # N * D
				p0: npt.NDArray[np.double] = pts[:, pes.DIM:] # N * D
				x2: npt.NDArray[np.double] = np.empty_like(x0) # N * D
				p1: npt.NDArray[np.double] = np.empty_like(p0) # N * D
				x4: npt.NDArray[np.double] = np.empty_like(x0) # N * D
				p2: npt.NDArray[np.double] = np.empty_like(p0) # N * D
				current_indices: npt.NDArray[np.bool_] = belonging_idx == pes.tril_element_indices[:, np.newaxis]
				for iBelongPES in range(pes.NUM_PES):
					for jBelongPES in range(iBelongPES + 1):
						TrilIndex = pes.flatten_tril_index[iBelongPES, jBelongPES]
						indices: npt.NDArray[np.int_] = np.flatnonzero(current_indices[TrilIndex]) # n
						num_pts: int = indices.size
						if num_pts > 0: # only have elements
							# evolve
							x2[indices], p1[indices] = evolve.evolve_coordinates_adiabatically(x0[indices], p0[indices], mass, dt / 2.0, evolve.Direction.Forward, iBelongPES, jBelongPES)
							x4[indices], p2[indices] = evolve.evolve_coordinates_adiabatically(x2[indices], p1[indices], mass, dt / 2.0, evolve.Direction.Forward, iBelongPES, jBelongPES)
							# hopping
							# check energy availability
							num_states_avail: int = (1 if iBelongPES == jBelongPES else 2) * (pes.NUM_PES - 1)
							state_prob: npt.NDArray[np.double] = np.zeros((num_pts, num_states_avail + 1))
							# weights
							velocity: npt.NDArray[np.double] = p2[indices] / mass # npt * D
							coupling: npt.NDArray[np.double] = pes.adiabatic_coupling(x4[indices])[:, :, coup_row_idx[TrilIndex, :num_states_avail], coup_col_idx[TrilIndex, :num_states_avail]] # npt * D * nst
							state_prob[:, :-1] = np.abs(np.sum(velocity[..., np.newaxis] * coupling, -2) * dt) # npt * nst, |v*d*dt|
							# energy conservation rule
							potential: npt.NDArray[np.double] = pes.adiabatic_potential(x4[indices]) # npt * N
							momentum_rescale_factor_sq: npt.NDArray[np.double] = 1.0\
								+ (potential[:, coup_col_idx[TrilIndex, :num_states_avail]] - potential[:, coup_row_idx[TrilIndex, :num_states_avail]]) / np.sum(pts[indices, pes.DIM:] ** 2 / mass, -1, keepdims=True) # npt * nst
							state_prob[:, :-1] *= np.where(momentum_rescale_factor_sq >= 0, 1, 0)
							state_prob[:, -1] = 1.0 # stay at original state
							state_prob /= np.sum(state_prob, -1, keepdims=True) # normalize
							# choose the one to jump to
							idx_of_dest_idx: npt.NDArray[np.int_] = np.array([sample.np_rng.choice(num_states_avail + 1, p=p) for p in state_prob])
							# chaneg index and momentum
							transition: npt.NDArray[np.bool_] = idx_of_dest_idx != num_states_avail
							belonging_idx[indices[transition]] = dest_idx[TrilIndex, idx_of_dest_idx[transition]]
							p2[indices[transition]] *= np.sqrt(momentum_rescale_factor_sq[transition, idx_of_dest_idx[transition]])[:, np.newaxis]
							pts[indices, :pes.DIM], pts[indices, pes.DIM:] = x4[indices], p2[indices]
			# print to file
			np.savetxt(pts_f, pts.T, footer='\n', comments='')
			np.savetxt(bln_f, belonging_idx[np.newaxis], footer='\n', comments='')
		print("End FSSH", datetime.datetime.now())

	# drawer
	fig: matplotlib.figure.Figure
	axs: np.ndarray[collections.abc.Sequence[collections.abc.Sequence[matplotlib.axes.Axes]], np.dtype[np.object_]]
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
	for iTick in range(total_ticks):
		print("{}/{}".format(iTick + 1, total_ticks), datetime.datetime.now())
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
	with tarfile.open('ticks.tar.gz', 'w:gz') as tf:
		for iTick in range(total_ticks):
			name: str = 'Tick = {:03}.png'.format(iTick)
			tf.add(name)
			os.remove(name)


if __name__ == "__main__":
	main_func()
