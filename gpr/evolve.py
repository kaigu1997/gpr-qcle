"""
evolve
======
This module provides the adiabatic and non-adiabatic evolution scheme.
"""
import enum
import os
import sys
import typing

import numpy as np
import numpy.typing as npt

sys.path.append(os.path.dirname(__file__))

import pes
import sample


def calculate_destination_and_coupling_indices() -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]]:
	"""
	To calculate the row and column index of destination element from corresponding element, and the row and column index of the corresponding coupling element

	Returns
	-------
	tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]]
		Row index of destination element, Column index of destination element, row index of corresponding coupling element, and column index of corresponding coupling element.
		Notice the destination will always locate in the lower-triangular part (i.e., row index would be no less than column index). Destination on strictly upper-triangular elements will be transposed to lower part.
		There is no such limitation on coupling element.
	"""
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
	return dest_row_idx, dest_col_idx, coup_row_idx, coup_col_idx


dest_row_idx: npt.NDArray[np.int_]
dest_col_idx: npt.NDArray[np.int_]
coup_row_idx: npt.NDArray[np.int_] # also the diff index of destination
coup_col_idx: npt.NDArray[np.int_] # also the diff index of source
dest_row_idx, dest_col_idx, coup_row_idx, coup_col_idx = calculate_destination_and_coupling_indices()
dest_idx: npt.NDArray[np.int_] = dest_row_idx * pes.NUM_PES + dest_col_idx

class Direction(enum.IntEnum):
	"""
	Enumerate of directions

	Attributes
	----------
	Forward : typing.Literal[Direction.Forward]
		Indicating evolving forward. Its value is 1
	Backward : typing.Literal[Direction.Backward]
		Indicating evolving backward. Its value is -1
	"""
	Forward = 1
	Backward = -1


def adiabatic_component(x: npt.NDArray[np.double], p: npt.NDArray[np.double], mass: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
	"""
	To give the contribution of non-adiabaticity at given phase coordinates

	Parameters
	----------
	x : npt.NDArray[np.double], shape of (..., DIM)
		Positions of interest
	p : npt.NDArray[np.double], shape of (..., DIM)
		Momenta of interest
	mass : npt.NDArray[np.double], shape of (DIM,)
		Mass of classical degree of freedom

	Returns
	-------
	npt.NDArray[np.double], shape of (...)
		Coupling contribution of each point, range in [0, 1]
	"""
	adiabatic_component.FACTOR = 200 # dimensionless factor to magnify the weights
	nac: npt.NDArray[np.double] = pes.adiabatic_coupling(x) # ... * D * N * N
	E: npt.NDArray[np.double] = pes.adiabatic_potential(x) # ... * N
	strict_tril_row: npt.NDArray[np.int_]
	strict_tril_col: npt.NDArray[np.int_]
	strict_tril_row, strict_tril_col = np.tril_indices(pes.NUM_PES, -1)
	v_dot_d: npt.NDArray[np.double] = (p / mass)[..., np.newaxis] * nac[..., strict_tril_row, strict_tril_col] # ... * D * NUM_TRIG
	weights: npt.NDArray[np.double] = np.abs(v_dot_d.sum(axis=-2) * pes.HBAR / (E[..., strict_tril_row] - E[..., strict_tril_col])) # ... * NUM_TRIG
	# adiabatic: 1, non-adiabatic: exp(weights)-1; after normalization, adiabatic weights = exp(-weights)
	return np.exp(-weights.max(axis=-1) * adiabatic_component.FACTOR) # ...


def evolve_coordinates_adiabatically(
	x0: npt.NDArray[np.double],
	p0: npt.NDArray[np.double],
	mass: npt.NDArray[np.double],
	dt: float,
	drc: Direction,
	RowIndex: int,
	ColIndex: int
) -> tuple[npt.NDArray[np.double], npt.NDArray[np.double]]:
	"""
	To evolve the phase space coordinates adiabatically of given element for given interval along given direction

	Parameters
	----------
	x0 : npt.NDArray[np.double], shape of (..., DIM)
		Starting positions
	p0 : npt.NDArray[np.double], shape of (..., DIM)
		Starting momenta
	mass : npt.NDArray[np.double], shape of (DIM,)
		Mass of classical degree of freedom
	dt : float
		Time interval
	drc : Direction
		The direction of evolution
	RowIndex : int
		Index of row of the element in density matrix
	ColIndex : int
		Index of column of the element in density matrix

	Returns
	-------
	tuple[npt.NDArray[np.double], npt.NDArray[np.double]]
		The destination positions and momenta
	"""
	def position_evolve(x: npt.NDArray[np.double], p: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
		"""
		To evolve positions for half step

		Parameters
		----------
		x : npt.NDArray[np.double], shape of (..., DIM)
			Positions
		p : npt.NDArray[np.double], shape of (..., DIM)
			Momenta

		Returns
		-------
		npt.NDArray[np.double], shape of (..., DIM)
			Positions after half step
		"""
		return x + drc.value * dt / 2.0 * p / mass

	def momentum_diagonal_nonbranch_evolve(x: npt.NDArray[np.double], p: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
		"""
		To evolve momenta

		Parameters
		----------
		x : npt.NDArray[np.double], shape of (..., DIM)
			Positions
		p : npt.NDArray[np.double], shape of (..., DIM)
			Momenta

		Returns
		-------
		npt.NDArray[np.double], shape of (..., DIM)
			Momenta after evolving the step
		"""
		force: npt.NDArray[np.double] = pes.adiabatic_force(x) # ... * D * N * N
		return p + drc.value * dt / 2.0 * (force[..., RowIndex, RowIndex] + force[..., ColIndex, ColIndex])

	x1: npt.NDArray[np.double] = position_evolve(x0, p0)
	p1: npt.NDArray[np.double] = momentum_diagonal_nonbranch_evolve(x1, p0)
	return position_evolve(x1, p1), p1


def evolve_density_adiabatically(
	density: npt.NDArray[np.cdouble],
	x0: npt.NDArray[np.double],
	x2: npt.NDArray[np.double],
	x4: npt.NDArray[np.double] | None,
	drc: Direction,
	dt: float,
	RowIndex: int,
	ColIndex: int
) -> None:
	"""
	To evolve the given density matrix element adiabatically

	Parameters
	----------
	density : npt.NDArray[np.cdouble], shape of (...)
		The density matrix elements to evolve adiabatically
	x0 : npt.NDArray[np.double], shape of (..., DIM)
		The initial positions
	x2 : npt.NDArray[np.double], shape of (..., DIM)
		The final positions if x4 does not exist, otherwise the intermediate
	x4 : npt.NDArray[np.double], shape of (..., DIM)
		The final positions if exists
	drc : Direction
		The direction of evolution
	dt : float
		Time interval
	RowIndex : int
		Index of row of the element in density matrix
	ColIndex : int
		Index of column of the element in density matrix
	"""
	if RowIndex != ColIndex:
		E0: npt.NDArray[np.double] = pes.adiabatic_potential(x0)
		E2: npt.NDArray[np.double] = pes.adiabatic_potential(x2)
		if x4 is not None:
			E4: npt.NDArray[np.double] = pes.adiabatic_potential(x4)
			density *= np.exp(-drc.value * dt / 4.0 / pes.HBAR * 1.0j * (E0[..., RowIndex] - E0[..., ColIndex] + 2.0 * (E2[..., RowIndex] - E2[..., ColIndex]) + E4[..., RowIndex] - E4[..., ColIndex]))
		else:
			density *= np.exp(-drc.value * dt / 2.0 / pes.HBAR * 1.0j * (E0[..., RowIndex] - E0[..., ColIndex] + E2[..., RowIndex] - E2[..., ColIndex]))


def evolve_density_non_adiabatically(
	density: npt.NDArray[np.cdouble] | None,
	x0: npt.NDArray[np.double],
	p0: npt.NDArray[np.double],
	x2: npt.NDArray[np.double],
	p1: npt.NDArray[np.double],
	mass: npt.NDArray[np.double],
	dt: float,
	predictor: typing.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]],
	RowIndex: int,
	ColIndex: int
) -> npt.NDArray[np.cdouble]:
	"""
	To predict the density matrix element at the given phase space coordinates according to non-adiabatic dynamics

	Parameters
	----------
	x0 : npt.NDArray[np.double], shape of (..., DIM)
		The positions to predict density
	p0 : npt.NDArray[np.double], shape of (..., DIM)
		The momenta to predict density
	x2 : npt.NDArray[np.double], shape of (..., DIM)
		The positions of half step before
	p1 : npt.NDArray[np.double], shape of (..., DIM)
		The momenta of half step before
	density : npt.NDArray[np.cdouble], shape of (...) | None
		The density matrix element of given index at give points, or not given. Notice all the elements passed to this function is in lower-triangular part.
	mass : npt.NDArray[np.double], shape of (DIM,)
		Mass of classical degree of freedom
	dt : float
		Time interval
	predictor : typing.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]
		The function that gives the density matrix element at corresponding phase points
	RowIndex : int
		Index of row of the element in density matrix
	ColIndex : int
		Index of column of the element in density matrix

	Returns
	-------
	npt.NDArray[np.cdouble], shape of (...)
		The density after back-propagated non-adiabatic evolution

	Raises
	------
	NotImplementedError
		In case the model is unknown
	"""
	evolve_density_non_adiabatically.drc = Direction.Backward
	evolve_density_non_adiabatically.offdiagonal_branches = np.array([-1, 0, 1], np.int_)
	evolve_density_non_adiabatically.offdiagonal_zero_branch_index = np.argwhere(evolve_density_non_adiabatically.offdiagonal_branches == 0)[0, 0]
	match pes.NUM_PES:
		case 2:
			def offdiagonal_rotation(rho_part: npt.NDArray[np.cdouble], x_part: npt.NDArray[np.double], p_part: npt.NDArray[np.double], dt_offdiag: float) -> None:
				"""
				To have a off-diagonal rotation on the given density matrices

				Parameters
				----------
				rho_part : npt.NDArray[np.cdouble], shape of (NUM_TRIG * ...)
					The density matrices
				x_part : npt.NDArray[np.double], shape of (... * D)
					The positions corresponding to the density matrices
				p_part : npt.NDArray[np.double], shape of (... * D)
					The momenta corresponding to the density matrices
				dt : float
					Time interval
				"""
				phi: npt.NDArray[np.double] = np.sum(p_part / mass * pes.adiabatic_coupling(x_part)[..., 0, 1], -1) # v.dot(NAC), ...
				sinphi: npt.NDArray[np.double] = np.sin(2.0 * dt_offdiag * phi)
				cosphi: npt.NDArray[np.double] = np.cos(2.0 * dt_offdiag * phi)
				rho_save: npt.NDArray[np.cdouble] = np.copy(rho_part)
				rho_part.real[0] = (1.0 + cosphi) / 2.0 * rho_save[0].real - sinphi * rho_save[1].real + (1.0 - cosphi) / 2.0 * rho_save[2].real
				rho_part.imag[0] = 0.0
				rho_part.real[1] = sinphi / 2.0 * rho_save[0].real + cosphi * rho_save[1].real - sinphi / 2.0 * rho_save[2].real
				rho_part.imag[1] = rho_save[1].imag
				rho_part.real[2] = (1.0 - cosphi) / 2.0 * rho_save[0].real + sinphi * rho_save[1].real + (1.0 + cosphi) / 2.0 * rho_save[2].real
				rho_part.imag[2] = 0.0

			# first step: (x0, p0) -> (x2, p1)
			# second step, off-diagonal branching to p2, and broadcast to x3
			# (backward) direction is included. So when evolve forward, the branch correspondence remains the same
			p2: npt.NDArray[np.double] = p1 + dt * evolve_density_non_adiabatically.offdiagonal_branches.reshape((-1,) + tuple(1 for _ in range(x0.ndim))) * pes.adiabatic_force(x2)[..., 0, 1] # 3 * ... * D
			x3: npt.NDArray[np.double] = x2 + evolve_density_non_adiabatically.drc.value * dt / 4.0 * p2 / mass # 3 * ... * D
			# then adiabatic branching to p3, and broadcast to x4
			f_x3: npt.NDArray[np.double] = pes.adiabatic_force(x3) # 3 * ... * D * N * N
			f_diag_ave_tril: npt.NDArray[np.double] = (f_x3[..., pes.tril_row_indices, pes.tril_row_indices] + f_x3[..., pes.tril_col_indices, pes.tril_col_indices]) / 2.0 # 3 * ... * D * NUM_TRIG
			p3: npt.NDArray[np.double] = p2 + evolve_density_non_adiabatically.drc.value * dt / 2.0 * np.moveaxis(f_diag_ave_tril, -1, 0) # NUM_TRIG * 3 * ... * D
			x4: npt.NDArray[np.double] = x3 + evolve_density_non_adiabatically.drc.value * dt / 4.0 * p3 / mass # NUM_TRIG * 3 * ... * D
			# predict
			rho_predict: npt.NDArray[np.cdouble] = np.empty((pes.NUM_TRIG, 3) + x0.shape[:-1], np.cdouble) # NUM_TRIG * 3 * ...
			for index, iElement in enumerate(pes.tril_element_indices):
				branch_row_index: int = iElement // pes.NUM_PES
				branch_col_index: int = iElement % pes.NUM_PES
				rho_predict[index] = predictor(np.concatenate((x4[index], p3[index]), -1), iElement)
				# assign the known density to it
				if branch_row_index == RowIndex and branch_col_index == ColIndex and density is not None:
					rho_predict[index, evolve_density_non_adiabatically.offdiagonal_zero_branch_index] = density
				# first half-step adiabatic evolve. (x4, p3) -> (x2, p2) with an adiabatic rotation
				evolve_density_adiabatically(rho_predict[index], x2, x4[index], None, Direction.Forward, dt / 2.0, branch_row_index, branch_col_index)
			rho_combined_offdiag: npt.NDArray[np.cdouble] = np.zeros((pes.NUM_TRIG,) + rho_predict.shape[2:], np.cdouble) # NUM_TRIG * ...
			for index, branch in enumerate(evolve_density_non_adiabatically.offdiagonal_branches):
				# now they are at (x2, p2). A off-diagonal rotation is needed.
				offdiagonal_rotation(rho_predict[:, index], x2, p2[index], dt / 2.0)
				# then the off-diagonal force evolution combination with a rotation matrix, (x2, p2) -> (x2, p1)
				value: npt.NDArray[np.double]
				match branch:
					case -1:
						rho_combined_offdiag.real += (rho_predict[0, index].real + 2.0 * rho_predict[1, index].real + rho_predict[2, index].real) / 4.0
					case 0:
						value = (rho_predict[0, index].real - rho_predict[2, index].real) / 2.0
						rho_combined_offdiag.real[0] += value
						rho_combined_offdiag.imag[1] += rho_predict[1, index].imag
						rho_combined_offdiag.real[2] -= value
					case 1:
						value = (rho_predict[0, index].real - 2.0 * rho_predict[1, index].real + rho_predict[2, index].real) / 4.0
						rho_combined_offdiag.real[0] += value
						rho_combined_offdiag.real[1] -= value
						rho_combined_offdiag.real[2] += value
					case _:
						raise ValueError("Unexpected Off-Diagonal Branch!")
			# the other off-diagonal rotation at (x2, p1)
			offdiagonal_rotation(rho_combined_offdiag, x2, p1, dt / 2.0)
			# another adiabatic step, (x2, p1) -> (x0, p0)
			trig_index: int = np.argwhere(pes.tril_element_indices == RowIndex * pes.NUM_PES + ColIndex)[0, 0]
			evolve_density_adiabatically(rho_combined_offdiag[trig_index], x0, x2, None, Direction.Forward, dt / 2.0, RowIndex, ColIndex)
			return rho_combined_offdiag[trig_index]
		case _:
			raise NotImplementedError("Model NOT Implemented!")


def evolve_element_density(
	x0: npt.NDArray[np.double],
	p0: npt.NDArray[np.double],
	x2: npt.NDArray[np.double],
	p1: npt.NDArray[np.double],
	x4: npt.NDArray[np.double],
	p2: npt.NDArray[np.double],
	density: npt.NDArray[np.cdouble],
	mass: npt.NDArray[np.double],
	dt: float,
	predictor: typing.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]],
	RowIndex: int,
	ColIndex: int
) -> None:
	"""
	To evolve the density and phase space coordinates.

	This routine is separated so that surface hopping could be added to dynamics easily.

	Parameters
	----------
	x0 : npt.NDArray[np.double], shape of (..., DIM)
		The initial positions
	p0 : npt.NDArray[np.double], shape of (..., DIM)
		The initial momenta
	x2 : npt.NDArray[np.double], shape of (..., DIM)
		The positions of half step later
	p1 : npt.NDArray[np.double], shape of (..., DIM)
		The momenta of half step later
	x4 : npt.NDArray[np.double], shape of (..., DIM)
		The positions of full step later
	p2 : npt.NDArray[np.double], shape of (..., DIM)
		The momenta of full step later
	density : npt.NDArray[np.cdouble], shape of (NUM_PTS,)
		Density matrix element of the points
	mass : npt.NDArray[np.double], shape of (DIM,)
		Mass of classical degree of freedom
	dt : float
		Time interval
	predictor : typing.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]
		It predicts the density matrix element based on given coordinates and element index
	RowIndex : int
		Index of row of the element in density matrix
	ColIndex : int
		Index of column of the element in density matrix
	"""
	density[...] = evolve_density_non_adiabatically(
		density,
		x4,
		p2,
		x2,
		p1,
		mass,
		dt,
		predictor,
		RowIndex,
		ColIndex
	)


def evolve(
	points: npt.NDArray[np.double],
	densities: npt.NDArray[np.cdouble],
	mass: npt.NDArray[np.double],
	dt: float,
	predictor: typing.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]
) -> None:
	"""
	To evolve the given points and density matrices

	Parameters
	----------
	points : npt.NDArray[np.double], shape of (NUM_TRIG, NUM_PTS, PHASEDIM)
		Phase space coordinates of selected points for each density matrix element
	densities : npt.NDArray[np.cdouble], shape of (NUM_TRIG, NUM_PTS)
		Density matrix element of the points
	mass : npt.NDArray[np.double], shape of (DIM,)
		Mass of classical degree of freedom
	dt : float
		Time interval
	predictor : typing.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]
		It predicts the density matrix element based on given coordinates and element index
	"""
	# lower triangular loop, evolve coordinates and density
	evolve.drc = Direction.Forward
	for iPES in range(pes.NUM_PES):
		for jPES in range(iPES + 1):
			TrilIndex = pes.flatten_tril_index[iPES, jPES]
			# get x and p, and 2 semi adiabatic steps
			x0: npt.NDArray[np.double] = points[TrilIndex][:, :pes.DIM] # M * D
			p0: npt.NDArray[np.double] = points[TrilIndex][:, pes.DIM:] # M * D
			x2: npt.NDArray[np.double] # M * D
			p1: npt.NDArray[np.double] # M * D
			x2, p1 = evolve_coordinates_adiabatically(x0, p0, mass, dt / 2.0, evolve.drc, iPES, jPES)
			x4: npt.NDArray[np.double] # M * D
			p2: npt.NDArray[np.double] # M * D
			x4, p2 = evolve_coordinates_adiabatically(x2, p1, mass, dt / 2.0, evolve.drc, iPES, jPES)
			evolve_element_density(x0, p0, x2, p1, x4, p2, densities[TrilIndex], mass, dt, predictor, iPES, jPES)
			# finally set up the point coordinates and density
			points[TrilIndex][:, :pes.DIM] = x4
			points[TrilIndex][:, pes.DIM:] = p2


def sh_evolve(
	points: npt.NDArray[np.double],
	densities: npt.NDArray[np.cdouble],
	belonging_idx: npt.NDArray[np.int_],
	mass: npt.NDArray[np.double],
	dt: float,
	predictor: typing.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]
) -> None:
	"""
	Non-adiabatic dynamics with surface hopping

	Parameters
	----------
	points : npt.NDArray[np.double], shape of (NUM_TRIG, NUM_PTS, PHASEDIM)
		Phase space coordinates of selected points for each density matrix element
	densities : npt.NDArray[np.cdouble], shape of (NUM_TRIG, NUM_PTS)
		Density matrix element of the points
	belonging_idx : npt.NDArray[np.int_], shape of (NUM_PTS,)
		The index of density matrix element where the points belong to
	mass : npt.NDArray[np.double], shape of (DIM,)
		Mass of classical degree of freedom
	dt : float
		Time interval
	predictor : typing.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]
		It predicts the density matrix element based on given coordinates and element index
	"""
	# evolve coordinates and hopping
	x0: npt.NDArray[np.double] = points[:, :, :pes.DIM] # TRIG * N * D
	p0: npt.NDArray[np.double] = points[:, :, pes.DIM:] # TRIG * N * D
	x2: npt.NDArray[np.double] = np.empty_like(x0) # TRIG * N * D
	p1: npt.NDArray[np.double] = np.empty_like(p0) # TRIG * N * D
	x4: npt.NDArray[np.double] = np.empty_like(x0) # TRIG * N * D
	p2: npt.NDArray[np.double] = np.empty_like(p0) # TRIG * N * D
	current_indices: npt.NDArray[np.bool_] = belonging_idx == pes.tril_element_indices[:, np.newaxis]
	all_transitions: npt.NDArray[np.bool_] = np.zeros(points.shape[1], np.bool_)
	for iPES in range(pes.NUM_PES):
		for jPES in range(iPES + 1):
			TrilIndex = pes.flatten_tril_index[iPES, jPES]
			indices: npt.NDArray[np.int_] = np.flatnonzero(current_indices[TrilIndex]) # NUM_PT
			num_pts: int = indices.size
			if num_pts > 0: # only have elements
				# get x and p, and 2 semi adiabatic steps
				x2[:, indices], p1[:, indices] = evolve_coordinates_adiabatically(x0[:, indices], p0[:, indices], mass, dt / 2.0, Direction.Forward, iPES, jPES)
				x4[:, indices], p2[:, indices] = evolve_coordinates_adiabatically(x2[:, indices], p1[:, indices], mass, dt / 2.0, Direction.Forward, iPES, jPES)
				# surface hopping
				num_states_avail: int = (1 if iPES == jPES else 2) * (pes.NUM_PES - 1)
				state_prob: npt.NDArray[np.double] = np.zeros((num_pts, num_states_avail + 1)) # +1 for keep the same
				# weights
				velocity: npt.NDArray[np.double] = p2[TrilIndex, indices] / mass # NUM_PT * DIM
				coupling: npt.NDArray[np.double] = pes.adiabatic_coupling(x4[TrilIndex, indices])[..., coup_row_idx[TrilIndex, :num_states_avail], coup_col_idx[TrilIndex, :num_states_avail]] # NUM_PT * DIM * NUM_STATE
				state_prob[:, :-1] = np.abs(np.sum(velocity[..., np.newaxis] * coupling, -2) * dt) # NUM_PT * NUM_STATE, |v*d*dt|
				# energy conservation rule
				potential: npt.NDArray[np.double] = pes.adiabatic_potential(x4[TrilIndex, indices]) # NUM_PT * NUM_PES
				momentum_rescale_factor_sq: npt.NDArray[np.double] = 1.0\
					+ (potential[:, coup_col_idx[TrilIndex, :num_states_avail]] - potential[:, coup_row_idx[TrilIndex, :num_states_avail]]) / np.sum(p2[TrilIndex, indices] ** 2 / mass, -1, keepdims=True) # NUM_PT * NUM_STATE
				state_prob[:, :-1] *= np.where(momentum_rescale_factor_sq >= 0.0, 1.0, 0.0)
				state_prob[:, -1] = 1.0 # stay at original state
				state_prob /= np.sum(state_prob, -1, keepdims=True) # normalize
				# choose the one to jump to
				idx_of_dest_idx: npt.NDArray[np.int_] = np.array([sample.np_rng.choice(num_states_avail + 1, p=p) for p in state_prob])
				# change index, momentum, and the back propagation
				transition: npt.NDArray[np.bool_] = idx_of_dest_idx != num_states_avail
				if np.any(transition):
					# indices[transition] equivalent to [current_indices[TrilIndex]][transition]
					# change belonging
					belonging_idx[indices[transition]] = dest_idx[TrilIndex, idx_of_dest_idx[transition]]
					# momentum jump
					p2[:, indices[transition]] *= np.sqrt(momentum_rescale_factor_sq[transition, idx_of_dest_idx[transition]])[:, np.newaxis]
					all_transitions[indices[transition]] = True
	for iPES in range(pes.NUM_PES):
		for jPES in range(iPES + 1):
			TrilIndex: int = pes.flatten_tril_index[iPES, jPES]
			ElementIndex: int = iPES * pes.NUM_PES + jPES
			# change source coordinates / density due to momentum jump / not belonging
			need_predict: npt.NDArray[np.bool_] = np.logical_or(all_transitions, belonging_idx != ElementIndex) # in case the transition happens to this element
			if np.any(need_predict):
				x2[TrilIndex, need_predict], p1[TrilIndex, need_predict] = evolve_coordinates_adiabatically(x4[TrilIndex, need_predict], p2[TrilIndex, need_predict], mass, dt / 2.0, Direction.Backward, iPES, jPES)
				x0[TrilIndex, need_predict], p0[TrilIndex, need_predict] = evolve_coordinates_adiabatically(x2[TrilIndex, need_predict], p1[TrilIndex, need_predict], mass, dt / 2.0, Direction.Backward, iPES, jPES)
				densities[TrilIndex, need_predict] = predictor(points[TrilIndex, need_predict], ElementIndex)
			# evolve density
			evolve_element_density(x0[TrilIndex], p0[TrilIndex], x2[TrilIndex], p1[TrilIndex], x4[TrilIndex], p2[TrilIndex], densities[TrilIndex], mass, dt, predictor, iPES, jPES)
	# finally set up the point coordinates
	x0[...] = x4
	p0[...] = p2
