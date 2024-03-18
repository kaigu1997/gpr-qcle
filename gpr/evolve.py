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


def is_coupling(x: npt.NDArray[np.double], p: npt.NDArray[np.double], mass: npt.NDArray[np.double], dt: float) -> npt.NDArray[np.bool_]:
	"""
	To judge if there is strong enough non-adiabatic coupling at given phase coordinates

	Parameters
	----------
	x : npt.NDArray[np.double], shape of (..., DIM)
		Positions of interest
	p : npt.NDArray[np.double], shape of (..., DIM)
		Momenta of interest
	mass : npt.NDArray[np.double], shape of (DIM,)
		Mass of classical degree of freedom
	dt : float
		Time interval

	Returns
	-------
	npt.NDArray[np.bool_], shape of (..., DIM)
		Coupling of each point and each dimension
	"""
	is_coupling.CouplingCriterion = 0.0
	force: npt.NDArray[np.double] = pes.adiabatic_force(x) # ... * D * N * N
	diag_f: npt.NDArray[np.double] = np.average(np.diagonal(force, 0, -2, -1), -1)[..., np.newaxis, np.newaxis] # ... * D * 1 * 1
	nac: npt.NDArray[np.double] = pes.adiabatic_coupling(x) # ... * D * N * N
	return np.any(np.logical_or(np.abs(np.tril(force, -1)) > np.abs(is_coupling.CouplingCriterion * diag_f), dt * (p / mass)[:, np.newaxis, np.newaxis] * nac > is_coupling.CouplingCriterion), axis=(-2, -1))


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
	IsCouple: npt.NDArray[np.bool_],
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
	IsCouple : npt.NDArray[np.bool_], shape of (..., DIM)
		The coupling at each phase points of each dimension
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
				phi: npt.NDArray[np.double] = np.sum(p_part / mass * pes.adiabatic_coupling(x_part)[..., 0, 1] * is_coupling(x_part, p_part, mass, dt_offdiag).astype(np.double), -1) # v.dot(NAC), ...
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
			p2: npt.NDArray[np.double] = p1 + dt * evolve_density_non_adiabatically.offdiagonal_branches.reshape((-1,) + tuple(1 for i in range(IsCouple.ndim))) * (pes.adiabatic_force(x2)[..., 0, 1] * is_coupling(x2, p1, mass, dt).astype(np.double)) # 3 * ... * D
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
	evolve_element_density.drc = Direction.Forward
	# judge coupling
	IsCouple: npt.NDArray[np.bool_] = is_coupling(x4, p2, mass, dt) # M * D
	IsCouplePerPoint: npt.NDArray[np.bool_] = np.any(IsCouple, -1) # M
	adiabatic_indices: npt.NDArray[np.int_] = np.flatnonzero(np.logical_not(IsCouplePerPoint)) # adiabatic points
	non_adiabatic_indices: npt.NDArray[np.int_] = np.flatnonzero(IsCouplePerPoint) # non-adiabatic points
	evolve_density_adiabatically(density.ravel()[adiabatic_indices], x0.reshape(-1, pes.DIM)[adiabatic_indices], x2.reshape(-1, pes.DIM)[adiabatic_indices], x4.reshape(-1, pes.DIM)[adiabatic_indices], evolve_element_density.drc, dt, RowIndex, ColIndex)
	density[non_adiabatic_indices] = evolve_density_non_adiabatically(
		density[non_adiabatic_indices],
		x4.reshape(-1, pes.DIM)[non_adiabatic_indices],
		p2.reshape(-1, pes.DIM)[non_adiabatic_indices],
		x2.reshape(-1, pes.DIM)[non_adiabatic_indices],
		p1.reshape(-1, pes.DIM)[non_adiabatic_indices],
		mass,
		IsCouple.reshape(-1, pes.DIM)[non_adiabatic_indices],
		dt,
		predictor,
		RowIndex,
		ColIndex
	)


def evolve(
	points: list[npt.NDArray[np.double]],
	densities: list[npt.NDArray[np.cdouble]],
	mass: npt.NDArray[np.double],
	dt: float,
	predictor: typing.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]
) -> None:
	"""
	To evolve the given points and density matrices

	Parameters
	----------
	points : list[npt.NDArray[np.double]], len of NUM_TRIG, each of shape (NUM_PTS, PHASEDIM)
		Phase space coordinates of selected points for each density matrix element
	densities : list[npt.NDArray[np.cdouble]], len of NUM_TRIG, each of shape (NUM_PTS,)
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
	points : npt.NDArray[np.double], shape of (NUM_TRIG * NUM_PTS, PHASEDIM)
		Phase space coordinates of selected points for each density matrix element
	densities : npt.NDArray[np.cdouble], shape of (NUM_TRIG * NUM_PTS,)
		Density matrix element of the points
	belonging_idx : npt.NDArray[np.int_], shape of (NUM_TRIG * NUM_PTS,)
		The index of density matrix element where the points belong to
	mass : npt.NDArray[np.double], shape of (DIM,)
		Mass of classical degree of freedom
	dt : float
		Time interval
	predictor : typing.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]
		It predicts the density matrix element based on given coordinates and element index
	"""
	# evolve coordinates and hopping
	x0: npt.NDArray[np.double] = points[:, :pes.DIM] # N * D
	p0: npt.NDArray[np.double] = points[:, pes.DIM:] # N * D
	x2: npt.NDArray[np.double] = np.empty_like(x0) # N * D
	p1: npt.NDArray[np.double] = np.empty_like(p0) # N * D
	x4: npt.NDArray[np.double] = np.empty_like(x0) # N * D
	p2: npt.NDArray[np.double] = np.empty_like(p0) # N * D
	current_indices: npt.NDArray[np.bool_] = belonging_idx == pes.tril_element_indices[:, np.newaxis]
	for iBelongPES in range(pes.NUM_PES):
		for jBelongPES in range(iBelongPES + 1):
			TrilIndex = pes.flatten_tril_index[iBelongPES, jBelongPES]
			indices: npt.NDArray[np.int_] = np.flatnonzero(current_indices[TrilIndex]) # npt
			num_pts: int = indices.size
			if num_pts > 0: # only have elements
				# get x and p, and 2 semi adiabatic steps
				x2[indices], p1[indices] = evolve_coordinates_adiabatically(x0[indices], p0[indices], mass, dt / 2.0, Direction.Forward, iBelongPES, jBelongPES)
				x4[indices], p2[indices] = evolve_coordinates_adiabatically(x2[indices], p1[indices], mass, dt / 2.0, Direction.Forward, iBelongPES, jBelongPES)
				# surface hopping
				num_states_avail: int = (1 if iBelongPES == jBelongPES else 2) * (pes.NUM_PES - 1)
				state_prob: npt.NDArray[np.double] = np.zeros((num_pts, num_states_avail + 1))
				# weights
				velocity: npt.NDArray[np.double] = p2[indices] / mass # npt * D
				coupling: npt.NDArray[np.double] = pes.adiabatic_coupling(x4[indices])[..., coup_row_idx[TrilIndex, :num_states_avail], coup_col_idx[TrilIndex, :num_states_avail]] # npt * D * nst
				state_prob[:, :-1] = np.abs(np.sum(velocity[..., np.newaxis] * coupling, -2) * dt) # npt * nst, |v*d*dt|
				# energy conservation rule
				potential: npt.NDArray[np.double] = pes.adiabatic_potential(x4[indices]) # npt * N
				momentum_rescale_factor_sq: npt.NDArray[np.double] = 1.0\
					+ (potential[:, coup_col_idx[TrilIndex, :num_states_avail]] - potential[:, coup_row_idx[TrilIndex, :num_states_avail]]) / np.sum(p2[indices] ** 2 / mass, -1, keepdims=True) # npt * nst
				state_prob[:, :-1] *= np.where(momentum_rescale_factor_sq >= 0.0, 1.0, 0.0)
				state_prob[:, -1] = 1.0 # stay at original state
				state_prob /= np.sum(state_prob, -1, keepdims=True) # normalize
				# choose the one to jump to
				idx_of_dest_idx: npt.NDArray[np.int_] = np.array([sample.np_rng.choice(num_states_avail + 1, p=p) for p in state_prob])
				# change index, momentum, and the back propagation
				transition: npt.NDArray[np.bool_] = idx_of_dest_idx != num_states_avail
				if np.any(transition):
					belonging_idx[indices[transition]] = dest_idx[TrilIndex, idx_of_dest_idx[transition]] # indices[transition] equivalent to [current_indices[TrilIndex]][transition]
					p2[indices[transition]] *= np.sqrt(momentum_rescale_factor_sq[transition, idx_of_dest_idx[transition]])[:, np.newaxis]
					# change density
					for iPredictPES in range(pes.NUM_PES):
						for jPredictPES in range(iPredictPES + 1):
							if iBelongPES == iPredictPES and jBelongPES == jPredictPES:
								pass
							PredictElementIndex: int = iPredictPES * pes.NUM_PES + jPredictPES
							need_predict: npt.NDArray[np.bool_] = dest_idx[TrilIndex, idx_of_dest_idx[transition]] == PredictElementIndex # in case the transition happens to this element
							if np.any(need_predict):
								x2[indices[transition][need_predict]], p1[indices[transition][need_predict]] = evolve_coordinates_adiabatically(x4[indices[transition][need_predict]], p2[indices[transition][need_predict]], mass, dt / 2.0, Direction.Backward, iPredictPES, jPredictPES)
								x0[indices[transition][need_predict]], p0[indices[transition][need_predict]] = evolve_coordinates_adiabatically(x2[indices[transition][need_predict]], p1[indices[transition][need_predict]], mass, dt / 2.0, Direction.Backward, iPredictPES, jPredictPES)
								densities[indices[transition][need_predict]] = predictor(points[indices[transition][need_predict]], PredictElementIndex)
	# evolve density
	for iPES in range(pes.NUM_PES):
		for jPES in range(iPES + 1):
			filters: npt.NDArray[np.bool_] = belonging_idx == iPES * pes.NUM_PES + jPES
			density_copy: npt.NDArray[np.cdouble] = np.copy(densities[filters])
			evolve_element_density(
				x0[filters],
				p0[filters],
				x2[filters],
				p1[filters],
				x4[filters],
				p2[filters],
				density_copy,
				mass,
				dt,
				predictor,
				iPES,
				jPES
			)
			densities[filters] = density_copy
			# finally set up the point coordinates
			points[filters, :pes.DIM] = x4[filters]
			points[filters, pes.DIM:] = p2[filters]
