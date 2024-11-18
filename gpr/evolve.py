"""
evolve
======
This module provides the adiabatic and non-adiabatic evolution scheme.
"""
import collections.abc
import enum
import os
import sys

import numpy as np
import numpy.typing as npt

sys.path.append(os.path.dirname(__file__))

import pes


class Direction(enum.IntEnum):
	"""
	Enumerate of directions

	Attributes
	----------
	FORWARD : typing.Literal[Direction.FORWARD]
		Indicating evolving forward. Its value is 1
	BACKWARD : typing.Literal[Direction.BACKWARD]
		Indicating evolving backward. Its value is -1
	"""
	FORWARD = 1
	BACKWARD = -1


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
	x2: npt.NDArray[np.double] | None,
	p1: npt.NDArray[np.double] | None,
	mass: npt.NDArray[np.double],
	dt: float,
	predictor: collections.abc.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]],
	RowIndex: int,
	ColIndex: int
) -> npt.NDArray[np.cdouble]:
	"""
	To predict the density matrix element at the given phase space coordinates according to non-adiabatic dynamics

	Parameters
	----------
	density : npt.NDArray[np.cdouble], shape of (...) | None
		The density matrix element of given index at give points, or not given. Notice all the elements passed to this function is in lower-triangular part.
	x0 : npt.NDArray[np.double], shape of (..., DIM)
		The positions to predict density
	p0 : npt.NDArray[np.double], shape of (..., DIM)
		The momenta to predict density
	x2 : npt.NDArray[np.double], shape of (..., DIM) | None
		The positions of half step before
	p1 : npt.NDArray[np.double], shape of (..., DIM) | None
		The momenta of half step before
	mass : npt.NDArray[np.double], shape of (DIM,)
		Mass of classical degree of freedom
	dt : float
		Time interval
	predictor : collections.abc.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]
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
	evolve_density_non_adiabatically.drc = Direction.BACKWARD
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
			if x2 is None or p1 is None:
				x2, p1 = evolve_coordinates_adiabatically(x0, p0, mass, dt / 2.0, Direction.BACKWARD, RowIndex, ColIndex)
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
				evolve_density_adiabatically(rho_predict[index], x2, x4[index], None, Direction.FORWARD, dt / 2.0, branch_row_index, branch_col_index)
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
			evolve_density_adiabatically(rho_combined_offdiag[trig_index], x0, x2, None, Direction.FORWARD, dt / 2.0, RowIndex, ColIndex)
			return rho_combined_offdiag[trig_index]
		case _:
			raise NotImplementedError("Model NOT Implemented!")


def evolve(
	points: list[npt.NDArray[np.double]],
	densities: list[npt.NDArray[np.cdouble]],
	mass: npt.NDArray[np.double],
	dt: float,
	predictor: collections.abc.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]
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
	predictor : collections.abc.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]
		It predicts the density matrix element based on given coordinates and element index
	"""
	# lower triangular loop, evolve coordinates and density
	evolve.drc = Direction.FORWARD
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
			densities[TrilIndex][...] = evolve_density_non_adiabatically(densities[TrilIndex], x4, p2, x2, p1, mass, dt, predictor, iPES, jPES)
			# finally set up the point coordinates and density
			points[TrilIndex][:, :pes.DIM] = x4
			points[TrilIndex][:, pes.DIM:] = p2
