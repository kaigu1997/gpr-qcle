import pes

import enum
import numpy as np
import numpy.typing as npt
import time
import typing


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
	nac: npt.NDArray[np.double] = pes.non_adiabatic_coupling(x) # ... * D * N * N
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


def get_points(condition: npt.NDArray[np.bool_], *args: npt.NDArray) -> tuple[npt.NDArray, ...]:
	"""
	_summary_

	Parameters
	----------
	condition : npt.NDArray[np.bool_], shape of (...)
		The condition of each point whether each array should be selected or not
	*args : tuple[npt.NDArray, ...], each of shape (..., DIM)
		Quantities at the points

	Returns
	-------
	tuple[npt.NDArray, ...], each of shape (np.count(condition), DIM)
		Quantities that satisfies the condition
	"""
	condition_broadcast: npt.NDArray[np.bool_] = np.repeat(condition[..., np.newaxis], pes.DIM, -1)
	return tuple(x[condition_broadcast].reshape(-1, pes.DIM) for x in args)


def calculate_omega0(
	x0: npt.NDArray[np.double],
	x2: npt.NDArray[np.double],
	drc: Direction,
	RowIndex: int,
	ColIndex: int
) -> npt.NDArray[np.double]:
	if RowIndex == ColIndex:
		return np.zeros(np.broadcast_shapes(x0.shape[:-1], x2.shape[:-1]), np.double)
	else:
		E0: npt.NDArray[np.double] = pes.adiabatic_potential(x0)
		E2: npt.NDArray[np.double] = pes.adiabatic_potential(x2)
		return drc.value * (E0[..., RowIndex] - E0[..., ColIndex] + E2[..., RowIndex] - E2[..., ColIndex]) / 2.0 / pes.HBAR


def evolve_density_non_adiabatically(
	x0: npt.NDArray[np.double],
	p0: npt.NDArray[np.double],
	density: npt.NDArray[np.cdouble] | None,
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
				_summary_

				Parameters
				----------
				rho_part : npt.NDArray[np.cdouble], shape of ((N(N+1)/2) * ...)
					_description_
				x_part : npt.NDArray[np.double], shape of (... * D)
					_description_
				p_part : npt.NDArray[np.double], shape of (... * D)
					_description_
				dt : float
					Time interval
				"""
				phi: npt.NDArray[np.double] = np.sum(p_part / mass * pes.non_adiabatic_coupling(x_part)[..., 0, 1] * is_coupling(x_part, p_part, mass, dt_offdiag).astype(np.double), -1) # v.dot(NAC), ...
				sinphi: npt.NDArray[np.double] = np.sin(2.0 * dt_offdiag * phi)
				cosphi: npt.NDArray[np.double] = np.cos(2.0 * dt_offdiag * phi)
				rho_save: npt.NDArray[np.cdouble] = np.copy(rho_part)
				rho_part[0].real = (1.0 + cosphi) / 2.0 * rho_save[0].real - sinphi * rho_save[1].real + (1.0 - cosphi) / 2.0 * rho_save[2].real
				rho_part[0].imag = 0.0
				rho_part[1].real = sinphi / 2.0 * rho_save[0].real + cosphi * rho_save[1].real - sinphi / 2.0 * rho_save[2].real
				rho_part[1].imag = rho_save[1].imag
				rho_part[2].real = (1.0 - cosphi) / 2.0 * rho_save[0].real + sinphi * rho_save[1].real + (1.0 + cosphi) / 2.0 * rho_save[2].real
				rho_part[2].imag = 0.0
				# rho_part[0] = (1.0 + cosphi) / 2.0 * rho_save[0] - sinphi * rho_save[1].real + (1.0 - cosphi) / 2.0 * rho_save[2]
				# rho_part[1] = sinphi / 2.0 * (rho_save[0] - rho_save[2]) + cosphi * rho_save[1].real + 1.0j * rho_save[1].imag
				# rho_part[2] = (1.0 - cosphi) / 2.0 * rho_save[0] + sinphi * rho_save[1].real + (1.0 + cosphi) / 2.0 * rho_save[2]
			# first step: (x0, p0) -> (x2, p1)
			x2: npt.NDArray[np.double] # ... * D
			p1: npt.NDArray[np.double] # ... * D
			x2, p1 = evolve_coordinates_adiabatically(x0, p0, mass, dt / 2.0, evolve_density_non_adiabatically.drc, RowIndex, ColIndex)
			# second step, off-diagonal branching to p2, and broadcast to x3
			# (backward) direction is included. So when evolve forward, the branch correspondence remains the same
			p2: npt.NDArray[np.double] = p1 + dt * evolve_density_non_adiabatically.offdiagonal_branches.reshape((-1,) + tuple(1 for i in range(IsCouple.ndim))) * (pes.adiabatic_force(x2)[..., 0, 1] * is_coupling(x2, p1, mass, dt).astype(np.double)) # 3 * ... * D
			x3: npt.NDArray[np.double] = x2 + evolve_density_non_adiabatically.drc.value * dt / 4.0 * p2 / mass # 3 * ... * D
			# then adiabatic branching to p3, and broadcast to x4
			f_x3: npt.NDArray[np.double] = pes.adiabatic_force(x3) # 3 * ... * D * N * N
			f_diag_ave_tril: npt.NDArray[np.double] = (f_x3[..., pes.tril_row_indices, pes.tril_row_indices] + f_x3[..., pes.tril_col_indices, pes.tril_col_indices]) / 2.0 # 3 * ... * D * (N(N+1)/2)
			p3: npt.NDArray[np.double] = p2 + evolve_density_non_adiabatically.drc.value * dt / 2.0 * np.moveaxis(f_diag_ave_tril, -1, 0) # (N(N+1)/2) * 3 * ... * D
			x4: npt.NDArray[np.double] = x3 + evolve_density_non_adiabatically.drc.value * dt / 4.0 * p3 / mass # (N(N+1)/2) * 3 * ... * D
			# predict
			rho_predict: npt.NDArray[np.cdouble] = np.empty((pes.NUM_TRIG, 3) + x0.shape[:-1], np.cdouble) # (N(N+1)/2) * 3 * ...
			for index, iElement in enumerate(pes.tril_element_indices):
				branch_row_index: int = iElement // pes.NUM_PES
				branch_col_index: int = iElement % pes.NUM_PES
				start_time: float = time.time()
				rho_predict[index] = predictor(np.concatenate((x4[index], p3[index]), -1), iElement)
				# # assign the known density to it
				if branch_row_index == RowIndex and branch_col_index == ColIndex and density is not None:
					rho_predict[index, evolve_density_non_adiabatically.offdiagonal_zero_branch_index] = density
				end_time = time.time()
				print("\tPrediction costs {} seconds".format(end_time - start_time))
			rho_combined_offdiag: npt.NDArray[np.cdouble] = np.zeros((pes.NUM_TRIG,) + rho_predict.shape[2:], np.cdouble) # (N(N+1)/2) * ...
			for index, branch in enumerate(evolve_density_non_adiabatically.offdiagonal_branches):
				# first half-step adiabatic evolve. (x4, p3) -> (x2, p2) with an adiabatic rotation
				rho_predict[1, index] *= np.exp(dt / 2.0 * 1.0j * calculate_omega0(x2, x4[1, index], Direction.Forward, 0, 1))
				# now they are at (x2, p2). A off-diagonal rotation is needed.
				offdiagonal_rotation(rho_predict[:, index], x2, p2[index], dt / 2.0)
				# then the off-diagonal force evolution combination with a rotation matrix, (x2, p2) -> (x2, p1)
				value: npt.NDArray[np.double]
				match branch:
					case -1:
						rho_combined_offdiag.real += (rho_predict[0, index].real + 2.0 * rho_predict[1, index].real + rho_predict[2, index].real) / 4.0
					case 0:
						value = (rho_predict[0, index].real - rho_predict[2, index].real) / 2.0
						rho_combined_offdiag[0].real += value
						rho_combined_offdiag[1].imag += rho_predict[1, index].imag
						rho_combined_offdiag[2].real -= value
					case 1:
						value = (rho_predict[0, index].real - 2.0 * rho_predict[1, index].real + rho_predict[2, index].real) / 4.0
						rho_combined_offdiag[0].real += value
						rho_combined_offdiag[1].real -= value
						rho_combined_offdiag[2].real += value
				# value: npt.NDArray[np.cdouble]
				# match branch:
				# 	case -1:
				# 		rho_combined_offdiag += (rho_predict[0, index] + 2.0 * rho_predict[1, index].real + rho_predict[2, index]) / 4.0
				# 	case 0:
				# 		value = (rho_predict[0, index] - rho_predict[2, index]) / 2.0
				# 		rho_combined_offdiag[0] += value
				# 		rho_combined_offdiag[1] += 1.0j * rho_predict[1, index].imag
				# 		rho_combined_offdiag[2] -= value
				# 	case 1:
				# 		value = (rho_predict[0, index] - 2.0 * rho_predict[1, index].real + rho_predict[2, index]) / 4.0
				# 		rho_combined_offdiag[0] += value
				# 		rho_combined_offdiag[1] -= value
				# 		rho_combined_offdiag[2] += value
					case _:
						raise ValueError('Unexpected Off-Diagonal Branch!')
			# the other off-diagonal rotation at (x2, p1)
			offdiagonal_rotation(rho_combined_offdiag, x2, p1, dt / 2.0)
			# another adiabatic step, (x2, p1) -> (x0, p0)
			trig_index: int = np.argwhere(pes.tril_element_indices == RowIndex * pes.NUM_PES + ColIndex)[0, 0]
			if RowIndex != ColIndex:
				rho_combined_offdiag[trig_index] *= np.exp(dt / 2.0 * 1.0j * calculate_omega0(x0, x2, Direction.Forward, 0, 1))
			return rho_combined_offdiag[trig_index]
		case _:
			raise NotImplementedError('Model NOT Implemented!')


def evolve(
	points: npt.NDArray[np.double],
	densities: npt.NDArray[np.cdouble],
	mass: npt.NDArray[np.double],
	dt: float,
	predictor: typing.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]
) -> None:
	"""
	_summary_

	Parameters
	----------
	points : npt.NDArray[np.double], shape of (NUM_ELM, NUM_PTS, PHASEDIM)
		_description_
	densities : npt.NDArray[np.double], shape of (NUM_ELM, NUM_PTS)
		_description_
	mass : npt.NDArray[np.double], shape of (DIM,)
		Mass of classical degree of freedom
	dt : float
		Time interval
	predictor : typing.Callable[[npt.NDArray[np.double], int], npt.NDArray[np.cdouble]]
		_description_
	"""
	evolve.drc = Direction.Forward
	# lower triangular loop, evolve coordinates and density
	for iPES in range(pes.NUM_PES):
		for jPES in range(iPES + 1):
			ElementIndex = iPES * pes.NUM_PES + jPES
			# get x and p, and 2 semi adiabatic steps
			x0: npt.NDArray[np.double] = points[ElementIndex, :, :pes.DIM] # M * D
			p0: npt.NDArray[np.double] = points[ElementIndex, :, pes.DIM:] # M * D
			x2: npt.NDArray[np.double] # M * D
			p1: npt.NDArray[np.double] # M * D
			x2, p1 = evolve_coordinates_adiabatically(x0, p0, mass, dt / 2.0, evolve.drc, iPES, jPES)
			x4: npt.NDArray[np.double] # M * D
			p2: npt.NDArray[np.double] # M * D
			x4, p2 = evolve_coordinates_adiabatically(x2, p1, mass, dt / 2.0, evolve.drc, iPES, jPES)
			# judge coupling
			IsCouple: npt.NDArray[np.bool_] = is_coupling(x4, p2, mass, dt) # M * D
			IsCouplePerPoint: npt.NDArray[np.bool_] = np.any(IsCouple, -1) # M
			# evolve adiabatic points
			x0_uncouple: npt.NDArray[np.double] # M' * D
			x2_uncouple: npt.NDArray[np.double] # M' * D
			x4_uncouple: npt.NDArray[np.double] # M' * D
			x0_uncouple, x2_uncouple, x4_uncouple = get_points(np.logical_not(IsCouplePerPoint), x0, x2, x4)
			if iPES != jPES:
				densities[ElementIndex] *= np.exp(-dt / 2.0 * 1.0j * (calculate_omega0(x0_uncouple, x2_uncouple, evolve.drc, iPES, jPES) + calculate_omega0(x2_uncouple, x4_uncouple, evolve.drc, iPES, jPES)))
			# evolve non-adiabatic_points
			x4_couple: npt.NDArray[np.double] # M'' * D
			p2_couple: npt.NDArray[np.double] # M'' * D
			is_couple_couple: npt.NDArray[np.bool_] # M'' * D
			x4_couple, p2_couple, is_couple_couple = get_points(IsCouplePerPoint, x4, p2, IsCouple)
			densities[ElementIndex][IsCouplePerPoint] = evolve_density_non_adiabatically(
				x4_couple,
				p2_couple,
				densities[ElementIndex][IsCouplePerPoint],
				mass,
				is_couple_couple,
				dt,
				predictor,
				iPES,
				jPES
			)
			# then set up the point coordinates and density
			points[ElementIndex, :, :pes.DIM] = x4
			points[ElementIndex, :, pes.DIM:] = p2
			if iPES != jPES:
				# strict upper triangular part, conjugate density and same coordinates
				densities[jPES * pes.NUM_PES + iPES] = np.conj(densities[ElementIndex])
				points[jPES * pes.NUM_PES + iPES] = points[ElementIndex]
