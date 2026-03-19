r"""evolve
======
This module provides the adiabatic and non-adiabatic evolution scheme.
"""
import enum
import typing

import torch

import constant
import pes

torch.set_default_dtype(constant.DTYPE)
torch.set_default_device(constant.DEVICE)


@typing.final
class Direction(enum.IntEnum):
	r"""Enumerate of directions

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
	model: pes.Potential,
	x0: torch.Tensor,
	p0: torch.Tensor,
	mass: torch.Tensor,
	dt: float,
	drc: Direction,
	RowIndex: int,
	ColIndex: int
) -> tuple[torch.Tensor, torch.Tensor]:
	r"""To evolve the phase space coordinates adiabatically of given element for given interval along given direction

	Parameters
	----------
	model : pes.Potential
		Quantities derived from potential
	x0 : torch.Tensor, shape of (..., DIM)
		Starting positions
	p0 : torch.Tensor, shape of (..., DIM)
		Starting momenta
	mass : torch.Tensor, shape of (DIM,)
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
	tuple[torch.Tensor, torch.Tensor]
		The destination positions and momenta
	"""
	def position_evolve(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
		r"""To evolve positions for half step

		Parameters
		----------
		x : torch.Tensor, shape of (..., DIM)
			Positions
		p : torch.Tensor, shape of (..., DIM)
			Momenta

		Returns
		-------
		torch.Tensor, shape of (..., DIM)
			Positions after half step
		"""
		return x + drc.value * dt / 2.0 * p / mass

	def momentum_diagonal_nonbranch_evolve(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
		r"""To evolve momenta

		Parameters
		----------
		x : torch.Tensor, shape of (..., DIM)
			Positions
		p : torch.Tensor, shape of (..., DIM)
			Momenta

		Returns
		-------
		torch.Tensor, shape of (..., DIM)
			Momenta after evolving the step
		"""
		force: typing.Final[torch.Tensor] = model.force(x) # ... * D * N * N
		return p + drc.value * dt / 2.0 * (force[..., RowIndex, RowIndex] + force[..., ColIndex, ColIndex])

	x1: typing.Final[torch.Tensor] = position_evolve(x0, p0)
	p1: typing.Final[torch.Tensor] = momentum_diagonal_nonbranch_evolve(x1, p0)
	return position_evolve(x1, p1), p1


def evolve_density_adiabatically(
	model: pes.Potential,
	density: torch.Tensor,
	x0: torch.Tensor,
	x2: torch.Tensor,
	x4: torch.Tensor | None,
	drc: Direction,
	dt: float,
	RowIndex: int,
	ColIndex: int
) -> None:
	r"""To evolve the given density matrix element adiabatically

	Parameters
	----------
	model : pes.Potential
		Quantities derived from potential
	density : torch.Tensor, shape of (...)
		The density matrix elements to evolve adiabatically
	x0 : torch.Tensor, shape of (..., DIM)
		The initial positions
	x2 : torch.Tensor, shape of (..., DIM)
		The final positions if x4 does not exist, otherwise the intermediate
	x4 : torch.Tensor, shape of (..., DIM)
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
		E0: typing.Final[torch.Tensor] = model.adiabatic_potential(x0)
		E2: typing.Final[torch.Tensor] = model.adiabatic_potential(x2)
		if x4 is not None:
			E4: typing.Final[torch.Tensor] = model.adiabatic_potential(x4)
			density[...] *= torch.exp(-drc.value * dt / 4.0 / constant.HBAR * 1.0j * (E0[..., RowIndex] - E0[..., ColIndex] + 2.0 * (E2[..., RowIndex] - E2[..., ColIndex]) + E4[..., RowIndex] - E4[..., ColIndex]))
		else:
			density[...] *= torch.exp(-drc.value * dt / 2.0 / constant.HBAR * 1.0j * (E0[..., RowIndex] - E0[..., ColIndex] + E2[..., RowIndex] - E2[..., ColIndex]))


class evolve_density_non_adiabatically:
	r"""To predict the density matrix element at the given phase space coordinates according to non-adiabatic dynamics

	Parameters
	----------
	model : pes.Potential
		Quantities derived from potential
	density : torch.Tensor, shape of (...)
		The density matrix element of given index at give points, or not given.
		Notice all the elements passed to this function is in lower-triangular part.
	x0 : torch.Tensor, shape of (..., DIM)
		The positions to predict density
	p0 : torch.Tensor, shape of (..., DIM)
		The momenta to predict density
	x2 : torch.Tensor, shape of (..., DIM) | None
		The positions of half step before
	p1 : torch.Tensor, shape of (..., DIM) | None
		The momenta of half step before
	mass : torch.Tensor, shape of (DIM,)
		Mass of classical degree of freedom
	dt : float
		Time interval
	predictor : constant.Predictor
		The function that gives the density matrix element at corresponding phase points
	RowIndex : int
		Index of row of the element in density matrix
	ColIndex : int
		Index of column of the element in density matrix

	Returns
	-------
	torch.Tensor, shape of (...)
		The density after back-propagated non-adiabatic evolution

	Raises
	------
	NotImplementedError
		In case the model is unknown
	"""
	drc: typing.Final = Direction.BACKWARD
	offdiagonal_branches: typing.Final = torch.tensor([-1, 0, 1], dtype=torch.int)
	offdiagonal_zero_branch_index: typing.Final = int(torch.argwhere(offdiagonal_branches == 0)[0, 0].item())
	def __new__(
		cls,
		model: pes.Potential,
		density: torch.Tensor,
		x0: torch.Tensor,
		p0: torch.Tensor,
		x2: torch.Tensor | None,
		p1: torch.Tensor | None,
		mass: torch.Tensor,
		dt: float,
		predictor: constant.Predictor,
		RowIndex: int,
		ColIndex: int
	) -> torch.Tensor:
		assert model.config.NUM_PES == 2, "Currently only 2 PES model is supported!"
		def offdiagonal_rotation(rho_part: torch.Tensor, x_part: torch.Tensor, p_part: torch.Tensor, dt_offdiag: float) -> None:
			r"""To have a off-diagonal rotation on the given density matrices

			Parameters
			----------
			rho_part : torch.Tensor, shape of (NUM_TRIG * ...)
				The density matrices
			x_part : torch.Tensor, shape of (... * D)
				The positions corresponding to the density matrices
			p_part : torch.Tensor, shape of (... * D)
				The momenta corresponding to the density matrices
			dt_offdiag : float
				Time interval
			"""
			phi: typing.Final[torch.Tensor] = torch.sum(p_part / mass * model.coupling(x_part)[..., 0, 1], -1) # v.dot(NAC), ...
			sinphi: typing.Final[torch.Tensor] = torch.sin(2.0 * dt_offdiag * phi)
			cosphi: typing.Final[torch.Tensor] = torch.cos(2.0 * dt_offdiag * phi)
			rho_save: typing.Final[torch.Tensor] = rho_part.clone()
			rho_part.real[0] = (1.0 + cosphi) / 2.0 * rho_save[0].real - sinphi * rho_save[1].real + (1.0 - cosphi) / 2.0 * rho_save[2].real
			rho_part.imag[0] = 0.0
			rho_part.real[1] = sinphi / 2.0 * rho_save[0].real + cosphi * rho_save[1].real - sinphi / 2.0 * rho_save[2].real
			rho_part.imag[1] = rho_save[1].imag
			rho_part.real[2] = (1.0 - cosphi) / 2.0 * rho_save[0].real + sinphi * rho_save[1].real + (1.0 + cosphi) / 2.0 * rho_save[2].real
			rho_part.imag[2] = 0.0

		# first step: (x0, p0) -> (x2, p1)
		if x2 is None or p1 is None:
			x2, p1 = evolve_coordinates_adiabatically(model, x0, p0, mass, dt / 2.0, evolve_density_non_adiabatically.drc, RowIndex, ColIndex)
		# second step, off-diagonal branching to p2, and broadcast to x3
		# (backward) direction is included. So when evolve forward, the branch correspondence remains the same
		p2: typing.Final[torch.Tensor] = p1 + dt * evolve_density_non_adiabatically.offdiagonal_branches.reshape((-1,) + tuple(1 for _ in range(x0.ndim))) * model.force(x2)[..., 0, 1] # 3 * ... * D
		x3: typing.Final[torch.Tensor] = x2 + evolve_density_non_adiabatically.drc.value * dt / 4.0 * p2 / mass # 3 * ... * D
		# then adiabatic branching to p3, and broadcast to x4
		f_x3: typing.Final[torch.Tensor] = model.force(x3) # 3 * ... * D * N * N
		f_diag_ave_tril: typing.Final[torch.Tensor] = (f_x3[..., model.config.TRIL_ROW_INDICES, model.config.TRIL_ROW_INDICES] + f_x3[..., model.config.TRIL_COL_INDICES, model.config.TRIL_COL_INDICES]) / 2.0 # 3 * ... * D * NUM_TRIG
		p3: typing.Final[torch.Tensor] = p2 + evolve_density_non_adiabatically.drc.value * dt / 2.0 * torch.moveaxis(f_diag_ave_tril, -1, 0) # NUM_TRIG * 3 * ... * D
		x4: typing.Final[torch.Tensor] = x3 + evolve_density_non_adiabatically.drc.value * dt / 4.0 * p3 / mass # NUM_TRIG * 3 * ... * D
		# predict
		rho_predict: torch.Tensor = torch.empty((model.config.NUM_TRIG, 3) + x0.shape[:-1], dtype=torch.cdouble) # NUM_TRIG * 3 * ...
		for iTrig, (iPES, jPES, iElement) in enumerate(zip(model.config.TRIL_ROW_INDICES, model.config.TRIL_COL_INDICES, model.config.TRIL_ELEMENT_INDICES)):
			rho_predict[iTrig] = predictor(torch.cat((x4[iTrig], p3[iTrig]), -1), iElement)
			# assign the known density to it
			if iPES == RowIndex and jPES == ColIndex and density is not None:
				rho_predict[iTrig, evolve_density_non_adiabatically.offdiagonal_zero_branch_index] = density
			# first half-step adiabatic evolve. (x4, p3) -> (x2, p2) with an adiabatic rotation
			evolve_density_adiabatically(model, rho_predict[iTrig], x2, x4[iTrig], None, Direction.FORWARD, dt / 2.0, iPES, jPES)
		rho_combined_offdiag: torch.Tensor = torch.zeros((model.config.NUM_TRIG,) + rho_predict.shape[2:], dtype=torch.cdouble) # NUM_TRIG * ...
		for index, branch in enumerate(evolve_density_non_adiabatically.offdiagonal_branches):
			# now they are at (x2, p2). A off-diagonal rotation is needed.
			offdiagonal_rotation(rho_predict[:, index], x2, p2[index], dt / 2.0)
			# then the off-diagonal force evolution combination with a rotation matrix, (x2, p2) -> (x2, p1)
			value: torch.Tensor
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
		trig_index: typing.Final[int] = model.config.FLATTEN_TRIL_INDEX[RowIndex * model.config.NUM_PES + ColIndex]
		evolve_density_adiabatically(model, rho_combined_offdiag[trig_index], x0, x2, None, Direction.FORWARD, dt / 2.0, RowIndex, ColIndex)
		return rho_combined_offdiag[trig_index]


def evolve(
	model: pes.Potential,
	points: list[torch.Tensor],
	densities: list[torch.Tensor] | None,
	mass: torch.Tensor,
	dt: float,
	predictor: constant.Predictor
) -> None:
	r"""To evolve the given points and density matrices

	Parameters
	----------
	model : pes.Potential
		Quantities derived from potential
	points : list[torch.Tensor], len of NUM_TRIG, each of shape (NUM_PTS, PHASEDIM)
		Phase space coordinates of selected points for each density matrix element
	densities : list[torch.Tensor], len of NUM_TRIG, each of shape (NUM_PTS,) | None
		Density matrix element of the points
	mass : torch.Tensor, shape of (DIM,)
		Mass of classical degree of freedom
	dt : float
		Time interval
	predictor : model.Predictor
		It predicts the density matrix element based on given coordinates and element index
	"""
	# lower triangular loop, evolve coordinates and density
	evolve.drc = Direction.FORWARD
	for iPES, jPES, iTrig in zip(model.config.TRIL_ROW_INDICES, model.config.TRIL_COL_INDICES, model.config.TRIG_RANGE):
		# get x and p, and 2 semi adiabatic steps
		x0: torch.Tensor = points[iTrig][:, :model.config.DIM] # M * D
		p0: torch.Tensor = points[iTrig][:, model.config.DIM:] # M * D
		x2: torch.Tensor # M * D
		p1: torch.Tensor # M * D
		x2, p1 = evolve_coordinates_adiabatically(model, x0, p0, mass, dt / 2.0, evolve.drc, iPES, jPES)
		x4: torch.Tensor # M * D
		p2: torch.Tensor # M * D
		x4, p2 = evolve_coordinates_adiabatically(model, x2, p1, mass, dt / 2.0, evolve.drc, iPES, jPES)
		if densities is not None:
			densities[iTrig][...] = evolve_density_non_adiabatically(model, densities[iTrig], x4, p2, x2, p1, mass, dt, predictor, iPES, jPES)
		# finally set up the point coordinates and density
		points[iTrig][:, :model.config.DIM] = x4
		points[iTrig][:, model.config.DIM:] = p2
