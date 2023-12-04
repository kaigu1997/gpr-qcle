"""
pes
===

This module provides methods for the model, including adiabatic potential energy surfaces,
corresponding Hellmann-Feynmann forces, and non-adiabatic coupling.
"""

import enum
import typing

import numpy as np
import numpy.typing as npt
import torch

torch.set_default_dtype(torch.double)


class Model(enum.IntEnum):
	"""
	Enumerate of known models

	Attributes
	----------
	SAC : typing.Literal[Model.SAC]
		Single Avoided Crossing, John Tully's first model
	DAC : typing.Literal[Model.DAC]
		Dual Avoided Crossing, John Tully's second model
	ECR : typing.Literal[Model.ECR]
		Extended Coupling with Reflection, John Tully's third model
	"""
	SAC = enum.auto()
	DAC = enum.auto()
	ECR = enum.auto()


MODEL: typing.Literal[Model.DAC] = Model.DAC
NUM_PES: typing.Literal[2] = 2
NUM_ELM: typing.Literal[4] = NUM_PES * NUM_PES
NUM_TRIG: typing.Literal[3] = NUM_PES * (NUM_PES + 1) // 2
DIM: typing.Literal[1] = 1
PHASEDIM: typing.Literal[2] = 2
HBAR: float = 1.0
tril_row_indices: npt.NDArray[np.int_]
tril_col_indices: npt.NDArray[np.int_]
tril_row_indices, tril_col_indices = np.tril_indices(NUM_PES)
tril_element_indices: npt.NDArray[np.int_] = tril_row_indices * NUM_PES + tril_col_indices


def lower_triangular_to_full(tril_part: np.ndarray) -> np.ndarray:
	"""
	To turn the lower triangular part of the matrix into the full matrix

	Parameters
	----------
	tril_part : np.ndarray
		The array of lower triangular part.
		If it is 1d, it corresponds to linearized lower triangular part (i.e., without strictly upper part).
		Otherwise, take the lower triangular part of last two dimension and flip them up.

	Returns
	-------
	np.ndarray
		Indices range [0, NUM_TRIG)
	"""
	mat_size: int
	if tril_part.ndim == 1:
		mat_size = int(np.sqrt(2 * tril_part.size))
		assert tril_part.size == mat_size * (mat_size + 1) // 2
		result: np.ndarray = np.zeros((mat_size, mat_size), tril_part.dtype)
		result[tril_row_indices, tril_col_indices] = tril_part
		return result + np.tril(result, -1).T.conj()
	else:
		assert tril_part.shape[-1] == tril_part.shape[-2]
		mat_size = tril_part.shape[-1]
		return np.tril(tril_part) + np.tril(tril_part, -1).T.conj()


flatten_tril_index: npt.NDArray[np.int_] = lower_triangular_to_full(np.arange(NUM_TRIG, dtype=np.int_))


def potential(x: torch.Tensor) -> torch.Tensor:
	"""
	To get the potential at give positions

	Parameters
	----------
	x : torch.Tensor, shape of (..., DIM)
		Positions of interest

	Returns
	-------
	torch.Tensor, shape of (..., NUM_PES, NUM_PES)
		Potential at give positions

	Raises
	------
	NotImplementedError
		In case the model is unknown
	"""
	potential.SAC_A = 0.01
	potential.SAC_B = 1.6
	potential.SAC_C = 0.005
	potential.SAC_D = 1.0
	potential.DAC_A = 0.10
	potential.DAC_B = 0.28
	potential.DAC_C = 0.015
	potential.DAC_D = 0.06
	potential.DAC_E = 0.05
	potential.ECR_A = 6e-4
	potential.ECR_B = 0.10
	potential.ECR_C = 0.90
	assert x.shape[-1] == DIM
	shape: tuple = x.shape[:-1] if x.ndim > 1 else ()
	result: torch.Tensor
	match MODEL:
		case Model.SAC:
			v00: torch.Tensor = torch.sign(x[..., 0]) * potential.SAC_A * (1.0 - torch.exp(-potential.SAC_B * torch.abs(x[..., 0])))
			v01: torch.Tensor = potential.SAC_C * torch.exp(-potential.SAC_D * torch.square(x[..., 0]))
			result = torch.stack([v00, v01, v01, -v00], -1)
		case Model.DAC:
			v01: torch.Tensor = potential.DAC_C * torch.exp(-potential.DAC_D * torch.square(x[..., 0]))
			result = torch.stack([torch.zeros(shape), v01, v01, potential.DAC_E - potential.DAC_A * torch.exp(-potential.DAC_B * torch.square(x[..., 0]))], -1)
		case Model.ECR:
			v01: torch.Tensor = potential.ECR_B * (1.0 - torch.sign(x[..., 0]) * (torch.exp(-potential.ECR_C * torch.abs(x[..., 0])) - 1.0))
			result = torch.Tensor([torch.full(shape, potential.ECR_A), v01, v01, torch.full(shape, -potential.ECR_A)], -1)
		case _:
			raise NotImplementedError('Model NOT Implemented!')
	return torch.reshape(result, shape + (NUM_PES, NUM_PES))


def force(x: torch.Tensor) -> torch.Tensor:
	"""
	To get the forces corresponding to the potential at give positions

	Parameters
	----------
	x : torch.Tensor, shape of (..., DIM)
		Positions of interest

	Returns
	-------
	torch.Tensor, shape of (..., DIM, NUM_PES, NUM_PES)
		Forces at give positions

	This function is done by autograd from torch. An colwise jacobian is calculated to fulfill the goal.
	"""
	assert x.shape[-1] == DIM
	return -torch.moveaxis(torch.func.vmap(torch.func.jacrev(potential))(x.reshape(-1, DIM)), -1, -3).reshape(x.shape[:-1] + (DIM, NUM_PES, NUM_PES))


def diabatic_potential(x: npt.NDArray[np.double]) -> npt.NDArray:
	"""
	To get the diabatic potential at give positions

	Parameters
	----------
	x : npt.NDArray[np.double], shape of (..., DIM)
		Positions of interest

	Returns
	-------
	npt.NDArray, shape of (..., NUM_PES, NUM_PES)
		Diabatic potential at give positions

	See Also
	-------
	potential: the source function. This is a wrapper of it
	"""
	return np.asarray(potential(torch.from_numpy(x)))


def diabatic_force(x: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
	"""
	To get the diabatic force at give positions

	Parameters
	----------
	x : npt.NDArray[np.double], shape of (..., DIM)
		Positions of interest

	Returns
	-------
	npt.NDArray[np.double], shape of (..., DIM, NUM_PES, NUM_PES)
		Diabatic force at give positions

	See Also
	-------
	force: the source function. This is a wrapper of it
	"""
	return np.asarray(force(torch.from_numpy(x)))


def adiabatic_potential(x: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
	"""
	To get the adiabatic potential at give positions by diagonalization

	Parameters
	----------
	x : npt.NDArray[np.double], shape of (..., DIM)
		Positions of interest

	Returns
	-------
	npt.NDArray[np.double], shape of (..., NUM_PES)
		Diagonal elements of adiabatic potential at give positions

	Raises
	------
	NotImplementedError
		In case the model is unknown
	"""
	V: npt.NDArray[np.double] = diabatic_potential(x)
	match NUM_PES:
		case 2:
			V00: npt.NDArray[np.double] = V[..., 0, 0] # ...
			V01: npt.NDArray[np.double] = V[..., 0, 1] # ...
			V11: npt.NDArray[np.double] = V[..., 1, 1] # ...
			discriminant: npt.NDArray[np.double] = np.sqrt(np.square(V00 - V11) + np.square(2.0 * V01)) # ...
			val: npt.NDArray[np.double] = ((V00 + V11)[..., np.newaxis] + np.array([-1.0, 1.0], dtype=np.double) * discriminant[..., np.newaxis]) / 2.0 # ... * N
			return val
		case _:
			raise NotImplementedError('Model NOT Implemented!')


def diabatic_to_adiabatic_matrices(x: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
	"""
	To get the basis transformation matrices at given positions

	Parameters
	----------
	x : npt.NDArray[np.double], shape of (..., DIM)
		Positions of interest

	Returns
	-------
	npt.NDArray[np.double], shape of (..., NUM_PES, NUM_PES)
		Basis transformation matrices at given positions

	Raises
	------
	NotImplementedError
		In case the model is unknown
	"""
	V: npt.NDArray[np.double] = diabatic_potential(x) # ... * N * N
	match NUM_PES:
		case 2:
			V00: npt.NDArray[np.double] = V[..., 0, 0] # ...
			V01: npt.NDArray[np.double] = V[..., 0, 1] # ...
			V11: npt.NDArray[np.double] = V[..., 1, 1] # ...
			discriminant: npt.NDArray[np.double] = np.sqrt(np.square(V00 - V11) + np.square(2.0 * V01)) # ...
			vec: npt.NDArray[np.double] = np.moveaxis(np.array([[-discriminant + V00 - V11, discriminant + V00 - V11], [2.0 * V01, 2.0 * V01]]), (0, 1), (-2, -1)) # ... * N * N
			diag_indices: npt.NDArray[np.int_] = np.argwhere(V01 == 0)
			if diag_indices.size != 0:
				vec[tuple(diag_indices.T)] = np.where((V00[tuple(diag_indices.T)] < V11[tuple(diag_indices.T)])[..., np.newaxis, np.newaxis], np.eye(2, dtype=np.double), 1.0 - np.eye(2, dtype=np.double))
			return vec / np.linalg.norm(vec, axis=-2, keepdims=True)
		case _:
			raise NotImplementedError('Model NOT Implemented!')


def adiabatic_force(x: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
	"""
	To get the adiabatic force at give positions

	Parameters
	----------
	x : npt.NDArray[np.double], shape of (..., DIM)
		Positions of interest

	Returns
	-------
	npt.NDArray[np.double], shape of (..., DIM, NUM_PES, NUM_PES)
		Adiabatic force at give positions

	The adiabatic force is calculated by basis transformation of all positions and all dimensions
	"""
	mat: npt.NDArray[np.double] = diabatic_to_adiabatic_matrices(x)[..., np.newaxis, :, :] # ... * 1 * N * N
	return np.swapaxes(mat, -2, -1) @ diabatic_force(x) @ mat


def adiabatic_coupling(x: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
	"""
	To get the non-adiabatic coupling at give positions

	Parameters
	----------
	x : npt.NDArray[np.double], shape of (..., DIM)
		Positions of interest

	Returns
	-------
	npt.NDArray[np.double], shape of (..., DIM, NUM_PES, NUM_PES)
		Non-adiabatic coupling at give positions

	The non-adiabatic coupling is calculated by
	.. math::
		d_{ij}=\\begin{cases}0,\\mathrm{if}i=j\\\\F_{ij}/E_i-E_j,\\mathrm{otherwise}\\end{cases}
	"""
	E: npt.NDArray[np.double] = adiabatic_potential(x) # ... * N
	F: npt.NDArray[np.double] = adiabatic_force(x) # ... * D * N * N
	result: npt.NDArray[np.double] = np.zeros(F.shape, np.double)
	# assign the strict lower part
	strict_tril_row: npt.NDArray[np.int_]
	strict_tril_col: npt.NDArray[np.int_]
	strict_tril_row, strict_tril_col = np.tril_indices(NUM_PES, -1)
	result[..., strict_tril_row, strict_tril_col] = F[..., strict_tril_row, strict_tril_col] / (E[..., np.newaxis, strict_tril_row] - E[..., np.newaxis, strict_tril_col])
	return result - result.swapaxes(-2, -1)


def tensor_slice(tensor: npt.NDArray[np.double], dim: int | None, row_index: int, col_index: int) -> npt.NDArray[np.double]:
	"""
	To slice a tensor based on the given dimension

	Parameters
	----------
	tensor : npt.NDArray[np.double], shape of (..., DIM, NUM_PES, NUM_PES)
		The tensor, generally the force
	dim : int | None
		Dimension of interest. If a number, return the force of the given dimension, otherwise returns the forces of all dimensions
	row_index : int
		Index of second last rank
	col_index : int
		Index of last rank

	Returns
	-------
	npt.NDArray[np.double], shape of (...) or (..., DIM)
		Slice
	"""
	if dim is None:
		return tensor[..., row_index, col_index]
	else:
		return tensor[..., dim, row_index, col_index]


def force_basis_force(x: npt.NDArray[np.double], dim: int | None = None) -> npt.NDArray[np.double]:
	"""
	To give the (diagonalized) force under the "force basis"

	Parameters
	----------
	x : npt.NDArray[np.double], shape of (..., DIM)
		Positions of interest
	dim : int | None, optional
		Dimension of interest, by default None
		If given, return the force of the given dimension, otherwise returns the forces of all dimensions

	Returns
	-------
	npt.NDArray[np.double], shape of (..., NUM_PES) or (..., DIM, NUM_PES)
		Diagonalized forces

	Raises
	------
	NotImplementedError
		In case the model is unknown
	"""
	assert dim is None or 0 <= dim < DIM
	force_dia: npt.NDArray[np.double] = diabatic_force(x) # ... * D * N * N
	match NUM_PES:
		case 2:
			f00: npt.NDArray[np.double] = tensor_slice(force_dia, dim, 0, 0) # ...( * D)
			f01: npt.NDArray[np.double] = tensor_slice(force_dia, dim, 0, 1) # ...( * D)
			f11: npt.NDArray[np.double] = tensor_slice(force_dia, dim, 1, 1) # ...( * D)
			discriminant: npt.NDArray[np.double] = np.sqrt(np.square(f00 - f11) + np.square(2.0 * f01)) # ...( * D)
			val: npt.NDArray[np.double] = ((f00 + f11)[..., np.newaxis] + np.array([-1.0, 1.0], dtype=np.double) * discriminant[..., np.newaxis]) / 2.0 # ...( * D) * N
			return val
		case _:
			raise NotImplementedError('Model NOT Implemented!')


def diabatic_to_force_basis(x: npt.NDArray[np.double], dim: int | None = None) -> npt.NDArray[np.double]:
	"""
	To get the basis transformation matrices at given positions of given dimensions

	Parameters
	----------
	x : npt.NDArray[np.double], shape of (..., DIM)
		Positions of interest
	dim : int | None, optional
		Dimension of interest, by default None
		If given, return the force of the given dimension, otherwise returns the forces of all dimensions

	Returns
	-------
	npt.NDArray[np.double], shape of (..., NUM_PES, NUM_PES) or (..., DIM, NUM_PES, NUM_PES)
		Basis transformation matrices at given positions of given dimensions

	Raises
	------
	NotImplementedError
		In case the model is unknown
	"""
	assert dim is None or 0 <= dim < DIM
	force_dia: npt.NDArray[np.double] = diabatic_force(x) # ... * D * N * N
	match NUM_PES:
		case 2:
			f00: npt.NDArray[np.double] = tensor_slice(force_dia, dim, 0, 0) # ...( * D)
			f01: npt.NDArray[np.double] = tensor_slice(force_dia, dim, 0, 1) # ...( * D)
			f11: npt.NDArray[np.double] = tensor_slice(force_dia, dim, 1, 1) # ...( * D)
			discriminant: npt.NDArray[np.double] = np.sqrt(np.square(f00 - f11) + np.square(2.0 * f01)) # ...( * D)
			vec: npt.NDArray[np.double] = np.moveaxis(np.array([[-discriminant + f00 - f11, discriminant + f00 - f11], [2.0 * f01, 2.0 * f01]]), (0, 1), (-2, -1)) # ...( * D) * N * N
			diag_indices: npt.NDArray[np.int_] = np.argwhere(f01 == 0)
			if diag_indices.size != 0:
				vec[tuple(diag_indices.T)] = np.where((f00[tuple(diag_indices.T)] < f11[tuple(diag_indices.T)])[..., np.newaxis, np.newaxis], np.eye(2, dtype=np.double), 1.0 - np.eye(2, dtype=np.double))
			vec /= np.linalg.norm(vec, axis=-2, keepdims=True)
			return vec
		case _:
			raise NotImplementedError('Model NOT Implemented!')


def force_basis_potential(x: npt.NDArray[np.double], dim: int | None = None) -> npt.NDArray[np.double]:
	"""
	To get the potential under force basis at give positions of given dimensions

	Parameters
	----------
	x : npt.NDArray[np.double]
		Positions of interest
	dim : int | None, optional
		Dimension of interest, by default None
		If given, return the force of the given dimension, otherwise returns the forces of all dimensions

	Returns
	-------
	npt.NDArray[np.double]
		Potential under force basis at give positions of given dimensions

	The potential under force basis is calculated by basis transformation of all positions of given dimensions
	"""
	mat: npt.NDArray[np.double] = diabatic_to_force_basis(x, dim) # ...( * D) * N * N
	Vd: npt.NDArray[np.double] = diabatic_potential(x)
	if dim is None:
		Vd = Vd[..., np.newaxis, :, :] # ...( * 1) * N * N
	return np.swapaxes(mat, -2, -1) @ diabatic_potential(x) @ mat
