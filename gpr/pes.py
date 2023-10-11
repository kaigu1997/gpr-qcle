"""
pes
===

This module provides methods for the model, including adiabatic potential energy surfaces,
corresponding Hellmann-Feynmann forces, and non-adiabatic coupling.
"""

import enum
import typing

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

jax.config.update('jax_enable_x64', True)


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
NUM_ELM: typing.Literal[4] = 4
NUM_TRIG: typing.Literal[3] = NUM_PES * (NUM_PES + 1) // 2
tril_row_indices: npt.NDArray[np.int_]
tril_col_indices: npt.NDArray[np.int_]
tril_row_indices, tril_col_indices = np.tril_indices(NUM_PES)
tril_element_indices: npt.NDArray[np.int_] = np.arange(NUM_ELM).reshape(NUM_PES, NUM_PES)[tril_row_indices, tril_col_indices]
DIM: typing.Literal[1] = 1
PHASEDIM: typing.Literal[2] = 2
HBAR: float = 1.0
PURITY_FACTOR: float = (2.0 * np.pi * HBAR) ** DIM


def potential(x: jax.Array) -> jax.Array:
	"""
	To get the potential at give positions

	Parameters
	----------
	x : jax.Array, shape of (..., DIM)
		Positions of interest

	Returns
	-------
	jax.Array, shape of (..., NUM_PES, NUM_PES)
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
	result: jax.Array
	match MODEL:
		case Model.SAC:
			v00: jax.Array = jnp.sign(x[..., 0]) * potential.SAC_A * (1.0 - jnp.exp(-potential.SAC_B * jnp.abs(x[..., 0])))
			v01: jax.Array = potential.SAC_C * jnp.exp(-potential.SAC_D * jnp.square(x[..., 0]))
			result = jnp.array([[v00, v01], [v01, -v00]])
		case Model.DAC:
			v01: jax.Array = potential.DAC_C * jnp.exp(-potential.DAC_D * jnp.square(x[..., 0]))
			result = jnp.array([[jnp.zeros(shape), v01], [v01, potential.DAC_E - potential.DAC_A * jnp.exp(-potential.DAC_B * jnp.square(x[..., 0]))]])
		case Model.ECR:
			v01: jax.Array = potential.ECR_B * (1.0 - jnp.sign(x[..., 0]) * (jnp.exp(-potential.ECR_C * jnp.abs(x[..., 0])) - 1.0))
			result = jnp.array([[jnp.full(shape, potential.ECR_A), v01], [v01, jnp.full(shape, -potential.ECR_A)]])
		case _:
			raise NotImplementedError('Model NOT Implemented!')
	return jnp.moveaxis(result, (0, 1), (-2, -1))


def force(x: jax.Array) -> jax.Array:
	"""
	To get the forces corresponding to the potential at give positions

	Parameters
	----------
	x : jax.Array, shape of (..., DIM)
		Positions of interest

	Returns
	-------
	jax.Array, shape of (..., DIM, NUM_PES, NUM_PES)
		Forces at give positions

	This function is done by autograd from jax. An colwise jacobian is calculated to fulfill the goal.
	"""
	assert x.shape[-1] == DIM
	return -jnp.moveaxis(jax.vmap(jax.jacrev(potential), 0)(x.reshape(-1, DIM)), -1, -3).reshape(x.shape[:-1] + (DIM, NUM_PES, NUM_PES))


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
	return np.asarray(potential(jnp.asarray(x)))


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
	return np.asarray(force(jnp.asarray(x)))


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
			f00: npt.NDArray[np.double] = V[..., 0, 0]
			f01: npt.NDArray[np.double] = V[..., 0, 1]
			f11: npt.NDArray[np.double] = V[..., 1, 1]
			discriminant: npt.NDArray[np.double] = np.sqrt(np.square(f00 - f11) + np.square(2.0 * f01))
			val: npt.NDArray[np.double] = ((f00 + f11)[..., np.newaxis] + np.array([-1.0, 1.0], dtype=np.double) * discriminant[..., np.newaxis]) / 2.0
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
	V: npt.NDArray[np.double] = diabatic_potential(x)
	match NUM_PES:
		case 2:
			f00: npt.NDArray[np.double] = V[..., 0, 0]
			f01: npt.NDArray[np.double] = V[..., 0, 1]
			f11: npt.NDArray[np.double] = V[..., 1, 1]
			discriminant: npt.NDArray[np.double] = np.sqrt(np.square(f00 - f11) + np.square(2.0 * f01))
			vec: npt.NDArray[np.double] = np.moveaxis(np.array([[-discriminant + f00 - f11, discriminant + f00 - f11], [2.0 * f01, 2.0 * f01]]), (0, 1), (-2, -1))
			vec /= np.linalg.norm(vec, axis=-2, keepdims=True)
			return vec
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
	mat: npt.NDArray[np.double] = diabatic_to_adiabatic_matrices(x)[..., np.newaxis, :, :]
	return np.swapaxes(mat, -2, -1) @ diabatic_force(x) @ mat


def non_adiabatic_coupling(x: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
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
