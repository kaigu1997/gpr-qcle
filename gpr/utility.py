"""
utility
=======
The module for some utility functions, such as array parsing.
"""
import collections.abc
import os
import sys
import typing

import numpy as np
import torch

sys.path.append(os.path.dirname(__file__))

import pes


def get_RI_label(ElementIndex: int) -> str:
	"""
	To have the real/imaginary part name of the given input row and column.

	Strictly-upper triangular is real, strictly-lower triangular is imaginary, and diagonal elements are as they are.

	Parameters
	----------
	ElementIndex: int
		Index of the element

	Returns
	-------
	str
		Name of the real/imaginary part. 
	"""
	RowIndex: int = ElementIndex // pes.NUM_PES
	ColIndex: int = ElementIndex % pes.NUM_PES
	if RowIndex == ColIndex:
		return "$\\rho_{{{},{}}}$".format(RowIndex, ColIndex)
	elif RowIndex < ColIndex:
		return "$\\Re\\rho_{{{},{}}}$".format(ColIndex, RowIndex)
	else:
		return "$\\Im\\rho_{{{},{}}}$".format(RowIndex, ColIndex)


def format_array(arr_name : str | None, arr: typing.Any) -> str:
	"""
	To print the flatten array as well as raw number

	Parameters
	----------
	arr_name : str
		The name of the array or variable
	arr : typing.Any
		An array or a raw number

	Returns
	-------
	str
		By default, result is simply `"arr_name = " + str(arr)`

		Complex has the form of `"{} + {}i".format(arr.real, arr.imag)`

		Torch tensor and numpy array are flattened and output one by one
		without any other stuff (parenthesis, "array", "Tensor", etc)

		Other iterables (list, tuple, etc) are output one by one without parenthesis too.
	"""
	result: str = (arr_name + " = ") if arr_name else ""
	if isinstance(arr, torch.Tensor) or isinstance(arr, np.ndarray):
		result += " ".join(format_array(None, val.item()) for val in arr.ravel())
	elif isinstance(arr, collections.abc.Iterable):
		result += " ".join(format_array(None, item) for item in arr)
	elif isinstance(arr, complex):
		result += "{} + {}i".format(arr.real, arr.imag)
	else:
		result += str(arr)
	return result


def dimension_name(DimIndex: int) -> str:
	"""
	Name corresponding to the dimension in phase space

	Parameters
	----------
	DimIndex : int
		Index of the dimension, range in [0, PHASEDIM)

	Returns
	-------
	str
		The name of the dimension
	"""
	assert 0 <= DimIndex < pes.PHASEDIM
	if pes.DIM == 1:
		if DimIndex == 0:
			return "x"
		else:
			return "p"
	else:
		if DimIndex < pes.DIM:
			return r"$x_{}$".format(DimIndex)
		else:
			return r"$p_{}$".format(DimIndex - pes.DIM)
