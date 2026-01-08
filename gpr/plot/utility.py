r"""plot.utility
============
The module for some utility functions, such as array parsing.
"""
import collections.abc
import gc
import math
import os
import string
import tarfile
import typing
import warnings

import matplotlib
import matplotlib.axes
import matplotlib.font_manager
import numpy as np
import numpy.typing as npt
import PIL.Image
import torch

import constant

TITLE_PROPERTY: typing.Final[matplotlib.font_manager.FontProperties] = matplotlib.font_manager.FontProperties(size="xx-large", weight="bold")
LEGEND_PROPERTY: typing.Final[dict[str, typing.Any]] = {"frameon": False, "prop": matplotlib.font_manager.FontProperties(size="large", weight="semibold")}
MARKERS: list[str] = ["o", "v", "^", "<", ">", "8", "s", "p", "P", "*", "h", "H", "X", "D", "d"]
LINE_WIDTH: typing.Final = 2.5
I_UB: typing.Final = 1.1
I_LB: typing.Final = -.1


def get_picname(prefix: str, max_outputs: int) -> str:
	r"""To get the figure name template

	Parameters
	----------
	prefix : str
		Type of figure, could be 
	max_outputs : int
		The maximum number of outputs

	Returns
	-------
	str
		The name for output figures, should be used with `.format(frame_index)`
	"""
	return f"{prefix}_{{:0{int(math.ceil(math.log10(max_outputs + 1)))}}}{constant.FIGURE_EXTENSION}"


class FromFile:
	r"""To read data from file

	Parameters
	----------
	filename : str
		Name of the file containing data
	lines_to_read : int
		The number of lines to read each time

	Methods
	-------
	total_ticks()
		To get the total number of outputs
	frame_index()
		To get the number of outputs already read
	have_content()
		Whether there is still content left to read
	get_data()
		To read data from file
	close()
		To close the data file
	"""
	# comment for layout conflict, may uncomment for static check
	# __slots__: typing.ClassVar[tuple] = ("__file", "__lines_to_read", "__n_ticks", "__file_size", "__current_tick")
	__FILESIZE_READLINE: typing.Final = 100 # in unit of GiB
	__file: typing.Final[typing.TextIO]
	__lines_to_read: typing.Final[int]
	__n_ticks: typing.Final[int]
	__file_size: typing.Final[int]
	__current_tick: int

	def __init__(self, filename: str, lines_to_read: int) -> None:
		self.__file = open(filename, "r", encoding=constant.ENC) # pylint: disable=consider-using-with
		self.__lines_to_read = lines_to_read
		self.__file_size = self.__file.seek(0, os.SEEK_END)
		self.__file.seek(0)
		if self.__file_size // 1024 // 1024 // 1024 > FromFile.__FILESIZE_READLINE: # file size > 100GiB
			self.__n_ticks = -1 # do not read #lines
		else: # check the #lines
			n_lines: typing.Final[int] = sum(1 for _ in self.__file)
			assert n_lines % (lines_to_read + 1) == 0
			self.__n_ticks = n_lines // (lines_to_read + 1)
			self.__file.seek(0)
		self.__current_tick = 0

	@property
	@typing.final
	def total_ticks(self) -> int:
		r"""To get the total number of outputs

		Returns
		-------
		int
			The total number of outputs frames from file
		"""
		return self.__n_ticks

	@property
	@typing.final
	def frame_index(self) -> int:
		r"""To get the number of outputs already read

		Returns
		-------
		int
			The number of outputs that has already been read
		"""
		return self.__current_tick

	@property
	@typing.final
	def have_content(self) -> bool:
		r"""Whether there is still content left to read

		Returns
		-------
		bool
			True if there is still content in the file, False otherwise
		"""
		return self.total_ticks != self.frame_index

	@typing.final
	def get_data(self) -> npt.NDArray[np.double]:
		r"""To read data from file

		Returns
		-------
		npt.NDArray[np.double]
			N lines of data, may need further process

		Raises
		------
		EOFError
			In case the file reaches its end but another reading requirement is given
		"""
		with warnings.catch_warnings(action="ignore"):
			try:
				data = np.loadtxt(self.__file, max_rows=self.__lines_to_read)
			except Warning as w:
				raise EOFError("End of File") from w
		self.__file.readline()
		self.__current_tick += 1
		gc.collect()
		return data

	@typing.final
	def close(self) -> None:
		r"""To close the data file
		"""
		return self.__file.close()


def file_read_density_matrix_format(
	pwtdm_from_file: torch.Tensor,
	num_grids_on_each_dimension: list[int],
	num_pes: int
) -> torch.Tensor:
	r"""_summary_

	Parameters
	----------
	pwtdm_from_file : torch.Tensor, dtype of `torch.double`, shape of (NUM_ELM, N_PHASE_GRIDS_TOTAL)
		The PWTDM read from file, include independent part only
	num_grids_on_each_dimension : list[int]
		The number of grids for each classical degree of freedom
	num_pes : int
		The number of potential energy surfaces

	Returns
	-------
	torch.Tensor, dtype of `torch.cdouble`, shape of (N_GRIDS, ..., N_GRIDS, ..., NUM_PES, NUM_PES)
		The formatted PWTDM
	"""
	n_sq: list[int] = num_grids_on_each_dimension * 2
	result: torch.Tensor = torch.empty(n_sq + [num_pes, num_pes], dtype=torch.cdouble)
	for iPES in range(num_pes):
		for jPES in range(iPES):
			result[..., iPES, jPES] = (pwtdm_from_file[jPES * num_pes + iPES] + 1.0j * pwtdm_from_file[iPES * num_pes + jPES]).reshape(n_sq)
			result[..., jPES, iPES] = result[..., iPES, jPES].conj()
		result[..., iPES, iPES] = pwtdm_from_file[iPES * num_pes + iPES].reshape(n_sq)
	return result


def format_array(arr: typing.Any, arr_name : str | None = None, sep: str = " ") -> str:
	r"""To print the flatten array as well as raw number

	Parameters
	----------
	arr : typing.Any
		An array or a raw number
	arr_name : str | None, optional
		The name of the array or variable, by default None
	sep : str, optional
		Separation for array elements, by default " "

	Returns
	-------
	str
		By default, result is simply `"arr_name = " + str(arr)`

		Complex has the form of `f"{arr.real} + {arr.imag}i"`

		Torch tensor and numpy array are flattened and output one by one
		without any other stuff (parenthesis, "array", "Tensor", etc)

		Other iterables (list, tuple, etc) are output one by one without parenthesis too.
	"""
	result: str = (arr_name + " = ") if arr_name else ""
	if isinstance(arr, torch.Tensor) or isinstance(arr, np.ndarray):
		result += sep.join(format_array(val.item()) for val in arr.ravel())
	elif isinstance(arr, collections.abc.Iterable):
		result += sep.join(format_array(item) for item in arr)
	elif isinstance(arr, complex):
		result += f"{arr.real} + {arr.imag}i"
	else:
		result += str(arr)
	return result


def get_element_label(RowIndex: int, ColIndex: int) -> str:
	r"""To have the name of the given element.

	Parameters
	----------
	RowIndex : int
		Row index of the element
	ColIndex:
		Column index of the element

	Returns
	-------
	str
		Name of the element, in bold form
	"""
	return f"$\\mathbf{{\\rho_{{{RowIndex},{ColIndex}}}}}$"


def get_RI_label(ElementIndex: int, NUM_PES: int) -> str:
	r"""To have the real/imaginary part name of the given element.

	Strictly-upper triangular is real, strictly-lower triangular is imaginary, and diagonal elements are as they are.

	Parameters
	----------
	ElementIndex : int
		Index of the element
	NUM_PES : int
		The number of potential energy surfaces

	Returns
	-------
	str
		Name of the real/imaginary part, in bold form
	"""
	RowIndex: typing.Final[int] = ElementIndex // NUM_PES
	ColIndex: typing.Final[int] = ElementIndex % NUM_PES
	if RowIndex == ColIndex:
		return f"$\\mathbf{{\\rho_{{{RowIndex},{ColIndex}}}}}$"
	elif RowIndex < ColIndex:
		return f"$\\mathbf{{\\Re\\rho_{{{ColIndex},{RowIndex}}}}}$"
	else:
		return f"$\\mathbf{{\\Im\\rho_{{{RowIndex},{ColIndex}}}}}$"


def dimension_name(DimIndex: int, DIM: int) -> str:
	r"""Name corresponding to the dimension in phase space

	Parameters
	----------
	DimIndex : int
		Index of the dimension, range in [0, PHASEDIM)
	DIM : int
		The dimension of classical degree of freedom

	Returns
	-------
	str
		The name of the dimension
	"""
	assert 0 <= DimIndex < DIM * 2
	if DIM == 1:
		if DimIndex == 0:
			return "R"
		else:
			return "P"
	else:
		if DimIndex < DIM:
			return f"$\\mathbf{{R_{DimIndex + 1}}}$"
		else:
			return f"$\\mathbf{{P_{DimIndex - DIM + 1}}}$"


def pes_name(PESIndex: int, NUM_PES: int) -> str:
	r"""Name corresponding to the potential energy surface

	Parameters
	----------
	PESIndex : int
		Index of the potential energy surface, range in [0, NUM_PES)
	NUM_PES : int
		The number of potential energy surfaces

	Returns
	-------
	str
		The name of the dimension
	"""
	assert 0 <= PESIndex <= NUM_PES
	if PESIndex == NUM_PES:
		return "Overall"
	elif NUM_PES == 2: # pylint: disable=magic-value-comparison
		return "Lower" if PESIndex == 0 else "Upper"
	else:
		return f"Surface {PESIndex + 1}"


def axes_decoration(
	ax: matplotlib.axes.Axes,
	ax_idx: int,
	*,
	xlabel: str | None = None,
	ylabel: str | None = None,
	title: str | None = None
) -> None:
	r"""To set up general form of axes

	Parameters
	----------
	ax : matplotlib.axes.Axes
		The axes to plot/contour
	ax_idx : int
		The index of the axes
	xlabel : str | None, optional
		Label for x axis, by default None
	ylabel : str | None, optional
		Label for y axis, by default None
	title : str | None, optional
		Title of the axes, by default None
	"""
	FONT_PROPERTY: typing.Final[matplotlib.font_manager.FontProperties] = matplotlib.font_manager.FontProperties(size="xx-large", weight="bold")
	# frame
	for spine in ax.spines.values():
		spine.set_linewidth(2.0)
	# title
	if title is not None:
		ax.set_title(title, fontproperties=FONT_PROPERTY)
	# x axis
	if xlabel is not None:
		ax.set_xlabel(xlabel, fontproperties=FONT_PROPERTY)
	# y axis
	if ylabel is not None:
		ax.set_ylabel(ylabel, fontproperties=FONT_PROPERTY)
	# ticks
	ax.tick_params(axis="both", which="both", direction="in", bottom=True, top=True, left=True, right=True, width=2.0)
	ax.minorticks_on()
	# tick label
	for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
		tick.label1.set_fontweight("semibold")
	# axes label
	ax.text(0.02, 0.98, f"({string.ascii_uppercase[ax_idx]})", fontproperties=FONT_PROPERTY, ha="left", transform=ax.transAxes, va="top")


def draw_animation(picname: str, total_ticks: int, separation: int) -> None:
	r"""To draw the animation from pictures of each tick that already exist

	Parameters
	----------
	picname : str
		Name template for pictures, should be used with `.format(frame_index)`
	total_ticks : int
		The total number of pictures. `frame_index` will be 0 to `total_ticks-1`
	separation : int
		The time interval between each picture, in unit of ms
	"""
	with PIL.Image.open(picname.format(0)) as first_img:
		filename: typing.Final[str] = picname[:picname.rfind("_")] + constant.ANIMATION_EXTENSION
		first_img.save(
			filename,
			save_all=True,
			append_images=[PIL.Image.open(picname.format(iframe)) for iframe in range(1, total_ticks)],
			loop=0,
			duration=separation
		)


def tar_files(picname: str, total_ticks: int) -> None:
	r"""To pack the pictures into a `.tgz` file

	Parameters
	----------
	picname : str
		Name template for pictures, should be used with `.format(frame_index)`
	total_ticks : int
		The total number of pictures. `frame_index` will be 0 to `total_ticks-1`
	"""
	filename: typing.Final[str] = picname[:picname.rfind("_")] + constant.TAR_EXTENSION
	with tarfile.open(filename, "w:gz") as tf:
		for iframe in range(total_ticks):
			tf.add(picname.format(iframe))
	for iframe in range(total_ticks):
		os.remove(picname.format(iframe))
