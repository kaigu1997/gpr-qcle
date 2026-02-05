r"""plot.dm
=======
Implementation of drawing density matrices
"""
import collections.abc
import functools
import math
import sys
import typing

import matplotlib
import matplotlib.axes
import matplotlib.cm
import matplotlib.colorbar
import matplotlib.colors
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import numpy.typing as npt
import torch

import constant
import param
import pes

from . import utility, wfn

torch.set_default_dtype(constant.DTYPE)

DIM_PLOT_DM: typing.Final = 2


class DensityMatrixDrawer:
	r"""To draw the density matrix.

	This could only be used if the model is 1D (so that the phase space is 2D).

	Parameters
	----------
	quantity : param.Quantity
		Quantities including grids, output intervals, etc
	draw_rescaled : bool
		True to make the maximum value of each element to the same value;
		False to keep the original value
	draw_logscale : bool
		True to draw a logscale plot of the marginal as well
		False not to draw it (and leave)
	draw_scattered : bool, optional
		Whether to scatter sample points or not, by default False
	max_outputs : int
		Estimation of maximum outputs
	initial_dm : torch.Tensor, dtype of `torch.cdouble`, shape of (N_GRID, ..., N_GRID, ..., NUM_PES, NUM_PES)
		Initial adiabatic PWTDM from GPR
	grid_solution_filename : str, optional
		The file containing grid solution, by default "" (meaning no such file)
	initial_points : collections.abc.Sequence[npt.NDArray[np.double]] | None, len of 2 * NUM_TRIG, each of shape (NUM_PTS, PHASEDIM), optional
		The points to scatter. Only used if `self.__draw_scattered` is True.
		By default None
	initial_scale : npt.NDArray[np.double], shape of (NUM_ELM,)
		The rescale factor, by default None (grid density will be used to calculate it instead)

	Attributes
	----------
	config : pes.ModelConfig
		Configuration of the model

	Methods
	-------
	picname()
		To get the naming template for pictures
	"""
	__CMAP: typing.Final[matplotlib.colors.Colormap] = matplotlib.colormaps["seismic"]
	__CEN_PT_SIZE: typing.Final = 0.5
	__XTR_PT_SIZE: typing.Final = 0.1
	__CEN_PT_COLOR: typing.Final = "black"
	__XTR_PT_COLOR: typing.Final = "green"

	@typing.final
	class __PNLogNorm(matplotlib.colors.Normalize):
		r"""Log scale for positive/negative values, e.g. (-1, -0.1, -0.01, 0, 0.01, 0.1, 1.0)

		Parameters
		----------
		abs_min : float, optional
			The absolute value minimum of the given dataset, by default sys.float_info.min
		abs_max : float, optional
			The absolute value maximum of the given dataset, by default sys.float_info.max
		clip : bool, optional
			Determines the behavior for mapping values outside the range, by default False

		Methods
		-------
		abs_min()
			The absolute value minimum that has color on the colorbar
		abs_max()
			The absolute value maxmimum that has color on the colorbar
		inverse(value)
			To map a value generally in [0, 1] back to [`-abs_max`, `abs_max`]
		"""
		__slots__: typing.Final[tuple] = ("__abs_min", "__abs_max", "__log2_abs_min", "__log2_abs_max", "__log2_diff")
		__abs_min: float
		__abs_max: float
		__log2_abs_min: float
		__log2_abs_max: float
		__log2_diff: float

		def __init__(self, abs_min: float = sys.float_info.min, abs_max: float = sys.float_info.max, clip: bool = False) -> None:
			assert 0 < abs_min < abs_max
			super().__init__(-abs_max, abs_max, clip)
			self.__abs_min = abs_min
			self.__abs_max = abs_max
			self.__log2_abs_min = math.log2(abs_min)
			self.__log2_abs_max = math.log2(abs_max)
			self.__log2_diff = self.__log2_abs_max - self.__log2_abs_min

		@property
		def abs_min(self) -> float:
			r"""The absolute value minimum that has color on the colorbar

			Returns
			-------
			float
				Min absolute value
			"""
			return self.__abs_min

		@abs_min.setter
		def abs_min(self, val: float) -> float:
			r"""To set the minimum value appeared on colorbar

			Parameters
			----------
			val : float
				The value

			Returns
			-------
			float
				Same as `val`
			"""
			self.__abs_min = abs(val)
			self.__log2_abs_min = math.log2(self.__abs_min)
			self.__log2_diff = self.__log2_abs_max - self.__log2_abs_min
			self._changed() # pyright: ignore[reportAttributeAccessIssue]
			return val

		@property
		def abs_max(self) -> float:
			r"""The absolute value maxmimum that has color on the colorbar

			Returns
			-------
			float
				Max absolute value
			"""
			return self.__abs_max

		@abs_max.setter
		def abs_max(self, val: float) -> float:
			r"""To set the maxmimum value appeared on colorbar

			Parameters
			----------
			val : float
				The value

			Returns
			-------
			float
				Same as `val`
			"""
			self.__abs_max = abs(val)
			self.__log2_abs_max = math.log2(self.__abs_max)
			self.__log2_diff = self.__log2_abs_max - self.__log2_abs_min
			self._changed() # pyright: ignore[reportAttributeAccessIssue]
			return val

		def __repr__(self) -> str:
			r"""The official string representation of an object

			Returns
			-------
			str
				The name of the class with min and max
			"""
			return f"{__class__.__name__}({self.__abs_min}, {self.__abs_max})"

		def __call__(self, value: typing.Any, clip: bool | None = None) -> float | npt.NDArray[np.double]:
			r"""Mapping the given values within [`-abs_max`, `abs_max`] to [0, 1]

			Values outside may map outside of [0, 1] based on `clip` parameter

			Parameters
			----------
			value : typing.Any
				Any value that could be regarded as `np.double` or `np.ndarray` of `np.double`
			clip : bool | None, optional
				Determines the behavior for mapping values outside the range, overwrite object behavior, by default None

			Returns
			-------
			float | npt.NDArray[np.double]
				Mapping result corresponding to value
			"""
			def process_single_value(value: float, clip: bool) -> float:
				r"""To map a single value.

				`value` > `abs_min` will be mapped to (0.5, `inf`) or (0.5, 1] if clipped

				`value` < `-abs_min` will be mapped to (`-inf`, 0.5) or [0, 0.5) if clipped

				`-abs_min` <= `value` <= `abs_min` will be mapped to 0.5

				Parameters
				----------
				value : float
					A value to be mapped
				clip : bool
					Determines the behavior for mapping values outside the range

				Returns
				-------
				float
					Mapping result to (`-inf`, `inf`) or [0, 1] if clipped
				"""
				if abs(value) <= self.__abs_min:
					return 0.5
				sgn_half: typing.Final[float] = 0.5 if value > 0 else -0.5
				if clip and abs(value) >= self.__abs_max:
					return sgn_half + 0.5
				else:
					return (math.log2(abs(value)) - self.__log2_abs_min) / self.__log2_diff * sgn_half + 0.5

			if clip is None:
				clip = self.clip
			result, is_scalar = super().process_value(value)
			if is_scalar == 1:
				return process_single_value(result.data.ravel()[0], clip)
			else:
				return np.ma.array(np.vectorize(process_single_value, otypes=[float])(result.data, clip), mask=result.mask)

		def inverse(self, value: typing.Any) -> float | npt.NDArray[np.double]:
			r"""To map a value generally in [0, 1] back to [`-abs_max`, `abs_max`]

			Parameters
			----------
			value : typing.Any
				Any value that could be regarded as `np.double` or `np.ndarray` of `np.double`

			Returns
			-------
			float | npt.NDArray[np.double]
				Mapping back result corresponding to value
			"""
			def process_single_value(value: float) -> float:
				# pylint: disable=magic-value-comparison
				r"""Maps a single value back

				`value` == 0.5 will be mapped back to 0

				`value` > 0.5 will be mapped back to positive values

				`value` < 0.5 will be mapped back to negative values

				Parameters
				----------
				value : float
					A value, generally in [0, 1]

				Returns
				-------
				float
					Generally in [`-abs_max`, `-abs_min`] U {0} U [`abs_min`, `abs_max`]
				"""
				if value == 0.5:
					return 0.0
				return (-1.0 if value < 0.5 else 1.0) * math.exp2(abs(value - 0.5) * 2 * self.__log2_diff + self.__log2_abs_min)

			result, is_scalar = super().process_value(value)
			if is_scalar == 1:
				return process_single_value(result.data.ravel()[0])
			else:
				return np.ma.array(np.vectorize(process_single_value, otypes=[float])(result.data), mask=result.mask)

	@staticmethod
	def __get_centered(
		abs_max: float,
		/,
		ctr_norm: matplotlib.colors.CenteredNorm | None = None,
		ctr_level: npt.NDArray[np.double] | None = None,
	) -> tuple[matplotlib.colors.CenteredNorm, npt.NDArray[np.double], npt.NDArray[np.double]]:
		r"""To get the `CenteredNorm` from given input, and its corresponding levels and colorbar ticks

		Parameters
		----------
		abs_max : float
			The range of data values
		ctr_norm: matplotlib.colors.CenteredNorm | None, optional
			The existing CenteredNorm, by default None
		ctr_level: npt.NDArray[np.double] | None, optional
			The existing levels from centered norm, by default None

		Returns
		-------
		tuple[matplotlib.colors.CenteredNorm, npt.NDArray[np.double], npt.NDArray[np.double]]
			Norm, levels, and ticks
		"""
		abs_max = abs(abs_max)
		abs_max_10_level: typing.Final[float] = math.pow(10, math.floor(math.log10(abs_max)))
		centered_norm: matplotlib.colors.CenteredNorm
		if ctr_norm is None:
			centered_norm = matplotlib.colors.CenteredNorm(0.0, math.ceil(abs_max / abs_max_10_level) * abs_max_10_level, True)
		else:
			ctr_norm.halfrange = math.ceil(abs_max / abs_max_10_level) * abs_max_10_level
			centered_norm = ctr_norm
		centered_levels: typing.Final[npt.NDArray[np.double]] = np.linspace(-centered_norm.halfrange, centered_norm.halfrange, DensityMatrixDrawer.__CMAP.N // 8 * 8 + 1, True)
		if ctr_level is not None:
			ctr_level[...] = centered_levels
		return centered_norm, centered_levels, centered_levels[::DensityMatrixDrawer.__CMAP.N // 8]

	@staticmethod
	def __get_posneg_log(
		abs_min: float,
		abs_max: float,
		/,
		log_norm: __PNLogNorm | None = None,
		log_level: npt.NDArray[np.double] | None = None,
	) -> tuple[__PNLogNorm, npt.NDArray[np.double], npt.NDArray[np.double]]:
		r"""To get the `PNLogNorm` from given input, and its corresponding levels and colorbar ticks

		Parameters
		----------
		abs_min : float
			The absolute value minimum of the given dataset
		abs_max : float
			The absolute value maximum of the given dataset
		log_norm: __PNLogNorm | None, optional
			The existing pos-neg log norm, by default None
		log_level: npt.NDArray[np.double] | None, optional
			The existing levels from pos-net log norm, by default None

		Returns
		-------
		tuple[PNLogNorm, npt.NDArray[np.double], npt.NDArray[np.double]]
				Norm, levels, and ticks
		"""
		abs_min = abs(abs_min)
		abs_max = abs(abs_max)
		if abs_max == 0: # pylint: disable=consider-using-assignment-expr
			abs_max = sys.float_info.min * 2
		if abs_min == 0:
			abs_min = sys.float_info.min
		abs_min = max(abs_min, abs_max * sys.float_info.epsilon)
		lb_log10: int = int(math.floor(math.log10(abs_min)))
		ub_log10: typing.Final[int] = int(math.ceil(math.log10(abs_max)))
		if ub_log10 - lb_log10 > 4: # pylint: disable=magic-value-comparison
			lb_log10 += (ub_log10 - lb_log10) % 4
		num_ticks: typing.Final[int] = DensityMatrixDrawer.__CMAP.N // 8 * 8 + 3
		pn_log_norm: DensityMatrixDrawer.__PNLogNorm
		if log_norm is not None:
			log_norm.abs_min = math.pow(10, lb_log10)
			log_norm.abs_max = math.pow(10, ub_log10)
			pn_log_norm = log_norm
		else:
			pn_log_norm = DensityMatrixDrawer.__PNLogNorm(math.pow(10, lb_log10), math.pow(10, ub_log10), True)
		pn_log_levels: npt.NDArray[np.double] = np.zeros(num_ticks)
		pn_log_levels[DensityMatrixDrawer.__CMAP.N // 8 * 4 + 2:] = np.exp2(np.linspace(math.log2(pn_log_norm.abs_min), math.log2(pn_log_norm.abs_max), DensityMatrixDrawer.__CMAP.N // 8 * 4 + 1, True))
		pn_log_levels[DensityMatrixDrawer.__CMAP.N // 8 * 4::-1] = -pn_log_levels[DensityMatrixDrawer.__CMAP.N // 8 * 4 + 2:]
		if log_level is not None:
			log_level[...] = pn_log_levels
		return pn_log_norm, pn_log_levels, np.concatenate([pn_log_levels[0:DensityMatrixDrawer.__CMAP.N // 8 * 4:DensityMatrixDrawer.__CMAP.N // 8], pn_log_levels[DensityMatrixDrawer.__CMAP.N // 8 * 4 + 1:DensityMatrixDrawer.__CMAP.N // 8 * 4 + 2], pn_log_levels[DensityMatrixDrawer.__CMAP.N // 8 * 5 + 2::DensityMatrixDrawer.__CMAP.N // 8]])

	@staticmethod
	def __get_factor(
		fig: matplotlib.figure.Figure,
		ax: matplotlib.axes.Axes,
		array_shape: tuple[int, ...]
	) -> tuple[int, int]:
		r"""To get the omitting factors in plotting

		Parameters
		----------
		fig : matplotlib.figure.Figure
			The figure to plot
		ax : matplotlib.axes.Axes
			The axes to draw
		array_shape : tuple[int, ...]
			The shape of array to draw on the axes

		Returns
		-------
		tuple[int, int]
			The divisor of row and column of the array
		"""
		def get_divisor(dots: float, size: int) -> int:
			r"""To get the divisor of row or column

			Parameters
			----------
			dots : float
				The number of individual dots in the direction
			size : int
				An integer

			Returns
			-------
			set[int]
				All the factors of the parameter
			"""
			if dots > size:
				return 1
			else:
				all_factors: set[int] = set(functools.reduce(list.__add__, ([i, (size - 1) // i] for i in range(1, int(np.sqrt(size - 1)) + 1) if (size - 1) % i == 0)))
				factors_below_lim: set[int] = {i for i in all_factors if i < size / dots}
				return max(factors_below_lim)

		figsize_in_dots: typing.Final[npt.NDArray[np.double]] = fig.get_size_inches() * fig.dpi
		ax_bbox: typing.Final[matplotlib.transforms.Bbox] = ax.get_position()
		return get_divisor(figsize_in_dots[0] * ax_bbox.width, array_shape[0]), get_divisor(figsize_in_dots[1] * ax_bbox.height, array_shape[1])

	@staticmethod
	def __update_colorbar_limit(
		colorbar_limit: float,
		ctr_norm: matplotlib.colors.CenteredNorm,
		ctr_level: npt.NDArray[np.double],
		ctr_colorbar: matplotlib.colorbar.Colorbar,
		log_norm: __PNLogNorm | None,
		log_level: npt.NDArray[np.double] | None,
		log_colorbar: matplotlib.colorbar.Colorbar | None,
	) -> None:
		r"""Based on the given colorbar limit, update colorbar and norm

		Parameters
		----------
		colorbar_limit : float
			The maximum value on colorbar
		ctr_norm : matplotlib.colors.CenteredNorm
			Centered norm, for linear scale contour
		ctr_level : npt.NDArray[np.double]
			Level for linear scale contour
		ctr_colorbar : matplotlib.colorbar.Colorbar
			Linear colorbar
		log_norm : __PNLogNorm | None
			Pos-neg log norm, for logscale contour
		log_level : npt.NDArray[np.double] | None
			Level for logscale contour
		log_colorbar : matplotlib.colorbar.Colorbar | None
			Log colorbar
		"""
		_, _, ctr_ticks = DensityMatrixDrawer.__get_centered(
			colorbar_limit * utility.I_UB,
			ctr_norm=ctr_norm,
			ctr_level=ctr_level
		)
		ctr_colorbar.update_normal(matplotlib.cm.ScalarMappable(ctr_norm, DensityMatrixDrawer.__CMAP))
		ctr_colorbar.set_ticks(ctr_ticks.tolist())
		ctr_colorbar.ax.set_yticklabels(ctr_colorbar.ax.get_yticklabels(False), minor=False, fontweight="bold")
		assert (log_norm is None) == (log_level is None) == (log_colorbar is None)
		if log_norm is not None: # draw logscale:
			assert log_level is not None and log_colorbar is not None
			_, _, log_ticks = DensityMatrixDrawer.__get_posneg_log(
				colorbar_limit * sys.float_info.epsilon,
				colorbar_limit,
				log_norm=log_norm,
				log_level=log_level
			)
			log_colorbar.update_normal(matplotlib.cm.ScalarMappable(log_norm, DensityMatrixDrawer.__CMAP))
			log_colorbar.set_ticks(log_ticks.tolist())
			log_colorbar.ax.set_yticklabels(log_colorbar.ax.get_yticklabels(False), minor=False, fontweight="bold")

	@staticmethod
	def __get_ax_title(row_idx: int, col_idx: int, NUM_PES: int, has_grid: bool) -> str:
		r"""_summary_

		Parameters
		----------
		row_idx : int
			Index of the row in the axes
		col_idx : int
			Index of the column in the axes
		NUM_PES : int
			The number of potential energy surfaces
		has_grid : bool
			Whether has grid solution as comparison or not

		Returns
		-------
		str
			Axes title
		"""
		assert 0 <= row_idx < NUM_PES
		if has_grid:
			assert 0 <= col_idx < NUM_PES * 2
			return f"{"GP" if col_idx % 2 == 0 else "Grid"} {utility.get_RI_label(row_idx * NUM_PES + col_idx // 2, NUM_PES)}"
		else:
			assert 0 <= col_idx < NUM_PES
			return utility.get_RI_label(row_idx * NUM_PES + col_idx, NUM_PES)

	@staticmethod
	def __fig_axs_initialize(
		fig: matplotlib.figure.Figure,
		axs: np.ndarray,
		config: pes.ModelConfig,
		colorbar_limit: float,
		draw_logscale: bool,
		draw_scattered: bool,
		xv: npt.NDArray[np.double],
		pv: npt.NDArray[np.double],
	) -> tuple[matplotlib.colors.CenteredNorm, npt.NDArray[np.double], matplotlib.colorbar.Colorbar, __PNLogNorm | None, npt.NDArray[np.double] | None, matplotlib.colorbar.Colorbar | None]:
		r"""To initialize figure an axes

		Parameters
		----------
		fig : matplotlib.figure.Figure
			The figure
		axs : np.ndarray
			Array of axes
		config : pes.ModelConfig
			Configuration of the model
		colorbar_limit : float
			The maximum value on color bar
		draw_logscale : bool
			Whether to draw logscale figures or not
		draw_scattered : bool
			Whether to draw the sample points or not
		xv : npt.NDArray[np.double], shape of (N_GRIDS, N_GRIDS)
			Grid coordinates for position
		pv : npt.NDArray[np.double], shape of (N_GRIDS, N_GRIDS)
			Grid coordinates for momentum

		Returns
		-------
		tuple[matplotlib.colors.CenteredNorm, npt.NDArray[np.double], __PNLogNorm, npt.NDArray[np.double]]
			Centered norm, centered ticks, log norm that has positive and negative (pos-neg log norm, or PN log norm), and pos-neg log ticks
		"""
		def set_colorbar(cbar: matplotlib.colorbar.Colorbar) -> None:
			r"""To set up the colorbar axes

			Parameters
			----------
			cbar : matplotlib.colorbar.Colorbar
				Colorbar
			"""
			cbar.ax.set_title("Population Density", fontweight="semibold")
			cbar.ax.tick_params(axis="y", length=10.0, width=3.0)
			cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(False), minor=False, fontweight="bold")
			for spine in cbar.ax.spines.values():
				spine.set_linewidth(5.0)

		ctr_norm, ctr_level, ctr_ticks = DensityMatrixDrawer.__get_centered(colorbar_limit * utility.I_UB)
		ctr_cbar: typing.Final[matplotlib.colorbar.Colorbar] = fig.colorbar(
			matplotlib.cm.ScalarMappable(ctr_norm, DensityMatrixDrawer.__CMAP),
			ax=axs[:config.NUM_PES * (1 + int(draw_scattered)), :],
			ticks=ctr_ticks.tolist()
		)
		set_colorbar(ctr_cbar)
		for iRow in range(config.NUM_PES * (1 + int(draw_scattered))):
			for iCol in range(axs.shape[1]):
				ax: matplotlib.axes.Axes = axs[iRow, iCol]
				utility.axes_decoration(
					ax,
					iRow * config.NUM_PES + iCol,
					xlabel="R [a.u.]" if iRow == config.NUM_PES - 1 and not draw_logscale else None,
					ylabel="P [a.u.]" if iCol == 0 else None,
					title=("Scattered " if iRow > config.NUM_PES else "") + DensityMatrixDrawer.__get_ax_title(iRow % config.NUM_PES, iCol, config.NUM_PES, axs.shape[1] == config.NUM_PES)
				)
				ax.contourf(xv, pv, np.zeros_like(xv), levels=ctr_level, cmap=DensityMatrixDrawer.__CMAP, norm=ctr_norm)
		log_norm: DensityMatrixDrawer.__PNLogNorm | None = None
		log_level: npt.NDArray[np.double] | None = None
		log_cbar: matplotlib.colorbar.Colorbar | None = None
		if draw_logscale: # draw logscale:
			log_norm, log_level, log_ticks = DensityMatrixDrawer.__get_posneg_log(colorbar_limit * sys.float_info.epsilon, colorbar_limit)
			log_cbar = fig.colorbar( # logscale colorbar
				matplotlib.cm.ScalarMappable(log_norm, DensityMatrixDrawer.__CMAP),
				ax=axs[config.NUM_PES * (1 + int(draw_scattered)):, :],
				ticks=log_ticks.tolist(),
				format=r"%+.1e"
			)
			set_colorbar(log_cbar)
			for iRow in config.PES_RANGE:
				for iCol in range(axs.shape[1]):
					ax: matplotlib.axes.Axes = axs[iRow + config.NUM_PES * (1 + int(draw_scattered)), iCol]
					utility.axes_decoration(
						ax,
						iRow * config.NUM_PES + iCol + config.NUM_PES * (1 + int(draw_scattered)) * axs.shape[1],
						xlabel="R [a.u.]" if iRow == config.NUM_PES - 1 else None,
						ylabel="P [a.u.]" if iCol == 0 else None,
						title="Logscale " + DensityMatrixDrawer.__get_ax_title(iRow, iCol, config.NUM_PES, axs.shape[1] == config.NUM_PES)
					)
					ax.contourf(xv, pv, np.zeros_like(xv), levels=log_level, cmap=DensityMatrixDrawer.__CMAP, norm=log_norm)
		return ctr_norm, ctr_level, ctr_cbar, log_norm, log_level, log_cbar

	__slots__: typing.ClassVar[tuple] = ("config", "n_grids", "__output_interval", "__draw_rescaled", "__draw_logscale", "__draw_scattered", "__title", "__picname", "__row_divisor", "__col_divisor", "__xv", "__pv", "__ctr_norm", "__ctr_level", "__ctr_cbar", "__log_norm", "__log_level", "__log_cbar", "__fig", "__axs")
	config: typing.Final[pes.ModelConfig]
	n_grids: typing.Final[int]
	__output_interval: typing.Final[float]
	__draw_rescaled: typing.Final[bool]
	__draw_logscale: typing.Final[bool]
	__draw_scattered: typing.Final[bool]
	__title: typing.Final[str]
	__picname: typing.Final[str]
	__row_divisor: typing.Final[int]
	__col_divisor: typing.Final[int]
	__xv: typing.Final[npt.NDArray[np.double]]
	__pv: typing.Final[npt.NDArray[np.double]]
	__ctr_norm: typing.Final[matplotlib.colors.CenteredNorm]
	__ctr_level: typing.Final[npt.NDArray[np.double]]
	__ctr_cbar: typing.Final[matplotlib.colorbar.Colorbar]
	__log_norm: typing.Final[__PNLogNorm | None]
	__log_level: typing.Final[npt.NDArray[np.double] | None]
	__log_cbar: typing.Final[matplotlib.colorbar.Colorbar | None]
	__fig: typing.Final[matplotlib.figure.Figure]
	__axs: typing.Final[np.ndarray]

	def __init__(
		self,
		quantity: param.Quantity,
		draw_rescaled: bool,
		draw_logscale: bool,
		draw_scattered: bool,
		max_outputs: int,
		initial_dm: torch.Tensor,
		initial_grid_dm: torch.Tensor | None = None,
		initial_points: collections.abc.Sequence[npt.NDArray[np.double]] | None = None,
		initial_scale: npt.NDArray[np.double] | None = None,
	) -> None:
		assert quantity.config.PHASEDIM == DIM_PLOT_DM
		# directly save from parameters
		self.config = quantity.config
		self.n_grids = quantity.num_grids_in_total
		self.__output_interval = quantity.output_interval
		self.__draw_rescaled = draw_rescaled
		self.__draw_logscale = draw_logscale
		self.__draw_scattered = draw_scattered
		self.__title = f"{"Rescaled " if self.__draw_rescaled else ""}Adiabatic Partial Wigner-Transformed Density Matrix"
		self.__picname = utility.get_picname(f"{constant.PWTDM_FILENAME}_adiabatic", max_outputs)
		# figure and axes
		N_ROWS: typing.Final = (1 + int(draw_logscale) + int(draw_scattered)) * self.config.NUM_PES
		N_COLS: typing.Final = (2 if initial_grid_dm is not None else 1) * self.config.NUM_PES
		self.__fig = plt.figure(figsize=(constant.FIGSIZE[0] * N_COLS, constant.FIGSIZE[1] * N_ROWS), layout="constrained")
		self.__axs = self.__fig.subplots(nrows=N_ROWS, ncols=N_COLS, squeeze=False)
		self.__row_divisor, self.__col_divisor = DensityMatrixDrawer.__get_factor(self.__fig, self.__axs[0, 0], initial_dm.shape[:-2])
		self.__xv, self.__pv = np.meshgrid(quantity.x_grids_each_dim[0].reshape(-1).detach().cpu().numpy(), quantity.p_grids_each_dim[0].reshape(-1).detach().cpu().numpy(), indexing="xy")
		self.__xv = self.__xv[::self.__row_divisor, ::self.__col_divisor]
		self.__pv = self.__pv[::self.__row_divisor, ::self.__col_divisor]
		self.__ctr_norm, self.__ctr_level, self.__ctr_cbar, self.__log_norm, self.__log_level, self.__log_cbar = DensityMatrixDrawer.__fig_axs_initialize(
			self.__fig,
			self.__axs,
			quantity.config,
			1.0 if draw_rescaled else utility.I_UB * np.max(np.abs(initial_dm)),
			self.__draw_logscale,
			self.__draw_scattered,
			self.__xv,
			self.__pv
		)
		DensityMatrixDrawer.__call__(self, 0, initial_dm, initial_grid_dm, initial_points, initial_scale)

	@typing.final
	@property
	def picname(self) -> str:
		r"""To get the naming template for pictures

		Returns
		-------
		str
			Naming template, should be used with `.format(frame_index)`
		"""
		return self.__picname

	def __call__(
		self,
		frame_index: int,
		dm: torch.Tensor,
		grid_dm: torch.Tensor | None = None,
		points: collections.abc.Sequence[npt.NDArray[np.double]] | None = None,
		scale: npt.NDArray[np.double] | None = None
	) -> None:
		r"""To draw a frame and save the picture

		Parameters
		----------
		frame_index : int
			The index of the frame. Product with output interval gives the duration since beginning
		dm : torch.Tensor, dtype of `torch.cdouble`, shape of (NUM_PES, NUM_PES, N_GRID, N_GRID)
			Adiabatic PWTDM
		points : collections.abc.Sequence[npt.NDArray[np.double]] | None, len of 2 * NUM_TRIG, each of shape (NUM_PTS, PHASEDIM), optional
			The points to scatter. Only used if `self.__draw_scattered` is True.
			By default None
		scale : npt.NDArray[np.double], shape of (NUM_ELM,)
			The rescale factor, by default None (grid density will be used to calculate it instead)
		"""
		def draw_ax(
			ax: matplotlib.axes.Axes,
			row_idx: int,
			col_idx: int,
			data: npt.NDArray[np.double],
			level: npt.NDArray[np.double],
			norm: matplotlib.colors.Normalize,
			central_points: npt.NDArray[np.double] | None = None,
			extra_points: npt.NDArray[np.double] | None = None
		) -> None:
			r"""To draw an axe

			Parameters
			----------
			ax : matplotlib.axes.Axes
				The axes to draw the data
			row_idx : int
				Index of the row in the axes
			col_idx : int
				Index of the column in the axes
			data : npt.NDArray[np.double]
				The data to be drawn on the axes
			level : npt.NDArray[np.double]
				Levels of the contours
			norm : matplotlib.colors.Normalize
				Mapping from data to colorbar [0,1] range
			central_points : npt.NDArray[np.double], shape of (NUM_PES, NUM_PTS) | None, optional
				The central points to scatter on the element, by default None
			extra_points : npt.NDArray[np.double], shape of (NUM_PES, NUM_PTS * EXTRA_RATIO) | None, optional
				The extra points to scatter on the element, by default None
			"""
			rescale_factor: float
			if self.__draw_rescaled:
				# get rescaled factor, print it, and rescale the data
				if scale is not None:
					rescale_factor = scale[row_idx * self.config.NUM_PES + col_idx // (1 if self.__axs.shape[1] == self.config.NUM_PES else 2)]
				else:
					rescale_factor = 1.0 / np.max(np.abs(data))
				ax.set_title(
					DensityMatrixDrawer.__get_ax_title(row_idx, col_idx, self.config.NUM_PES, grid_dm is not None) + "\n" + constant.RESCALE_TEMPLATE.format(rescale_factor),
					fontproperties=utility.TITLE_PROPERTY
				)
			else:
				rescale_factor = 1.0
			ax.contourf(
				self.__xv,
				self.__pv,
				data * rescale_factor,
				levels=level,
				cmap=DensityMatrixDrawer.__CMAP,
				norm=norm
			)
			# scatter the central points on top
			if extra_points is not None:
				ax.scatter(
					np.clip(extra_points[0], self.__xv[0, 0], self.__xv[-1, -1]),
					np.clip(extra_points[1], self.__pv[0, 0], self.__pv[-1, -1]),
					DensityMatrixDrawer.__XTR_PT_SIZE,
					DensityMatrixDrawer.__XTR_PT_COLOR
				)
			if central_points is not None:
				ax.scatter(
					np.clip(central_points[0], self.__xv[0, 0], self.__xv[-1, -1]),
					np.clip(central_points[1], self.__pv[0, 0], self.__pv[-1, -1]),
					DensityMatrixDrawer.__CEN_PT_SIZE,
					DensityMatrixDrawer.__CEN_PT_COLOR
				)

		dm_to_draw: typing.Final[list[npt.NDArray[np.double]]] = [d[::self.__row_divisor, ::self.__col_divisor] for d in dm.detach().cpu().numpy().reshape(self.config.NUM_ELM, *dm.shape[2:])] + ([d[::self.__row_divisor, ::self.__col_divisor] for d in grid_dm.detach().cpu().numpy().reshape(self.config.NUM_ELM, *grid_dm.shape[2:])] if grid_dm is not None else []) # gpr first, then grid
		if not self.__draw_rescaled: # update colorbar
			DensityMatrixDrawer.__update_colorbar_limit(
				utility.I_UB * max(np.max(np.abs(dm)) for dm in dm_to_draw),
				self.__ctr_norm,
				self.__ctr_level,
				self.__ctr_cbar,
				self.__log_norm if self.__log_norm is not None else None,
				self.__log_level if self.__log_level is not None else None,
				self.__log_cbar if self.__log_cbar is not None else None
			)
		for iPES in self.config.PES_RANGE:
			for jPES in (self.config.PES_RANGE if grid_dm is None else range(self.__axs.shape[1])):
				draw_ax(
					self.__axs[iPES, jPES],
					iPES,
					jPES,
					dm_to_draw[iPES * self.config.NUM_PES + jPES + (self.config.NUM_ELM if grid_dm is not None else 0)],
					self.__ctr_level,
					self.__ctr_norm
				)
				if self.__draw_scattered:
					assert points is not None
					TrilIndex: int = self.config.FLATTEN_TRIL_INDEX[iPES * self.config.NUM_PES + jPES // (1 if grid_dm is None else 2)]
					draw_ax(
						self.__axs[iPES + self.config.NUM_PES, jPES],
						iPES,
						jPES,
						dm_to_draw[iPES * self.config.NUM_PES + jPES + (self.config.NUM_ELM if grid_dm is not None else 0)],
						self.__ctr_level,
						self.__ctr_norm,
						points[TrilIndex].T,
						points[TrilIndex + self.config.NUM_TRIG].T,
					)
				if self.__draw_logscale:
					assert self.__log_norm is not None and self.__log_level is not None
					draw_ax(
						self.__axs[iPES + self.config.NUM_PES * (1 + int(self.__draw_scattered)), jPES],
						iPES,
						jPES,
						dm_to_draw[iPES * self.config.NUM_PES + jPES + (self.config.NUM_ELM if grid_dm is not None else 0)],
						self.__log_level,
						self.__log_norm
					)
		self.__fig.suptitle(self.__title + "\n" + constant.TIME_TEMPLATE.format(frame_index * self.__output_interval), fontproperties=utility.TITLE_PROPERTY)
		self.__fig.savefig(self.__picname.format(frame_index))


@typing.final
class DMDrawerFromFile(DensityMatrixDrawer, utility.FromFile):
	r"""To draw the density matrix whose data is read from file

	Parameters
	----------
	quantity : param.QuantityForEvolution,
		Quantities including grids, output intervals, etc
	draw_rescaled : bool
		True to make the maximum value of each element to the same value;
		False to keep the original value.
	draw_diabatic : bool
		Whether to draw diabatic marginals as well
	dm_data_filename : str, optional
		The file name saving the partial Wigner-transformed density matrix,
		by default `f"{constant.PWTDM_FILENAME}{constant.DATA_EXTENSION}"`
	pts_filename : str, optional
		The file name saving the indices of the sample points,
		by default `f"{constant.POINTS_FILENAME}{constant.DATA_EXTENSION}"`
	belonging_filename : str, optional
		The file name saving the indices of the points
		by default `f"{constant.BELONGING_FILENAME}{constant.DATA_EXTENSION}"`
	grid_solution_filename : str, optional
		The file containing grid solution, by default "" (meaning no such file)
	scale : npt.NDArray[np.double], shape of (NUM_ELM,)
		The rescale factor, by default None (grid density will be used to calculate it instead)


	Methods
	-------
	total_ticks()
		To get the total number of frames in the file
	"""
	__slots__ = ("__grid_solution", "__points", "__belongings")
	__grid_solution: typing.Final[utility.FromFile | None]
	__points: typing.Final[npt.NDArray[np.double] | None]
	__belongings: typing.Final[npt. NDArray[np.int_] | None]

	def __init__(
		self,
		quantity: param.Quantity,
		draw_rescaled: bool,
		draw_logscale: bool,
		draw_scattered: bool,
		dm_data_filename: str = constant.PWTDM_FILENAME + constant.DATA_EXTENSION,
		pts_filename: str = constant.POINTS_FILENAME + constant.DATA_EXTENSION,
		belonging_filename: str = constant.BELONGING_FILENAME + constant.DATA_EXTENSION,
		grid_solution_filename: str = "",
		initial_scale: npt.NDArray[np.double] | None = None
	) -> None:
		# get from file
		utility.FromFile.__init__(self, dm_data_filename, quantity.config.NUM_ELM)
		if grid_solution_filename != "":
			self.__grid_solution = utility.FromFile(grid_solution_filename, quantity.config.NUM_ELM)
		else:
			self.__grid_solution = None
		if draw_scattered:
			# read from file
			self.__points = np.loadtxt(pts_filename)
			self.__points = self.__points.reshape(self.__points.shape[0] // quantity.config.PHASEDIM, quantity.config.PHASEDIM, self.__points.shape[1]) # N_TICKS * PHASEDIM * NUM_PTS
			self.__belongings = np.loadtxt(belonging_filename).astype(np.int_) # N_TICKS * NUM_PTS
		else:
			self.__points = None
			self.__belongings = None
		# first frame
		DensityMatrixDrawer.__init__(
			self,
			quantity,
			draw_rescaled,
			draw_logscale,
			draw_scattered,
			self.total_ticks if self.total_ticks != -1 else quantity.total_ticks,
			wfn.file_data_to_dm(self, quantity.config, [quantity.num_grids_in_total]),
			wfn.file_data_to_dm(self.__grid_solution, quantity.config, [quantity.num_grids_in_total]) if self.__grid_solution is not None else None,
			([self.__points[0, :, self.__belongings[0] == idx] for idx in quantity.config.TRIL_ELEMENT_INDICES] + [self.__points[0, :, self.__belongings[0] == idx + quantity.config.NUM_ELM] for idx in quantity.config.TRIL_ELEMENT_INDICES]) if self.__points is not None and self.__belongings is not None else None,
			initial_scale
		)

	def __call__(self, scale: npt.NDArray[np.double] | None = None) -> bool:
		r"""To draw a frame and save the picture

		Parameters
		----------
		scale : npt.NDArray[np.double], shape of (NUM_ELM,)
			The rescale factor, by default None (grid density will be used to calculate it instead)

		Returns
		----------
		bool
			Whether there are remaining figures to draw or not
		"""
		if self.have_content:
			try:
				DensityMatrixDrawer.__call__(
					self,
					self.frame_index - 1,
					wfn.file_data_to_dm(self, self.config, [self.n_grids]),
					wfn.file_data_to_dm(self.__grid_solution, self.config, [self.n_grids]) if self.__grid_solution is not None else None,
					([self.__points[self.frame_index - 1, :, self.__belongings[self.frame_index - 1] == idx] for idx in self.config.TRIL_ELEMENT_INDICES] + [self.__points[self.frame_index - 1, :, self.__belongings[self.frame_index - 1] == idx + self.config.NUM_ELM] for idx in self.config.TRIL_ELEMENT_INDICES]) if self.__points is not None and self.__belongings is not None else None,
					scale
				) # this already increase frame index
				return True
			except EOFError:
				utility.FromFile.close(self)
				return False
		else:
			utility.FromFile.close(self)
			return False
