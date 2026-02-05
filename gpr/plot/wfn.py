r"""plot.wfn
========
Implementation of plotting marginal distributions
"""
import io
import os
import subprocess
import typing

import matplotlib
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch

import constant
import param
import pes

from . import utility

torch.set_default_dtype(constant.DTYPE)


def file_data_to_dm(
	file: utility.FromFile,
	config: pes.ModelConfig,
	num_grids_each_dim: list[int]
) -> torch.Tensor:
	r"""To extra density matrix from data from file

	Parameters
	----------
	file : utility.FromFile
		File handler containing data
	config : pes.ModelConfig
		The configuration of the model
	num_grids_each_dim : list[int]
		The number of grids on each dimension, used for shaping the data lines from file

	Returns
	-------
	torch.Tensor, dtype of torch.cdouble, shape of : N_GRIDS, ..., N_GRIDS, ..., NUM_PES, NUM_PES
		PWTDM read from File
	"""
	return utility.file_read_density_matrix_format(torch.from_numpy(file.get_data()), num_grids_each_dim, config.NUM_PES)


class MarginalProbabilityPlot:
	r"""To plot the probability, i.e., the squared norm of wavefunction or the marginal distribution of phase space.

	Parameters
	----------
	config : pes.ModelConfig
		Configuration of the model
	output_interval : float
		Interval between outputs, in unit of a.u.
	draw_rescaled : bool
		True to make the maximum value of each surface to the same value;
		False to keep the original value
	from_dm : bool
		True if it is marginal distribution from density matrices; False if it is squared norm of wavefunction
	max_outputs : int
		Estimation of maximum outputs
	grids_each_dim : list[npt.NDArray[np.double]], len of (1~2) * PHASEDIM, each of shape (N_GRID,)
		The grids for position and momentum
	initial_probability : list[npt.NDArray[np.double]], len of (1~2) * PHASEDIM, each of shape (N_GRID, NUM_PES)
		Initial wavefunction, in configuration space and in momentum space
	picname : str | None, optional
		Naming template, should be used with `.format(frame_index)`, by default None (and use the class picname instead)

	Attributes
	----------
	config : pes.ModelConfig
		Configuration of the model

	Methods
	-------
	picnames()
		To get the naming template for pictures
	"""
	__slots__: typing.ClassVar[tuple] = ("config", "__output_interval", "__draw_rescaled", "__marginal_ranges", "__title", "__picname", "__grids_each_dim", "__wfn_colors", "__fig", "__axs")
	config: typing.Final[pes.ModelConfig]
	__output_interval: typing.Final[float]
	__draw_rescaled: typing.Final[bool]
	__marginal_ranges: typing.Final[tuple[int, ...]]
	__title: typing.Final[str]
	__picname: typing.Final[str]
	__grids_each_dim: typing.Final[list[npt.NDArray[np.double]]]
	__wfn_colors: typing.Final[npt.NDArray[np.double]]
	__fig: typing.Final[matplotlib.figure.Figure]
	__axs: typing.Final[np.ndarray]

	def __get_ax(self, DimIndex: int) -> matplotlib.axes.Axes:
		r"""To get the corresponding axes for plot

		Axes are in such an order:

		0 & PHASEDIM & 2 & PHASEDIM+2 & ... //

		1 & PHASEDIM+1 & 3 & PHASEDIM+3 & ...

		Parameters
		----------
		DimIndex : int
			Index of the dimension

		Returns
		-------
		matplotlib.axes.Axes
			The axes to plot

		Raises
		------
		IndexError
			Unexpected index
		"""
		assert 0 <= DimIndex <= self.__marginal_ranges[-1]
		if self.__marginal_ranges[-1] == self.config.PHASEDIM:
			return self.__axs[DimIndex % 2, DimIndex // 2]
		else:
			return self.__axs[DimIndex % 2, DimIndex % self.config.PHASEDIM // 2 * 2 + DimIndex // self.config.PHASEDIM]

	def __init__(
		self,
		config: pes.ModelConfig,
		output_interval: float,
		draw_rescaled: bool,
		from_dm: bool,
		max_outputs: int,
		grids_each_dim: list[npt.NDArray[np.double]],
		initial_probability: list[npt.NDArray[np.double]],
		picname: str | None = None,
	) -> None:
		# directly save from parameters
		self.config = config
		self.__output_interval = output_interval
		self.__draw_rescaled = draw_rescaled
		# ranges, whether has grid solution as comparison or not
		assert len(grids_each_dim) == len(initial_probability)
		NUM_MARG: typing.Final[int] = len(initial_probability)
		assert NUM_MARG == config.PHASEDIM or NUM_MARG == 2 * config.PHASEDIM
		self.__marginal_ranges = tuple(range(NUM_MARG))
		has_dia: typing.Final[bool] = NUM_MARG // config.PHASEDIM == 2
		# title, filename, and grids
		self.__title = f"{"Rescaled " if draw_rescaled else ""}Adiabatic {"Marginal" if from_dm else "Population"} on Each Surface"
		self.__picname = utility.get_picname(f"{constant.WFN_FILENAME}_adiabatic", max_outputs)
		self.__grids_each_dim = [grids.reshape(-1) for grids in grids_each_dim] # x and p
		if config.NUM_PES < len(matplotlib.color_sequences["Set1"]):
			self.__wfn_colors = np.array(matplotlib.color_sequences["Set1"][:config.NUM_PES + 1])
		else:
			self.__wfn_colors = matplotlib.colormaps["gist_rainbow"](np.linspace(0.0, 1.0, config.NUM_PES + 1))
		# figure and axes
		self.__fig = plt.figure(figsize=(constant.FIGSIZE[0] * config.DIM * NUM_MARG // config.PHASEDIM, constant.FIGSIZE[1] * 2), layout="tight")
		self.__axs = self.__fig.subplots(2, config.DIM * NUM_MARG // config.PHASEDIM, squeeze=False)
		max_y: npt.NDArray[np.double] = np.ones(NUM_MARG, np.double)
		if not self.__draw_rescaled:
			max_y = np.array([utility.I_UB * np.max(dim_prob) for dim_prob in initial_probability], np.double)
		for iDim in self.__marginal_ranges:
			ax: matplotlib.axes.Axes = self.__get_ax(iDim)
			utility.axes_decoration(
				ax,
				iDim,
				xlabel=f"{utility.dimension_name(iDim % config.PHASEDIM, config.DIM)} [a.u.]",
				ylabel="Population" if iDim < 2 else None,
				title=f"{"GP" if iDim < config.PHASEDIM else "Grid"} {utility.dimension_name(iDim, config.DIM)}"
			)
			ax.set_xlim(self.__grids_each_dim[iDim][0], self.__grids_each_dim[iDim][-1])
			ax.set_ybound(0.0, 1.5 * max_y[iDim])
		MarginalProbabilityPlot.__call__(self, 0, initial_probability, picname)

	@property
	def picname(self) -> str:
		r"""To get the naming template for pictures for diabatic basis

		Returns
		-------
		str
			Naming template, should be used with `.format(frame_index)`
		"""
		return self.__picname

	def __call__(
		self,
		frame_index: int,
		probability: list[npt.NDArray[np.double]],
		picname: str | None = None,
	) -> None:
		r"""To draw a frame and save as png

		Parameters
		----------
		frame_index : int
			The index of the frame.
			Product with `self.__output_interval` gives the duration since beginning
		probability : list[npt.NDArray[np.double]], len of (1~2) * PHASEDIM, each of shape (NUM_PES, NUM_PES, N_GRIDS)
			Probability of each dimension
		picnames : str | None, optional
			Naming template, should be used with `.format(frame_index)`, by default None (and use the class picname instead)
		"""
		assert len(probability) == self.__axs.size
		for iDim in self.__marginal_ranges:
			ax: matplotlib.axes.Axes = self.__get_ax(iDim)
			if ax.legend_:
				ax.legend_.remove()
			for line in ax.lines:
				line.remove()
			rescale_factor: npt.NDArray[np.double] = 1.0 / np.where(np.all(probability[iDim] == 0.0, 0), 1.0, np.max(np.abs(probability[iDim]), 0)) if self.__draw_rescaled else np.ones(self.config.NUM_PES)
			for iPES in self.config.PES_RANGE:
				ax.plot(
					self.__grids_each_dim[iDim],
					probability[iDim][iPES, iPES] * rescale_factor[iPES],
					color=self.__wfn_colors[iPES],
					lw=utility.LINE_WIDTH,
					label=utility.pes_name(iPES, self.config.NUM_PES)
				)
			# if not self.__draw_rescaled: # the sum is plot only when no rescaled
			# 	ax.plot(
			# 		self.__grids_each_dim[iDim],
			# 		probability[iDim].sum(-1),
			# 		color=self.__wfn_colors[self.config.NUM_PES],
			# 		lw=utility.LINE_WIDTH,
			# 		label=utility.pes_name(self.config.NUM_PES, self.config.NUM_PES)
			# 	)
			ax.legend(**utility.LEGEND_PROPERTY)
		self.__fig.suptitle(self.__title + "\n" + constant.TIME_TEMPLATE.format(frame_index * self.__output_interval), fontproperties=utility.TITLE_PROPERTY)
		self.__fig.savefig((picname if picname is not None else self.__picname).format(frame_index))

class DensityMatrixMarginalPlotter(MarginalProbabilityPlot):
	r"""To plot marginal distribution of PWTDM

	Parameters
	----------
	quantity : param.Quantity,
		Quantities including grids, output intervals, etc
	draw_rescaled : bool
		True to make the maximum value of each surface to the same value;
		False to keep the original value
	max_outputs : int
		Estimation of maximum outputs
	initial_gpr_dm : torch.Tensor, dtype of `torch.cdouble`, shape of (N_GRID, ..., N_GRID, ..., NUM_PES, NUM_PES)
		Initial adiabatic PWTDM, from Gaussian process
	initial_grid_dm : torch.Tensor, dtype of `torch.cdouble`, shape of (N_GRID, ..., N_GRID, ..., NUM_PES, NUM_PES) | None
		Initial adiabatic PWTDM, from grid solution file, in accordance with `initial_gpr_dm`, if provided

	Methods
	-------
	picname()
		To get the naming template for pictures
	"""
	__slots__: typing.ClassVar[tuple] = ("__volume_elements_reduce_to_each_dim", "__picname") # pylint: disable=redefined-slots-in-subclass
	__volume_elements_reduce_to_each_dim: typing.Final[npt.NDArray[np.double]]
	__picname: typing.Final[str]

	@staticmethod
	def __dm_to_1d_probability(
		dm: torch.Tensor | None,
		config: pes.ModelConfig
	) -> list[npt.NDArray[np.double]]:
		r"""To convert the wavefunction in configuration space :math:`\psi(\mathbf{x})` to momentum space :math:`\phi(\mathbf{p})`, then reduce the other dimensions and get the population on each dimension

		Parameters
		----------
		dm : torch.Tensor, dtype of `torch.cdouble`, shape of (N_GRID, ..., N_GRID, ..., NUM_PES, NUM_PES) | None
			Adiabatic PWTDM
		config : pes.ModelConfig
			The configuration of the model

		Returns
		-------
		list[npt.NDArray[np.double]], len of PHASEDIM, each of shape (N_GRIDS, NUM_PES)
			Probability on each dimension of each surface
		"""
		if dm is None:
			return []
		else:
			dm_diag: typing.Final[npt.NDArray[np.double]] = dm[..., config.PES_RANGE, config.PES_RANGE].real.detach().cpu().numpy()
			return [dm_diag.sum(config.SUM_DIMS_FOR_PWTDM_MARGINAL[iDim]) * DensityMatrixMarginalPlotter.__volume_elements_reduce_to_each_dim[iDim].item() for iDim in config.PHASEDIM_RANGE]

	def __init__(
		self,
		quantity: param.Quantity,
		draw_rescaled: bool,
		max_outputs: int,
		initial_gpr_marginal: list[npt.NDArray[np.double]],
		initial_grid_dm: torch.Tensor | None
	) -> None:
		self.__volume_elements_reduce_to_each_dim = (quantity.dx.prod() * quantity.dp.prod()).item() / torch.cat([quantity.dx, quantity.dp]).detach().cpu().numpy()
		self.__picname = utility.get_picname(f"{constant.PWTDM_MARGINAL_FILENAME}_adiabatic", max_outputs)
		super().__init__(
			quantity.config,
			quantity.output_interval,
			draw_rescaled,
			True,
			max_outputs,
			[grids.detach().cpu().numpy() for grids in quantity.x_grids_each_dim + quantity.p_grids_each_dim],
			initial_gpr_marginal + DensityMatrixMarginalPlotter.__dm_to_1d_probability(initial_grid_dm, quantity.config),
			self.__picname,
		)

	@property
	def picname(self) -> str:
		r"""To get the naming template for pictures

		Returns
		-------
		str
			Naming template, should be used with `.format(frame_index)`
		"""
		return self.__picname

	def __call__( # pylint: disable=signature-differs
		self,
		frame_index: int,
		gpr_marginal: list[npt.NDArray[np.double]],
		grid_dm: torch.Tensor | None
	) -> None:
		r"""To draw a frame and save as png

		Parameters
		----------
		frame_index : int
			The index of the frame.
		gpr_dm : torch.Tensor, dtype of `torch.cdouble`, shape of (N_GRID, ..., N_GRID, ..., NUM_PES, NUM_PES)
			Adiabatic PWTDM from GPR
		grid_dm : torch.Tensor, dtype of `torch.cdouble`, shape of (N_GRID, ..., N_GRID, ..., NUM_PES, NUM_PES) | None
			Adiabatic PWTDM from grid solution file, in accordance with `gpr_dm`, if provided
		"""
		super().__call__(frame_index, gpr_marginal + DensityMatrixMarginalPlotter.__dm_to_1d_probability(grid_dm, self.config), self.__picname)


@typing.final
class DMMarginalPlotterFromFile(DensityMatrixMarginalPlotter):
	r"""To plot wavefunctions from marginal density matrices whose data is read from file

	Parameters
	----------
	quantity : param.QuantityForEvolution,
		Quantities including grids, output intervals, etc
	draw_rescaled : bool
		True to make the maximum value of each element to the same value;
		False to keep the original value.
	marginal_file : str, optional
		The file name saving the marginal data,
		by default `f"{constant.MARGINAL_FILENAME}{constant.DATA_EXTENSION}"`
	grid_solution_filename : str, optional
		The file containing grid solution, by default "" (meaning no such file)

	Methods
	-------
	total_ticks()
		To get the total number of outputs
	frame_index()
		To get the number of plots already drawn
	"""
	__slots__: typing.Final[tuple] = ("__n_grids_each_dim", "__max_outputs", "__marginal_file", "__current_idx", "__grid_solution_file")
	__n_grids_each_dim: typing.Final[list[int]]
	__max_outputs: typing.Final[int]
	__marginal_file: typing.Final[io.TextIOWrapper]
	__current_idx: int
	__grid_solution_file: typing.Final[utility.FromFile | None]

	def __init__(
		self,
		quantity: param.Quantity,
		draw_rescaled: bool,
		marginal_filename: str = constant.MARGINAL_FILENAME + constant.DATA_EXTENSION,
		grid_solution_filename: str = "",
	) -> None:
		# get shape
		self.__n_grids_each_dim = quantity.num_grids_on_each_dimension.tolist()
		# get from file
		self.__max_outputs = int(subprocess.check_output(("wc", "-l", marginal_filename)).split()[0]) // (quantity.config.PHASEDIM + 1)
		self.__marginal_file = open(marginal_filename, "r", encoding=constant.ENC)
		if grid_solution_filename != "":
			self.__grid_solution_file = utility.FromFile(grid_solution_filename, quantity.config.NUM_ELM)
		# first frame
		super().__init__(
			quantity,
			draw_rescaled,
			self.__max_outputs,
			[np.loadtxt(self.__marginal_file, encoding=constant.ENC, max_rows=1).reshape(-1) for _ in quantity.config.PHASEDIM_RANGE],
			file_data_to_dm(self.__grid_solution_file, quantity.config, self.__n_grids_each_dim) if self.__grid_solution_file is not None else None
		)
		self.__marginal_file.readline()
		self.__current_idx = 0

	@property
	def total_ticks(self) -> int:
		r"""To get the total number of outputs

		Returns
		-------
		int
			The total number of outputs frames from file
		"""
		if self.__grid_solution_file is not None and self.__grid_solution_file.total_ticks != -1:
			return min(self.__grid_solution_file.total_ticks, self.__max_outputs)
		else:
			return self.__max_outputs

	@property
	def frame_index(self) -> int:
		r"""To get the number of plots already drawn

		Returns
		-------
		int
			The number of plots that has already been drawn
		"""
		return self.__current_idx

	def __call__(self) -> bool:
		self.__current_idx += 1
		if self.__current_idx < self.__max_outputs and (self.__grid_solution_file is None or self.__grid_solution_file.have_content):
			try:
				DensityMatrixMarginalPlotter.__call__(
					self,
					self.__current_idx,
					[np.loadtxt(self.__marginal_file, encoding=constant.ENC, max_rows=1).reshape(-1) for _ in self.config.PHASEDIM_RANGE],
					file_data_to_dm(self.__grid_solution_file, self.config, self.__n_grids_each_dim) if self.__grid_solution_file is not None else None
				) # this already increase frame index for grid solution file
				self.__marginal_file.readline()
				return True
			except EOFError:
				self.__marginal_file.close()
				if self.__grid_solution_file is not None:
					self.__grid_solution_file.close()
				return False
		else:
			self.__marginal_file.close()
			if self.__grid_solution_file is not None:
				self.__grid_solution_file.close()
			return False
