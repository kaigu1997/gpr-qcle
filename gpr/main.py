#!/usr/bin/env python3
r"""main
====
The main module.
"""
import argparse
import datetime
import gc
import io
import os
import sys
import time
import traceback
import typing

import numpy as np
import numpy.typing as npt
import torch
import scipy.interpolate

import constant
import evolve
import expectation
import param
import pes
import plot

torch.set_default_dtype(constant.DTYPE)
torch.set_default_device(constant.DEVICE)


@typing.final
class Main:
	r"""All the quantities that could be used in further calculation

	Parameters
	----------
	to_draw : bool
		Whether to draw it or not
	grid_solution_file : str
		The file name for grid solution. If empty string, grid solution will not be read

	Methods
	-------
	total_ticks()
		The estimated total output times
	"""
	NUM_PTS: typing.Final = 256
	NUM_XTR_RATIO: typing.Final = 50
	NUM_MC_PTS: typing.Final = 1_000_000
	NUM_EVL_MC_PTS: typing.Final = 10_000
	__slots__: typing.Final[tuple] = ("__quantity", "__potential", "__grid_file", "__grid_data", "__grid_coord", "__pred_all_grids", "__pred_marginal", "__init_dist", "__predictors", "__mca", "__dm_drawer", "__wfn_plotter", "__pts_f", "__bln_f", "__all_f", "__mgn_f", "__ave_f", "__den_f", "__err_f", "__prm_f", "__scl_f", "__lss_f", "__start_time", "__end_time")
	__quantity: typing.Final[param.Quantity]
	__potential: typing.Final[pes.Potential]
	__grid_file: typing.Final[plot.FromFile | None]
	__grid_data: torch.Tensor | None
	__grid_coord: typing.Final[torch.Tensor | None]
	__pred_all_grids: typing.Final[torch.Tensor | None]
	__pred_marginal: typing.Final[tuple[torch.Tensor, ...]]
	__init_dist: typing.Final[pes.InitialDistribution]
	__predictors: typing.Final[expectation.GPRPredictors]
	__mca: typing.Final[expectation.MonteCarloAverage]
	__dm_drawer: typing.Final[plot.DensityMatrixDrawer | None]
	__wfn_plotter: typing.Final[plot.DensityMatrixMarginalPlotter | None]
	__pts_f: typing.Final[io.TextIOWrapper]
	__bln_f: typing.Final[io.TextIOWrapper]
	__all_f: typing.Final[io.TextIOWrapper]
	__mgn_f: typing.Final[io.TextIOWrapper]
	__ave_f: typing.Final[io.TextIOWrapper]
	__den_f: typing.Final[io.TextIOWrapper]
	__err_f: typing.Final[io.TextIOWrapper]
	__prm_f: typing.Final[io.TextIOWrapper]
	__scl_f: typing.Final[io.TextIOWrapper]
	__lss_f: typing.Final[io.TextIOWrapper]
	__start_time: typing.Final[int]
	__end_time: typing.Final[int | None]

	def __init__(
		self,
		to_draw: bool,
		grid_filename: str = ""
	) -> None:
		# get quantity and potential
		total_grids: int | None = None
		if grid_filename != "":
			try:
				total_grids = np.loadtxt(grid_filename, max_rows=1).size
			finally:
				pass
		self.__quantity = param.Quantity(total_grids=total_grids)
		self.__potential = pes.Potential(self.__quantity.model, self.__quantity.config)
		# compare with grid solution
		self.__grid_file = None
		self.__grid_data = None
		self.__grid_coord = None
		self.__pred_all_grids = None
		try:
			self.__grid_coord = torch.stack(torch.meshgrid(self.__quantity.r_grids_each_dim, indexing="xy"), 0).reshape(self.__quantity.config.PHASEDIM, -1).T
			self.__pred_all_grids = torch.empty((self.__quantity.config.NUM_PES, self.__quantity.config.NUM_PES) + tuple(int(n.item()) for n in self.__quantity.num_grids_on_each_dimension) * 2)
			if grid_filename != "":
				self.__grid_file = plot.FromFile(grid_filename, self.__quantity.config.NUM_ELM)
				self.__grid_data = torch.empty((self.__quantity.config.NUM_PES, self.__quantity.config.NUM_PES) + tuple(int(n.item()) for n in self.__quantity.num_grids_on_each_dimension) * 2)
		finally: # in case the memory requirement is too big
			if self.__pred_all_grids is None:
				self.__grid_file = None
				self.__grid_coord = None
			if self.__grid_file is None:
				self.__grid_data = None
		self.__pred_marginal = tuple(torch.empty(self.__quantity.config.NUM_PES, self.__quantity.config.NUM_PES, int(n.item())) for n in self.__quantity.num_grids_on_each_dimension) + tuple(torch.empty(self.__quantity.config.NUM_PES, self.__quantity.config.NUM_PES, int(n.item())) for n in self.__quantity.num_grids_on_each_dimension)
		# sampling. Initial point from gaussian directly; and the regressor
		self.__init_dist = pes.InitialDistribution(self.__potential, self.__quantity.r0, self.__quantity.sigma_r0, self.__quantity.init_ppl_and_phase)
		self.__predictors = expectation.GPRPredictors(
			self.__quantity.config,
			Main.NUM_PTS * Main.NUM_XTR_RATIO,
			self.__init_dist,
			torch.std(torch.cat([expectation.normal_sample(Main.NUM_XTR_RATIO, pt, self.__quantity.sigma_r0) for pt in expectation.normal_sample(Main.NUM_PTS, self.__quantity.r0, self.__quantity.sigma_r0)]), 0),
			Main.NUM_PTS,
			self.__quantity.sigma_r0,
			self.__potential,
			self.__quantity.mass,
		) # training of initial data included
		# average evaluators
		self.__mca = expectation.MonteCarloAverage(self.__quantity.config, Main.NUM_MC_PTS)
		# files for output
		self.__pts_f = open(constant.POINTS_FILENAME + constant.DATA_EXTENSION, "w", encoding=constant.ENC)
		self.__bln_f = open(constant.BELONGING_FILENAME + constant.DATA_EXTENSION, "w", encoding=constant.ENC)
		self.__all_f = open(constant.ALL_GRIDS_FILENAME + constant.DATA_EXTENSION, "w", encoding=constant.ENC)
		self.__mgn_f = open(constant.MARGINAL_FILENAME + constant.DATA_EXTENSION, "w", encoding=constant.ENC)
		self.__ave_f = open(constant.AVERAGE_FILENAME + constant.DATA_EXTENSION, "w", encoding=constant.ENC)
		self.__den_f = open("density" + constant.DATA_EXTENSION, "w", encoding=constant.ENC)
		self.__err_f = open(constant.ERROR_FILENAME + constant.DATA_EXTENSION, "w", encoding=constant.ENC)
		self.__prm_f = open(constant.PARAMETER_FILENAME + constant.DATA_EXTENSION, "w", encoding=constant.ENC)
		self.__scl_f = open(constant.SCALE_FILENAME + constant.DATA_EXTENSION, "w", encoding=constant.ENC)
		self.__lss_f = open(constant.LOSS_FILENAME + constant.DATA_EXTENSION, "w", encoding=constant.ENC)
		gc.collect()
		self.__predict_and_save_to_file(0)
		# drawer
		self.__dm_drawer = None
		self.__wfn_plotter = None
		if to_draw:
			if self.__quantity.config.PHASEDIM == plot.DIM_PLOT_DM and self.__pred_all_grids is not None: # None prediction means nothing provided for drawing
				self.__dm_drawer = plot.DensityMatrixDrawer(
					self.__quantity,
					True,
					True,
					True,
					self.__quantity.total_ticks,
					self.__pred_all_grids,
					self.__grid_data,
					[pt.detach().cpu().numpy() for pt in self.__predictors.inducing_points] + [pt.detach().cpu().numpy() for pt in self.__predictors.epmca.point_set],
					self.__predictors.scale.detach().cpu().numpy()
				)
			self.__wfn_plotter = plot.DensityMatrixMarginalPlotter(
				self.__quantity,
				False,
				self.__quantity.total_ticks,
				[m.detach().cpu().numpy() for m in self.__pred_marginal],
				self.__grid_data
			)
		self.__print_parameter_scale_loss()
		self.__start_time = int(time.time())
		self.__end_time = int(end_time) if (end_time := os.environ.get("SLURM_JOB_END_TIME")) is not None else None

	@property
	def total_ticks(self) -> int:
		r"""The estimated total output times

		Returns
		-------
		int
			The estimated total output times
		"""
		if self.__grid_file is not None and (gftt := self.__grid_file.total_ticks) != -1:
			return min(gftt, self.__quantity.total_ticks)
		else:
			return self.__quantity.total_ticks

	def __train(self, iTick: int, to_train: bool = True) -> None:
		r"""To train the parameter

		Parameters
		----------
		iTick : int
			Current time tick, to access grid data and for plotting
		to_train : bool, optional
			Whether to adjust parameters or not, by default True
		"""
		# get scale
		print(f"Tick {iTick}, {plot.format_array(self.__predictors.scale, "scales")}, {datetime.datetime.now()}", flush=constant.DEBUG_MODE)
		# save points
		ind_pts: typing.Final[torch.Tensor] = self.__predictors.inducing_points
		np.savetxt(self.__pts_f, torch.cat((ind_pts, self.__predictors.epmca.point_set), 1).reshape(-1, self.__quantity.config.PHASEDIM).T.detach().cpu().numpy(), constant.FMT, footer='\n', comments="", encoding=constant.ENC)
		print("", end="", file=self.__pts_f, flush=constant.DEBUG_MODE)
		np.savetxt(
			self.__bln_f,
			np.repeat(np.concatenate([[idx, idx + self.__quantity.config.NUM_ELM] for idx in self.__quantity.config.TRIL_ELEMENT_INDICES]), np.concatenate([[ind_pts.shape[1], self.__predictors.epmca.point_set.shape[1]] * self.__quantity.config.NUM_TRIG]))[np.newaxis],
			footer="\n",
			comments="",
			encoding=constant.ENC
		)
		print("", end="", file=self.__bln_f, flush=constant.DEBUG_MODE)
		# fit
		if to_train:
			self.__predictors.save_train(self.__potential, self.__quantity.mass)

	def __predict_and_save_to_file(self, iTick: int) -> None:
		r"""To make predictions on marginal/grids

		Parameters
		----------
		iTick : int
			Current time tick, to access grid data and for plotting

		Raises
		------
		NotImplementedError
			In case an unimplemented branch is reached
		"""
		# read from grid
		if self.__grid_file is not None and self.__grid_data is not None and self.__grid_file.have_content:
			try:
				self.__grid_data = plot.file_data_to_dm(self.__grid_file, self.__quantity.config, [int(n.item()) for n in self.__quantity.num_grids_on_each_dimension])
				self.__grid_data = self.__grid_data.reshape(self.__quantity.config.NUM_ELM, *self.__grid_data.shape[:-2])
			except EOFError:
				self.__grid_data = None
		# predict and marginal distribution
		success_pred_all: bool = self.__pred_all_grids is not None
		for iPES, jPES, iElement in zip(self.__quantity.config.TRIL_ROW_INDICES, self.__quantity.config.TRIL_COL_INDICES, self.__quantity.config.TRIL_ELEMENT_INDICES):
			pred_element: torch.Tensor | None = None
			if self.__grid_coord is not None and self.__pred_all_grids is not None and success_pred_all:
				try:
					pred_element = self.__predictors.predict(self.__grid_coord, iElement).reshape(self.__pred_all_grids.shape[2:])
				finally:
					success_pred_all = success_pred_all and pred_element is not None
			if iPES == jPES:
				if self.__pred_all_grids is not None and pred_element is not None:
					self.__pred_all_grids[iPES, jPES] = pred_element.real
				for iDim in self.__quantity.config.PHASEDIM_RANGE:
					self.__pred_marginal[iDim][iPES, jPES, :] = self.__predictors.get_marginal(self.__quantity.r_grids_each_dim[iDim].reshape(-1, 1), iElement, iDim).real
			else:
				if self.__pred_all_grids is not None and pred_element is not None:
					self.__pred_all_grids[jPES, iPES] = pred_element.real
					self.__pred_all_grids[iPES, jPES] = pred_element.imag
				for iDim in self.__quantity.config.PHASEDIM_RANGE:
					marginal_pred_element: torch.Tensor = self.__predictors.get_marginal(self.__quantity.r_grids_each_dim[iDim].reshape(-1, 1), iElement, iDim)
					self.__pred_marginal[iDim][jPES, iPES, :] = marginal_pred_element.real
					self.__pred_marginal[iDim][iPES, jPES, :] = marginal_pred_element.imag
		if self.__pred_all_grids is not None and success_pred_all:
			np.savetxt(self.__all_f, self.__pred_all_grids.reshape(self.__quantity.config.NUM_ELM, -1).detach().cpu().numpy(), constant.FMT, footer="\n", comments="", encoding=constant.ENC)
			print("", end="", file=self.__all_f, flush=constant.DEBUG_MODE)
		for iDim in self.__quantity.config.PHASEDIM_RANGE:
			np.savetxt(self.__mgn_f, self.__pred_marginal[iDim].reshape(self.__quantity.config.NUM_ELM, -1).detach().cpu().numpy(), constant.FMT, comments="", encoding=constant.ENC)
		print("", end="", file=self.__mgn_f, flush=constant.DEBUG_MODE)
		# calculate averages
		print(iTick * self.__quantity.output_interval, end=" ", file=self.__ave_f)
		self.__mca.update_pts([*self.__predictors.epmca.point_set], self.__predictors.predict)
		aver: expectation.Averager
		for aver in [self.__mca, self.__predictors, self.__predictors.epmca]:
			print(
				*aver.population().detach().cpu().numpy(),
				*aver.coordinates().detach().cpu().numpy(),
				*aver.covariance()[np.tril_indices(self.__quantity.config.PHASEDIM)].detach().cpu().numpy(),
				aver.potential(self.__potential),
				aver.kinetic(self.__quantity.mass),
				*aver.purity().reshape(-1).detach().cpu().numpy(),
				end=" ",
				file=self.__ave_f
			)
		print("", file=self.__ave_f, flush=constant.DEBUG_MODE)
		# calculate error, and predict density
		evolving_density: typing.Final[list[torch.Tensor]] = [*self.__predictors.epmca.density]
		if self.__grid_data is not None: # interpolate the data
			grid_interpolate: list[npt.NDArray[np.cdouble]] = [np.array([], dtype=np.cdouble) for _ in self.__quantity.config.TRIG_RANGE]
			evolving_errors: npt.NDArray[np.double] = np.empty((self.__quantity.config.NUM_PES, self.__quantity.config.NUM_PES), np.double)
			for iTrig, (iPES, jPES) in enumerate(zip(self.__quantity.config.TRIL_ROW_INDICES, self.__quantity.config.TRIL_COL_INDICES)):
				interpolator_re: scipy.interpolate.RegularGridInterpolator = scipy.interpolate.RegularGridInterpolator(
					tuple(grid.detach().cpu().numpy() for grid in self.__quantity.r_grids_each_dim),
					self.__grid_data[jPES * self.__quantity.config.NUM_PES + iPES].detach().cpu().numpy(), # upper part
					"cubic",
					False,
					0.0
				)
				grid_interpolate[iTrig] = interpolator_re(self.__predictors.epmca.point_set[iTrig].detach().cpu().numpy()).astype(np.cdouble)
				if iPES == jPES:
					evolving_errors[iPES, jPES] = np.sum((grid_interpolate[iTrig].real - evolving_density[iTrig].real.detach().cpu().numpy()) ** 2) / evolving_density[iTrig].numel()
				else:
					interpolator_im: scipy.interpolate.RegularGridInterpolator = scipy.interpolate.RegularGridInterpolator(
						tuple(grid.detach().cpu().numpy() for grid in self.__quantity.r_grids_each_dim),
						self.__grid_data[iPES * self.__quantity.config.NUM_PES + jPES].detach().cpu().numpy(), # strictly lower part
						"cubic",
						False,
						0.0
					)
					grid_interpolate[iTrig].imag = interpolator_im(self.__predictors.epmca.point_set[iTrig].detach().cpu().numpy())
					evolving_errors[jPES, iPES] = np.sum((grid_interpolate[iTrig].real - evolving_density[iTrig].real.detach().cpu().numpy()) ** 2) / evolving_density[iTrig].numel()
					evolving_errors[iPES, jPES] = np.sum((grid_interpolate[iTrig].imag - evolving_density[iTrig].imag.detach().cpu().numpy()) ** 2) / evolving_density[iTrig].numel()
			evolving_errors = evolving_errors.reshape(-1)
			assert self.__pred_all_grids is not None # grid data will only be read when already enough space for prediction
			diff: typing.Final[torch.Tensor] = (self.__pred_all_grids.reshape(self.__quantity.config.NUM_ELM, *self.__pred_all_grids.shape[2:]) - self.__grid_data).moveaxis(0, -1)
			original_errors: typing.Final[torch.Tensor] = torch.sum(diff ** 2, self.__quantity.config.PHASEDIM_RANGE)
			rescaled_errors: typing.Final[torch.Tensor] = original_errors * self.__predictors.scale ** 2
			np.savetxt(self.__err_f, (original_errors.detach().cpu().numpy(), rescaled_errors.detach().cpu().numpy(), evolving_errors), footer="\n", comments="", encoding=constant.ENC)
			print("", end="", file=self.__err_f, flush=constant.DEBUG_MODE)
			np.savetxt(self.__den_f, np.concatenate(grid_interpolate).view(np.double).reshape(-1, 2).T, constant.FMT, comments="", encoding=constant.ENC) # 2 stands for real and imag
		else:
			np.savetxt(self.__den_f, torch.cat(evolving_density).detach().cpu().numpy().view(np.double).reshape(-1, 2).T, constant.FMT, comments="", encoding=constant.ENC)
		np.savetxt(self.__den_f, torch.cat(evolving_density).detach().cpu().numpy().view(np.double).reshape(-1, 2).T, constant.FMT, comments="", encoding=constant.ENC)
		np.savetxt(self.__den_f, torch.cat([self.__predictors.predict(pt, idx) for pt, idx in zip(self.__predictors.epmca.point_set, self.__quantity.config.TRIL_ELEMENT_INDICES)]).detach().cpu().numpy().view(np.double).reshape(-1, 2).T, constant.FMT, footer="\n", comments="", encoding=constant.ENC)
		print("", end="", file=self.__den_f, flush=constant.DEBUG_MODE)

	def __draw(self, iTick: int) -> None:
		r"""To draw the PWTDM and marginals if required

		Parameters
		----------
		iTick : int
			Current time tick, to access grid data and for plotting
		"""
		if self.__dm_drawer is not None and self.__pred_all_grids is not None:
			self.__dm_drawer(
				iTick,
				self.__pred_all_grids,
				self.__grid_data,
				[pt.detach().cpu().numpy() for pt in self.__predictors.inducing_points] + [pt.detach().cpu().numpy() for pt in self.__predictors.epmca.point_set],
				self.__predictors.scale.detach().cpu().numpy()
			)
		if self.__wfn_plotter is not None:
			self.__wfn_plotter(iTick, [m.detach().cpu().numpy() for m in self.__pred_marginal], self.__grid_data)

	def __print_parameter_scale_loss(self) -> None:
		r"""To print parameters, rescale factor, and loss on point points to file
		"""
		self.__predictors.print(self.__prm_f)
		print("\n", file=self.__prm_f, flush=constant.DEBUG_MODE)
		np.savetxt(self.__scl_f, self.__predictors.scale.detach().cpu().numpy(), constant.FMT, footer="\n", comments="", encoding=constant.ENC)
		for i in self.__quantity.config.ELEMENT_RANGE:
			print(self.__predictors[i].error().item(), file=self.__lss_f)
		print("\n", file=self.__lss_f, flush=constant.DEBUG_MODE)

	def __call__(self, iTick: int) -> bool:
		r"""To do evolution, train parameters, and output

		Parameters
		----------
		iTick : int
			Current time tick, to access grid data and for plotting

		Returns
		-------
		bool
			Whether to stop evolution or not
		"""
		# evolve
		for _ in range(self.__quantity.output_ticks):
			# here num_dt is the number of steps used in the back-propagator
			self.__predictors.evolve_update(self.__potential, self.__quantity.mass, self.__quantity.dt)
			self.__print_parameter_scale_loss()
		# update and predict
		self.__train(iTick, iTick % self.__quantity.reopt_ticks == 0)
		self.__predict_and_save_to_file(iTick)
		self.__draw(iTick)
		self.__print_parameter_scale_loss()
		# check stopping criteria, when grid solution is not given
		# use predictors (aia) with old points
		if self.__grid_data is None and torch.any(self.__predictors.epmca.coordinates()[:self.__quantity.config.DIM] > torch.abs(self.__quantity.x0)).item():
			return True
		if self.__end_time is not None:
			current_time: int = int(time.time())
			time_pass: int = current_time - self.__start_time
			time_left: int = self.__end_time - current_time
			if time_left < time_pass // iTick:
				# time left is not enough for next output, kill and rerun the job
				print(f"Time left is {time_left} seconds, not enough for another iteration. Stop evolving after {time_pass} seconds, {iTick} iterations")
				return True
		return False

	def finalize(self) -> None:
		r"""To close files, draw time-dependent changes, and generate animation and tar files
		"""
		# close files
		for f in (self.__pts_f, self.__bln_f, self.__all_f, self.__mgn_f, self.__ave_f, self.__den_f, self.__err_f, self.__prm_f, self.__scl_f, self.__lss_f):
			f.close()
		# plot
		ticks: typing.Final[npt.NDArray[np.double]] = plot.plot_average(self.__quantity.config) # averages
		total_ticks = ticks.size
		if self.__grid_data is not None: # error
			plot.plot_error(self.__quantity.config, ticks)
		# parameters, loss and rescale factor
		param_loss_scale_ticks: typing.Final[npt.NDArray[np.double]] = np.concatenate([np.arange(i * self.__quantity.output_ticks, (i + 1) * self.__quantity.output_ticks + 1) for i in range(total_ticks + 1)]) * np.double(self.__quantity.dt) # This should be size of (total_ticks - 1) * (output_steps + 1)
		plot.plot_parameters(self.__quantity.config, param_loss_scale_ticks)
		plot.plot_loss_and_rescale_factors(self.__quantity.config, param_loss_scale_ticks)
		# tar figures
		if self.__dm_drawer is not None:
			# draw frame by frame and combine into a tarfile, removing the pics after tarfile successfully constructed
			plot.tar_files(self.__dm_drawer.picname, total_ticks)
		if self.__wfn_plotter is not None:
			plot.tar_files(self.__wfn_plotter.picname, total_ticks)


def parse_argument() -> tuple[bool, str]:
	r"""To parse arguments

	Returns
	-------
	tuple[bool, str]
		Whether to plot or not, and whether to read grid solution or not
	"""
	parser: argparse.ArgumentParser = argparse.ArgumentParser(description="To evolve the grid solution")
	parser.add_argument("--plot", "-p", action="store_true", help="Whether to plot each frame or not")
	parser.add_argument("--read", "-r", default="", type=str, help="Whether to read grid solution or not; if read, provide the file name")
	result: dict[str, typing.Any] = vars(parser.parse_args())
	return result["plot"], result["read"]


def main(to_draw: bool, grid_solution_file: str) -> None:
	r"""The main routine

	Parameters
	----------
	to_draw : bool
		Whether to draw it or not
	grid_solution_file : str
		The file name for grid solution. If empty string, grid solution will not be read

	Raises
	------
	NotImplementedError
		In case an unimplemented branch is reached
	"""
	exe: typing.Final = Main(to_draw, grid_solution_file)
	try:
		for iTick in range(1, exe.total_ticks):
			if exe(iTick):
				break
	except Exception as e:
		traceback.print_exception(e, file=sys.stdout)
	finally:
		exe.finalize()


if __name__ == "__main__":
	main(*parse_argument())
