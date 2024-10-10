import math
import os
import sys
import typing

import gpytorch
import matplotlib.axes
import matplotlib.cm
import matplotlib.colors
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch

sys.path.append(os.path.dirname(__file__))

import gp
import opt
import sample

torch.set_default_dtype(torch.float64)
FIGSIZE: list[float] = matplotlib.rcParams["figure.figsize"]


class PNLogNorm(matplotlib.colors.Normalize):
	__slots__ = ("__abs_min", "__abs_max", "__log2_abs_min", "__log2_abs_max", "__log2_diff")

	def __init__(self, abs_min: float = sys.float_info.min, abs_max: float = sys.float_info.max, clip: bool = False) -> None:
		assert 0 < abs_min < abs_max
		super().__init__(-abs_max, abs_max, clip)
		self.__abs_min: float = abs_min
		self.__abs_max: float = abs_max
		self.__log2_abs_min: float = math.log2(abs_min)
		self.__log2_abs_max: float = math.log2(abs_max)
		self.__log2_diff: float = self.__log2_abs_max - self.__log2_abs_min

	@property
	def abs_min(self):
		return self.__abs_min

	@property
	def abs_max(self):
		return self.__abs_max

	def __call__(self, value: typing.Any, clip: bool | None = None) -> float | npt.NDArray[np.double]:
		def process_single_value(value: float, clip: bool) -> float:
			if abs(value) <= self.__abs_min:
				return 0.5
			sgn_half: float = 0.5 if value > 0 else -0.5
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

	def __repr__(self) -> str:
		return __class__.__name__ + "({}, {})".format(self.__abs_min, self.__abs_max)
		
	def inverse(self, value: typing.Any) -> float | npt.NDArray[np.double]:
		def process_single_value(value: float) -> float:
			if value == 0.5:
				return 0.0
			return (-1.0 if value < 0.5 else 1.0) * math.exp2(abs(value - 0.5) * 2 * self.__log2_diff + self.__log2_abs_min)

		result, is_scalar = super().process_value(value)
		if is_scalar == 1:
			return process_single_value(result.data.ravel()[0])
		else:
			return np.ma.array(np.vectorize(process_single_value, otypes=[float])(result.data), mask=result.mask)


def get_centered_norm_and_levels(abs_max: float, cmap: str | matplotlib.colors.Colormap) -> tuple[matplotlib.colors.CenteredNorm, npt.NDArray[np.double], npt.NDArray[np.double]]:
	abs_max = abs(abs_max)
	if isinstance(cmap, str):
		cmap = matplotlib.colormaps[cmap]
	abs_max_10_level: float = math.pow(10, math.floor(math.log10(abs_max)))
	centered_norm: matplotlib.colors.CenteredNorm = matplotlib.colors.CenteredNorm(0.0, math.ceil(abs_max / abs_max_10_level) * abs_max_10_level, True)
	centered_levels: npt.NDArray[np.double] = np.linspace(-centered_norm.halfrange, centered_norm.halfrange, cmap.N // 8 * 8 + 1, True)
	return centered_norm, centered_levels, centered_levels[::cmap.N // 8]


def get_posneg_log_norm_and_levels(abs_min: float, abs_max: float, cmap: str | matplotlib.colors.Colormap) -> tuple[PNLogNorm, npt.NDArray[np.double], npt.NDArray[np.double]]:
	abs_min = abs(abs_min)
	abs_max = abs(abs_max)
	if abs_max == 0:
		abs_max = sys.float_info.min * 2
	if abs_min == 0:
		abs_min = sys.float_info.min
	abs_min = max(abs_min, abs_max * sys.float_info.epsilon)
	lb_log10: int = int(math.floor(math.log10(abs_min)))
	ub_log10: int = int(math.ceil(math.log10(abs_max)))
	lb_log10 += (ub_log10 - lb_log10) % 4
	if isinstance(cmap, str):
		cmap = matplotlib.colormaps[cmap]
	num_ticks: int = cmap.N // 8 * 8 + 3
	pn_log_norm: matplotlib.colors.Normalize = PNLogNorm(math.pow(10, lb_log10), math.pow(10, ub_log10), True)
	pn_log_levels: npt.NDArray[np.double] = np.zeros(num_ticks)
	pn_log_levels[cmap.N // 8 * 4 + 2:] = np.exp2(np.linspace(math.log2(pn_log_norm.abs_min), math.log2(pn_log_norm.abs_max), cmap.N // 8 * 4 + 1, True))
	pn_log_levels[cmap.N // 8 * 4::-1] = -pn_log_levels[cmap.N // 8 * 4 + 2:]
	return pn_log_norm, pn_log_levels, np.concatenate([pn_log_levels[0:cmap.N // 8 * 4:cmap.N // 8], pn_log_levels[cmap.N // 8 * 4 + 1:cmap.N // 8 * 4 + 2], pn_log_levels[cmap.N // 8 * 5 + 2::cmap.N // 8]])


def limited_region(arr2d: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
	return arr2d[sample.distribution.N_GRIDS // 8 * 3:sample.distribution.N_GRIDS // 8 * 5 + 1, sample.distribution.N_GRIDS // 8 * 3:sample.distribution.N_GRIDS // 8 * 5 + 1]


class PlotConstants:
	CMAP = "seismic"
	COLOR_RATIO: float = 1.1

	def __init__(self, y: npt.NDArray[np.double]) -> None:
		self.label: npt.NDArray[np.double] = y
		self.ABS_MAX_LIMIT: float = np.max(np.abs(y))
		self.ABS_MIN_LIMIT: float = np.min(np.abs(y[y != 0.0]))
		self.CTR_NORM: matplotlib.colors.Normalize
		self.CTR_LEVELS: npt.NDArray[np.double]
		self.CTR_TICKS: npt.NDArray[np.double]
		self.CTR_NORM, self.CTR_LEVELS, self.CTR_TICKS = get_centered_norm_and_levels(self.ABS_MAX_LIMIT * __class__.COLOR_RATIO, __class__.CMAP)
		self.LOG_NORM: matplotlib.colors.Normalize
		self.LOG_LEVELS: npt.NDArray[np.double]
		self.LOG_TICKS: npt.NDArray[np.double]
		self.LOG_NORM, self.LOG_LEVELS, self.LOG_TICKS = get_posneg_log_norm_and_levels(self.ABS_MIN_LIMIT, self.ABS_MAX_LIMIT, __class__.CMAP)

	def draw_function(self) -> None:
		fig: matplotlib.figure.Figure = plt.figure(figsize=(FIGSIZE[0] * 3, FIGSIZE[1]))
		ax0, ax1, ax2 = fig.subplots(ncols=3)
		ax0.contourf(limited_region(sample.distribution.x1), limited_region(sample.distribution.x2), limited_region(self.label), self.CTR_LEVELS, cmap=__class__.CMAP, norm=self.CTR_NORM)
		# ax0.scatter(CENTER[:, 0], CENTER[:, 1], 20 * np.abs(WEIGHT), 'black')
		ax1.contourf(sample.distribution.x1, sample.distribution.x2, self.label, self.CTR_LEVELS, cmap=__class__.CMAP, norm=self.CTR_NORM)
		# ax1.scatter(CENTER[:, 0], CENTER[:, 1], 10, 'black')
		fig.colorbar(matplotlib.cm.ScalarMappable(self.CTR_NORM, __class__.CMAP), ax=[ax0, ax1], ticks=self.CTR_TICKS)
		ax2.contourf(sample.distribution.x1, sample.distribution.x2, self.label, self.LOG_LEVELS, cmap=__class__.CMAP, norm=self.LOG_NORM)
		fig.colorbar(matplotlib.cm.ScalarMappable(self.LOG_NORM, __class__.CMAP), ax=ax2, ticks=self.LOG_TICKS, format="%+.1e")
		fig.savefig("distribution.png")
		plt.close(fig)

	def draw_sample_pts(self, central_pts: npt.NDArray[np.double], extra_pts: npt.NDArray[np.double], number: int) -> None:
		fig: matplotlib.figure.Figure = plt.figure(figsize=(FIGSIZE[0] * 3, FIGSIZE[1]))
		ax0, ax1, ax2 = fig.subplots(ncols=3)
		ax0.contourf(limited_region(sample.distribution.x1), limited_region(sample.distribution.x2), limited_region(self.label), self.CTR_LEVELS, cmap=__class__.CMAP, norm=self.CTR_NORM)
		ax0.scatter(np.clip(extra_pts[:, 0], sample.distribution.MIN / 4, sample.distribution.MAX / 4), np.clip(extra_pts[:, 1], sample.distribution.MIN / 4, sample.distribution.MAX / 4), 1, 'green')
		ax0.scatter(np.clip(central_pts[:, 0], sample.distribution.MIN / 4, sample.distribution.MAX / 4), np.clip(central_pts[:, 1], sample.distribution.MIN / 4, sample.distribution.MAX / 4), 10, 'black')
		ax1.contourf(sample.distribution.x1, sample.distribution.x2, self.label, self.CTR_LEVELS, cmap=__class__.CMAP, norm=self.CTR_NORM)
		fig.colorbar(matplotlib.cm.ScalarMappable(self.CTR_NORM, __class__.CMAP), ax=[ax0, ax1], ticks=self.CTR_TICKS)
		ax2.contourf(sample.distribution.x1, sample.distribution.x2, self.label, self.LOG_LEVELS, cmap=__class__.CMAP, norm=self.LOG_NORM)
		fig.colorbar(matplotlib.cm.ScalarMappable(self.LOG_NORM, __class__.CMAP), ax=ax2, ticks=self.LOG_TICKS, format="%.1e")
		for ax in [ax1, ax2]:
			ax.scatter(extra_pts[:, 0], extra_pts[:, 1], 1, 'green')
			ax.scatter(central_pts[:, 0], central_pts[:, 1], 10, 'black')
		fig.savefig("sample_points_{}.png".format(number))
		plt.close(fig)


def plot(
	pred: npt.NDArray[np.double],
	pc: PlotConstants,
	pts: sample.data,
	name: str,
	plot_extra: bool = True
) -> None:
	diff: npt.NDArray[np.double] = pred - pc.label
	mse: float = float(np.average(diff ** 2))
	mae: float = float(np.average(np.abs(diff)))
	max_e: float = np.max(np.abs(diff))
	print("Mean squared error =", mse)
	print("Mean absolute error =", mae)
	print("Maximum error =", max_e)
	if (mse_contribute := max_e ** 2 / (mse * pc.label.size) ) > 0.01:
		print("Maximum error point contribute {:.2f}% of MSE".format(mse_contribute * 100))
	if (mae_contribute := max_e / (mae * pc.label.size)) > 0.01:
		print("Maximum error point contribute {:.2f}% of MAE".format(mae_contribute * 100))
	print("", end="", flush=True)
	max_ind: tuple = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
	fig: matplotlib.figure.Figure = plt.figure(figsize=(FIGSIZE[0] * 3, FIGSIZE[1] * 4))
	axs: np.ndarray = fig.subplots(4, 3)
	titles: tuple = ("Exact Density", "Density Fit")
	data: tuple = (pc.label, pred)
	for i in range(2):
		axs[0, i].set_title(titles[i])
		axs[0, i].contourf(limited_region(sample.distribution.x1), limited_region(sample.distribution.x2), limited_region(data[i]), pc.CTR_LEVELS, cmap=PlotConstants.CMAP, norm=pc.CTR_NORM)
		axs[1, i].contourf(sample.distribution.x1, sample.distribution.x2, data[i], pc.CTR_LEVELS, cmap=PlotConstants.CMAP, norm=pc.CTR_NORM)
		axs[2, i].contourf(sample.distribution.x1, sample.distribution.x2, data[i], pc.CTR_LEVELS, cmap=PlotConstants.CMAP, norm=pc.CTR_NORM)
		if plot_extra:
			axs[2, i].scatter(pts.extra_x[:, 0], pts.extra_x[:, 1], 1, 'green')
			axs[2, i].scatter(pts.x[:, 0], pts.x[:, 1], 10, 'black')
		else:
			axs[2, i].scatter(pts.x[:, 0], pts.x[:, 1], 1, 'black')
		axs[3, i].contourf(sample.distribution.x1, sample.distribution.x2, data[i], pc.LOG_LEVELS, cmap=PlotConstants.CMAP, norm=pc.LOG_NORM)
	fig.colorbar(matplotlib.cm.ScalarMappable(pc.CTR_NORM, PlotConstants.CMAP), ax=axs[:3, :2].ravel().tolist(), ticks=pc.CTR_TICKS)
	fig.colorbar(matplotlib.cm.ScalarMappable(pc.LOG_NORM, PlotConstants.CMAP), ax=axs[3, :2].tolist(), ticks=pc.LOG_TICKS, format="%.1e")

	ctr_norm, ctr_levels, ctr_ticks = get_centered_norm_and_levels(np.max(np.abs(diff)) * PlotConstants.COLOR_RATIO, PlotConstants.CMAP)
	pn_log_norm, pn_log_levels, pn_log_ticks = get_posneg_log_norm_and_levels(np.min(np.abs(diff[diff != 0])), np.max(np.abs(diff)), PlotConstants.CMAP)
	axs[0, 2].set_title("Error\nMaximum at ({:.2f}, {:.2f})".format(sample.distribution.x1[max_ind], sample.distribution.x2[max_ind]))
	axs[0, 2].contourf(limited_region(sample.distribution.x1), limited_region(sample.distribution.x2), limited_region(diff), ctr_levels, cmap=PlotConstants.CMAP, norm=ctr_norm)
	axs[0, 2].scatter(np.clip(sample.distribution.x1[max_ind], sample.distribution.MIN / 4, sample.distribution.MAX / 4), np.clip(sample.distribution.x2[max_ind], sample.distribution.MIN / 4, sample.distribution.MAX / 4), 10, 'black')
	axs[1, 2].contourf(sample.distribution.x1, sample.distribution.x2, diff, ctr_levels, cmap=PlotConstants.CMAP, norm=ctr_norm)
	axs[1, 2].scatter(sample.distribution.x1[max_ind], sample.distribution.x2[max_ind], 10, 'black')
	axs[2, 2].contourf(sample.distribution.x1, sample.distribution.x2, diff, ctr_levels, cmap=PlotConstants.CMAP, norm=ctr_norm)
	if plot_extra:
		axs[2, 2].scatter(pts.extra_x[:, 0], pts.extra_x[:, 1], 1, 'green')
		axs[2, 2].scatter(pts.x[:, 0], pts.x[:, 1], 10, 'black')
	else:
		axs[2, 2].scatter(pts.x[:, 0], pts.x[:, 1], 1, 'black')
	axs[3, 2].contourf(sample.distribution.x1, sample.distribution.x2, diff, pn_log_levels, cmap=PlotConstants.CMAP, norm=pn_log_norm)
	fig.colorbar(matplotlib.cm.ScalarMappable(ctr_norm, PlotConstants.CMAP), ax=axs[:3, 2].tolist(), ticks=ctr_ticks)
	fig.colorbar(matplotlib.cm.ScalarMappable(pn_log_norm, PlotConstants.CMAP), ax=axs[3, 2], ticks=pn_log_ticks, format="%.1e")
	fig.savefig(name + ".png")
	plt.close(fig)


def plot_lengthscale_error(
	name: str,
	pc: PlotConstants,
	pts: sample.data,
	erf: opt.LossFuncType,
	min: float | npt.NDArray[np.double] = 0.25,
	max: float | npt.NDArray[np.double] = 1.25,
	n_grids_to_plot: int = 101,
	predictor: gp.PredType | None = None,
	**kwargs
) -> gp.GPR:
	NOISE: float = float(gpytorch.settings.min_fixed_noise.value(torch.double) or 1e-8)
	if not isinstance(min, np.ndarray):
		min = np.full(2, min, np.double)
	if not isinstance(max, np.ndarray):
		max = np.full(2, max, np.double)
	xv: npt.NDArray[np.double]
	pv: npt.NDArray[np.double]
	xv, pv = np.meshgrid(np.linspace(min[0], max[0], n_grids_to_plot), np.linspace(min[1], max[1], n_grids_to_plot))
	lengths: npt.NDArray[np.double] = np.concatenate((xv[..., np.newaxis], pv[..., np.newaxis]), axis=-1)
	values: npt.NDArray[np.double] = np.zeros((3, n_grids_to_plot, n_grids_to_plot))
	for i in range(n_grids_to_plot):
		for j in range(n_grids_to_plot):
			diff: npt.NDArray[np.double] = gp.gpytorch_gpr(
				gp.GPR(
					pts.x_t,
					pts.y_t,
					gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.full(pts.x.shape[:-1], NOISE)),
					gpytorch.kernels.RBFKernel(sample.distribution.DIM, lengthscale_constraint=gp.NoConstraint(torch.from_numpy(lengths[i, j])))
				).eval(),
				pts,
				predictor=predictor,
				**kwargs
			) - pts.y_test
			values[0, i, j] = np.average(diff ** 2)
			values[1, i, j] = np.average(np.abs(diff))
			values[2, i, j] = np.max(np.abs(diff))
	fig, axs = plt.subplots(1, 3, figsize=(FIGSIZE[0] * 3, FIGSIZE[1]))
	titles: list[str] = ['Mean Squared Error', 'Mean Absolute Error', 'Maximum Error']
	for i in range(3):
		ax: matplotlib.axes.Axes = axs[i]
		lg_lb: float = np.floor(np.log10(np.min(values[i])))
		lg_ub: float = np.ceil(np.log10(np.max(values[i])))
		norm: matplotlib.colors.LogNorm = matplotlib.colors.LogNorm(np.power(10.0, lg_lb), np.power(10.0, lg_ub), True)
		levels: npt.NDArray[np.double] = np.power(10.0, np.linspace(lg_lb, lg_ub, matplotlib.colormaps[PlotConstants.CMAP].N, True))
		ax.contourf(xv, pv, values[i], levels, cmap=PlotConstants.CMAP, norm=norm)
		argmin_ind = np.unravel_index(np.argmin(values[i]), values[i].shape)
		ax.set_title(titles[i] + '\nMinimum at {}'.format(lengths[argmin_ind]) + '\nMinimum is {}'.format(values[i][argmin_ind]))
		ax.scatter(lengths[argmin_ind][0], lengths[argmin_ind][1], 10, "black")
		ax.set_xlabel('Length of x')
		ax.set_ylabel('Length of p')
		fig.colorbar(matplotlib.cm.ScalarMappable(cmap=PlotConstants.CMAP, norm=norm), ax=ax, ticks=np.power(10.0, np.linspace(lg_lb, lg_ub, int(lg_ub - lg_lb) + 1, True)), format="%.1e")
	fig.savefig("length_{}.png".format(name))
	plt.close(fig)
	del fig
	model = opt.gpytorch_train(
		pts.x_t,
		pts.y_t,
		gpytorch.kernels.RBFKernel(sample.distribution.DIM),
		erf,
		initial_values=[torch.from_numpy(lengths[np.unravel_index(np.argmin(values[0]), values[0].shape)])],
		# initial_values=[torch.from_numpy(sample.distribution.WIDTH)],
		initial_value_search=False,
		pred=predictor,
		**kwargs
	)
	assert model is not None
	plot(
		gp.gpytorch_gpr(model, pts, predictor=predictor, **kwargs),
		pc,
		pts,
		"opt_{}".format(name)
	)
	return model
