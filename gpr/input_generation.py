#!/usr/bin/env python3
r"""input_generation
================
This module generates input file.
"""
import argparse
import collections.abc
import typing

import numpy as np
import numpy.typing as npt

import pes
import plot


def main(arguments: None | collections.abc.Sequence[str] = None) -> None:
	r"""
	The main routine

	Parameters
	----------
	arguments : None | collections.abc.Sequence[str], optional
		The arguments to parse, by default None
	"""
	model_parser: typing.Final[argparse.ArgumentParser] = argparse.ArgumentParser(description="Provide model to use", add_help=False)
	model_parser.add_argument("--model", default="dac", type=str, choices=pes.MODEL_DICT.keys(), help="Model to use")
	args, remaining_args = model_parser.parse_known_args(arguments)
	config: typing.Final[pes.ModelConfig] = pes.ModelConfig(pes.MODEL_DICT[args.model])

	parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Generate input file.", parents=[model_parser])
	parser.add_argument("-m", "--mass", nargs=config.DIM, default=[2000.0] * config.DIM, type=float, help="Mass ")
	parser.add_argument("x0", nargs=config.DIM, type=float, help="Initial positions")
	parser.add_argument("p0", nargs=config.DIM, type=float, help="Initial momenta")
	parser.add_argument("--sigma_p0", nargs=config.DIM, type=float, help="Initial standard deviations of momentum, by default 1/20 of initial momentum")
	parser.add_argument("--dx", nargs=config.DIM, default=[0.1] * config.DIM, type=float, help="Grid spacing in configurational space (for positions)")
	parser.add_argument("--population", "--ppl", nargs=config.NUM_PES, default=[1.0] + [0.0] * (config.NUM_PES - 1), type=float, help="Initial population. The squared sum will be normalized.")
	parser.add_argument("--phase-factor", nargs=config.NUM_PES, default=[0.0] * config.NUM_PES, type=float, help="Initial phase factor in degree.")
	parser.add_argument("output_interval", type=float, help="Interval between outputs (in a.u.)")
	parser.add_argument("--reoptimization-interval", "--reopt", "-r", default=-1.0, type=float, help="Interval between parameter optimization (in a.u.)")
	parser.add_argument("--dt", default=0.1, type=float, help="Time step")
	result: typing.Final[argparse.Namespace] = parser.parse_args(remaining_args)
	mass: typing.Final[npt.NDArray[np.double]] = np.array(result.mass, np.double)
	x0: typing.Final[npt.NDArray[np.double]] = np.array(result.x0, np.double)
	p0: typing.Final[npt.NDArray[np.double]] = np.array(result.p0, np.double)
	sigma_p0: typing.Final[npt.NDArray[np.double]] = p0 / 20.0 if result.sigma_p0 is None else np.array(result.sigma_p0, np.double)
	dx: typing.Final[npt.NDArray[np.double]] = np.array(result.dx, np.double)
	population: typing.Final[npt.NDArray[np.double]] = np.array(result.population, np.double)
	phase_factor: typing.Final[npt.NDArray[np.double]] = np.array(result.phase_factor, np.double)
	dt: typing.Final[float] = abs(float(result.dt))
	output_interval: typing.Final[float] = max(abs(float(result.output_interval)), dt)
	reoptimization_interval: typing.Final[float] = max(abs(float(result.reoptimization_interval)), output_interval)
	with open("input", "w", encoding="UTF-8") as f:
		content: typing.Final = f"""JobType: (choose from: se_diag, se_fft, qcle, gpr)
gpr
model: (choose from: {", ".join(pes.MODEL_DICT.keys())})
{config.NAME}
mass:
{plot.format_array(mass)}
x0:
{plot.format_array(x0)}
p0:
{plot.format_array(p0)}
sigma_p0:
{plot.format_array(sigma_p0)}
dx:
{plot.format_array(dx)}
initial population:
{plot.format_array(population)}
initial phase factor (in degree):
{plot.format_array(phase_factor)}
output interval:
{output_interval}
reoptimization interval:
{reoptimization_interval}
dt:
{dt}
"""
		print(content, file=f)


if __name__ == "__main__":
	main()
