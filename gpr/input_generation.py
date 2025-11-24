#!/usr/bin/env python3
"""
input_generation
================
This module generates input file.
"""
import argparse
import collections.abc
import os
import sys
import typing

import numpy as np

sys.path.append(os.path.dirname(__file__))

import pes
import utility


def main(arguments: None | collections.abc.Sequence[str] = None) -> None:
	"""
	The main routine

	Parameters
	----------
	arguments : None | collections.abc.Sequence[str], optional
		The arguments to parse, by default None
	"""
	parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Generate input file.")
	parser.add_argument("-m", "--mass", nargs=pes.DIM, default=[2000.0] * pes.DIM, type=float, help="Mass ")
	parser.add_argument("x0", nargs=pes.DIM, type=float, help="Initial positions")
	parser.add_argument("p0", nargs=pes.DIM, type=float, help="Initial momenta")
	parser.add_argument("--sigma_p0", nargs=pes.DIM, type=float, help="Initial standard deviations of momentum, by default 1/20 of initial momentum")
	parser.add_argument("--dx", nargs=pes.DIM, default=[0.1] * pes.DIM, type=float, help="Grid spacing in configurational space (for positions)")
	parser.add_argument("--population", "--ppl", nargs=pes.NUM_PES, default=[1.0] + [0.0] * (pes.NUM_PES - 1), type=float, help="Initial population. The squared sum will be normalized.")
	parser.add_argument("--phase-factor", nargs=pes.NUM_PES, default=[0.0] * pes.NUM_PES, type=float, help="Initial phase factor in degree.")
	parser.add_argument("output-interval", type=float, help="Interval between outputs (in a.u.)")
	parser.add_argument("--reoptimization-interval", "--reopt", "-r", default=-1.0, type=float, help="Interval between parameter optimization (in a.u.)")
	parser.add_argument("--dt", default=0.1, type=float, help="Time step")
	result: dict[str, typing.Any] = vars(parser.parse_args(arguments))
	result["mass"] = np.array(result["mass"])
	result["x0"] = np.array(result["x0"])
	result["p0"] = np.array(result["p0"])
	if result["sigma_p0"] is None:
		result["sigma_p0"] = result["p0"] / 20.0
	else:
		result["sigma_p0"] = np.array(result["sigma_p0"])
	result["dx"] = np.array(result["dx"])
	result["population"] = np.array(result["population"])
	result["phase_factor"] = np.array(result["phase_factor"])
	if result["reoptimization_interval"] <= 0.0:
		result["reoptimization_interval"] = result["output_interval"]
	with open("input", "w", encoding="UTF-8") as f:
		content: str = """JobType: (choose from: se_diag, se_fft, qcle)
qcle
mass:
{}
x0:
{}
p0:
{}
sigma_p0:
{}
dx:
{}
initial population:
{}
initial phase factor (in degree):
{}
output interval:
{}
reoptimization interval:
{}
dt:
{}"""
		print(
			content.format(
				utility.format_array(None, result["mass"]),
				utility.format_array(None, result["x0"]),
				utility.format_array(None, result["p0"]),
				utility.format_array(None, result["sigma_p0"]),
				utility.format_array(None, result["dx"]),
				utility.format_array(None, result["population"]),
				utility.format_array(None, result["phase_factor"]),
				result["output_interval"],
				result["reoptimization_interval"],
				result["dt"]
			),
			file=f
		)


if __name__ == "__main__":
	main()
