"""
job_arrangement
=================
To generate the jobs
"""
import argparse
import glob
import os
import subprocess
import sys
import typing

import numpy as np
import numpy.typing as npt

sys.path.append(os.path.dirname(__file__))

import input_generation
import pes


def generate_momentum() -> npt.NDArray[np.str_]:
	"""
	To generate all the momenta needed

	Returns
	-------
	npt.NDArray[np.str_]
		2D array of strings.
		Each row is the momenta for one case.
	"""
	parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Generate initial momenta")
	parser.add_argument("--energy", "-e", nargs=pes.DIM, type=float, help="In energy form; mass should be provided", metavar="MASS")
	parser.add_argument("min", nargs=pes.DIM, type=float, help="Minimum momentum, where to start from")
	parser.add_argument("max", nargs=pes.DIM, type=float, help="Maximum momentum, where to end")
	parser.add_argument("-d", nargs=pes.DIM, default=[0.1] * pes.DIM, type=float, help="Momentum (or energy) spacing")
	result: dict[str, typing.Any] = vars(parser.parse_args())
	result["min"] = np.array(result["min"])
	result["max"] = np.array(result["max"])
	result["d"] = np.array(result["d"])
	num_p: npt.NDArray[np.int_] = ((result["max"] - result["min"]) / result["d"]).round().astype(np.int_) + 1
	all_p: npt.NDArray[np.double] = np.stack(np.meshgrid(*tuple(np.linspace(pmin, pmax, n) for pmin, pmax, n in zip(result["min"], result["max"], num_p))), -1).reshape(-1, pes.DIM)
	if result["energy"]:
		all_p = np.sqrt(2.0 * np.array(result["energy"]) * np.exp(all_p))
	all_p_str: npt.NDArray[np.str_] = all_p.astype(np.str_)
	digits_before_dot: npt.NDArray[np.int_] = np.char.find(all_p_str, ".")
	digits_after_dot: npt.NDArray[np.int_] = np.char.str_len(all_p_str) - 1 - digits_before_dot
	add_zero = np.vectorize(lambda i: np.str_("0" * i))
	return np.char.add(np.char.add(add_zero(np.max(digits_before_dot, 0) - digits_before_dot), all_p_str), add_zero(np.max(digits_after_dot, 0) - digits_after_dot))


def main() -> None:
	"""
	The main routine
	"""
	momenta: npt.NDArray[np.str_] = generate_momentum()
	file_list: list[str] = glob.glob("*.py") # all py scripts, exclude this file and input generation
	if os.path.basename(__file__) in file_list:
		file_list.remove(os.path.basename(__file__))
	if "input_generation.py" in file_list:
		file_list.remove("input_generation.py")
	job_list: list[str] = []
	momentum: np.str_
	for momentum in momenta:
		dir_name: str = "gpr_" + "_".join(momentum)
		input_generation.main(["-8.0"] + momentum.astype(str).tolist() + ["10.0","--population","0.995","0.1","--phase-factor","0.0","45.0"]) # generate input. qcle as job type for grid solution is not used
		subprocess.run(["mkdir", "-p", dir_name]) # make directory
		subprocess.run(["cp"] + file_list + ["input", dir_name]) # copy elf and input file to dir
		job_list.append(dir_name)
	job_submit_script: str = """#!/usr/bin/env bash
#SBATCH --job-name=gpr
#SBATCH --time=1-0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --profile=all
#SBATCH --array=0-{}
#SBATCH --account=rrg-rkapral-ac

srun hostname | sort

module -q load StdEnv/2023 # environmental module
module -q load gcc/12.3 clang/17 # clang and clang-extras
module -q load python/3.11 # python interpreter
module -q load imkl tbb # intel mkl and tbb
module -q load eigen nlopt scipy-stack # numerical libs
module -q load qt # qt
module -q load fmt # formatter
source $HOME/.venv/venv/bin/activate

working_dirs=({})
working_dir=${{working_dirs[$SLURM_ARRAY_TASK_ID]}}
#begins here
cd ${{working_dir}}
python main.py --plot no
#ends here
"""
	script_name: str = "submit_" + str(len(job_list)) + "_jobs.sh"
	with open(script_name, "w") as f_submit: # generate script
		print(job_submit_script.format(len(job_list) - 1, '"' + '" "'.join(job_list) + '"'), file=f_submit)
	result: subprocess.CompletedProcess = subprocess.run(["sbatch", "--wait", script_name], capture_output=True) # submit script
	print(result.stderr.decode("utf-8"))
	print(result.stdout.decode("utf-8"))
	job_array_id: str = result.stdout.strip().split()[-1].decode("utf-8")
	for array_idx, dir_name in enumerate(job_list):
		subprocess.run(["mv", "slurm-" + job_array_id + "_" + str(array_idx) + ".out", dir_name])


if __name__ == "__main__":
	main()
