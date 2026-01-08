r"""job_arrangement
=================
To generate the jobs
"""
import argparse
import collections.abc
import subprocess
import typing

import numpy as np
import numpy.typing as npt

import constant
import input_generation as ig
import param
import pes


@typing.final
class Arguments:
	r"""To read Arguments from command line

	Parameters
	----------
	arguments : None | collections.abc.Sequence[str], optional
		Arguments to generate inputs, by default None (and will read from command line)

	Methods
	-------
	x0()
		Initial positions
	all_values()
		All momenta/ln(E), each entry in a row
	is_energy()
		Whether `all_values` are log energy or not
	all_momenta()
		All momenta, each entry in a row
	mass()
		Mass
	output_interval()
		Time between outputs in a.u.
	extra_arguments() 
		Extra arguments from command line
	"""
	__slots__: typing.Final[tuple] = ("__config", "__extra_args", "__x0", "__mass", "__all_vals", "__is_lnE", "__all_p", "__output_interval")
	__STEP_BETWEEN_OUTPUT: typing.Final[float] = 0.5
	__config: typing.Final[pes.ModelConfig]
	__extra_args: typing.Final[list[str]]
	__x0: typing.Final[npt.NDArray[np.double]]
	__mass: typing.Final[npt.NDArray[np.double]]
	__all_vals: typing.Final[npt.NDArray[np.double]]
	__is_lnE: typing.Final[bool]
	__all_p: typing.Final[npt.NDArray[np.double]]
	__output_interval: typing.Final[npt.NDArray[np.double]]

	def __generate_momentum(
		self,
		min_vals: npt.NDArray[np.double],
		max_vals: npt.NDArray[np.double],
		dp: npt.NDArray[np.double]
	) -> npt.NDArray[np.double]:
		r"""To generate all the momenta needed

		Parameters
		----------
		min_vals : npt.NDArray[np.double], shape of (DIM,)
			The minimum value of each dimension
		max_vals : npt.NDArray[np.double], shape of (DIM,)
			The maximum value of each dimension
		dp : npt.NDArray[np.double], shape of (DIM,)
			The spacing for each dimension

		Returns
		-------
		npt.NDArray[np.double], shape of (NUM_CASE, DIM)
			Each row is the momenta/ln(E) for one case
		"""
		num_p: typing.Final[npt.NDArray[np.int_]] = ((max_vals - min_vals) / dp).round().astype(np.int_) + 1
		return np.stack(np.meshgrid(*[np.linspace(pmin, pmax, n) for pmin, pmax, n in zip(min_vals, max_vals, num_p)], indexing="ij"), -1, None).reshape(-1, self.__config.DIM)

	def __init__(self, arguments: None | collections.abc.Sequence[str] = None) -> None:
		model_parser: typing.Final[argparse.ArgumentParser] = argparse.ArgumentParser(description="Provide model to use", add_help=False)
		model_parser.add_argument("--model", default="dac", type=str, choices=pes.MODEL_DICT.keys(), help="Model to use")
		model_args, remaining_args = model_parser.parse_known_args(arguments)
		self.__config = pes.ModelConfig(pes.MODEL_DICT[model_args.model])

		parser: typing.Final[argparse.ArgumentParser] = argparse.ArgumentParser(description="Generate multiple inputs and run them on slurm", epilog="Extra arguments will be passed to `input_generation.py` and see its documentation", parents=[model_parser])
		parser.add_argument("min", nargs=self.__config.DIM, type=float, help="Minimum momentum/ln(energy), where to start from")
		parser.add_argument("max", nargs=self.__config.DIM, type=float, help="Maximum momentum/ln(energy), where to end")
		parser.add_argument("--dp", nargs=self.__config.DIM, default=[0.1] * self.__config.DIM, type=float, help="Momentum/energy spacing")
		parser.add_argument("-e", "--energy", action="store_true", help="By default the input is momenta; this option would assume the input to be ln(energy)")
		parser.add_argument("--x0", nargs=self.__config.DIM, default=[-15.0] * self.__config.DIM, type=float, help="Initial positions")
		parser.add_argument("-m", "--mass", nargs=self.__config.DIM, default=[2000.0] * self.__config.DIM, type=float, help="Mass")
		parser.add_argument("-t", "--output-interval", type=float, help="Interval between outputs (in a.u.)")
		known_args: argparse.Namespace
		known_args, self.__extra_args = parser.parse_known_args(remaining_args)
		self.__x0 = np.array(known_args.x0, np.double)
		self.__mass = np.array(known_args.mass, np.double)
		self.__all_vals = self.__generate_momentum(
			np.array(known_args.min, np.double),
			np.array(known_args.max, np.double),
			np.array(known_args.dp, np.double)
		)
		self.__is_lnE = known_args.energy
		if self.__is_lnE:
			self.__all_p = np.sqrt(2.0 * self.__mass * np.exp(self.__all_vals))
		else:
			self.__all_p = self.__all_vals

		if known_args.output_interval is not None:
			self.__output_interval = np.full(self.__all_p.shape[0], float(known_args.output_interval))
		else:
			self.__output_interval = np.vectorize(param.cutoff)(np.min(Arguments.__STEP_BETWEEN_OUTPUT / (self.__all_p / self.__mass), axis=-1))

	@property
	def x0(self) -> npt.NDArray[np.double]:
		r"""Initial positions

		Returns
		-------
		npt.NDArray[np.double]
			Initial positions
		"""
		return self.__x0

	@property
	def all_values(self) -> npt.NDArray[np.double]:
		r"""All momenta/ln(E), each entry in a row

		Returns
		-------
		npt.NDArray[np.double], shape of (N_p, DIM)
			All momenta or ln(E)
		"""
		return self.__all_vals

	@property
	def is_energy(self) -> bool:
		r"""Whether `all_values` are log energy or not

		Returns
		-------
		bool
			True if it is log energy, False for momenta
		"""
		return self.__is_lnE

	@property
	def all_momenta(self) -> npt.NDArray[np.double]:
		r"""All momenta, each entry in a row

		Returns
		-------
		npt.NDArray[np.double], shape of (N_p, DIM)
			All momenta
		"""
		return self.__all_p

	@property
	def mass(self) -> npt.NDArray[np.double]:
		r"""Mass

		Returns
		-------
		npt.NDArray[np.double]
			Mass
		"""
		return self.__mass

	@property
	def output_interval(self) -> npt.NDArray[np.double]:
		r"""Time between outputs in a.u.

		Returns
		-------
		npt.NDArray[np.double]
			Output interval
		"""
		return self.__output_interval

	@property
	def extra_arguments(self) -> list[str]:
		r"""Extra arguments from command line

		Returns
		-------
		list[str]
			Extra arguments
		"""
		return self.__extra_args


@typing.final
class Script:
	r"""To generate submission script from an example script

	Parameters
	----------
	example_filename : str, optional
		Filename of the example script, by default "example.sh"
	"""
	__slots__: typing.Final[tuple] = ("__head", "__mid", "__end")
	__BATCH_COMMAND_NAME: typing.Final = "#SBATCH "
	__head: typing.Final[str]
	__mid: typing.Final[str]
	__end: typing.Final[str]

	def __init__(self, example_filename: str = "example.sh") -> None:
		with open(example_filename, "r", encoding=constant.ENC) as f:
			lines: typing.Final[list[str]] = f.readlines()
		try:
			idx: typing.Final[int] = lines.index("\n")
		except ValueError as ve:
			raise ValueError("Empty line not found") from ve
		assert all(s[:len(Script.__BATCH_COMMAND_NAME)] == Script.__BATCH_COMMAND_NAME for s in lines[:idx])
		self.__head = f"""#!/usr/bin/env bash
#SBATCH --job-name=grid_solution
{"".join(lines[:idx])}"""
		self.__mid = f"""srun hostname | sort

{"".join(lines[idx + 1:])}"""
		self.__end = """working_dir=${working_dirs[$SLURM_ARRAY_TASK_ID]}
#begins here
cd ${working_dir}
echo $(date +"%Y-%m-%d %H:%M:%S.%N")
python main.py -e
echo $(date +"%Y-%m-%d %H:%M:%S.%N")
#ends here

echo "scontrol show job $SLURM_JOB_ID"
scontrol show job $SLURM_JOB_ID
echo "sacct -j $SLURM_JOB_ID"
sacct -j $SLURM_JOB_ID --format="jobid,start,end,elapsed,nodelist,exitcode,state"
"""

	def __call__(self, job_set: set[str]) -> str:
		r"""To generate the job submit script based on the current job set

		Parameters
		----------
		job_set : set[str]
			The set of job names

		Returns
		-------
		str
			The slurm script to submit the job
		"""
		return f"""{self.__head}#SBATCH --array=0-{len(job_set) - 1}

{self.__mid}
working_dirs=({'"' + '" "'.join(job_set) + '"'})
{self.__end}"""


def main() -> None:
	r"""The main routine
	"""
	args: typing.Final[Arguments] = Arguments()
	file_list: typing.Final[list[str]] = ["constant.py", "pes", "plot", "evolve.py", "gp.py", "expectation.py", "point.py", "main.py"] # all py scripts, exclude this file and input generation
	job_set: set[str] = set()
	for value, momentum, output_interval in zip(args.all_values, args.all_momenta, args.output_interval):
		dir_name: str = f"gpr_{"_".join([f"{v:g}" for v in value])}"
		ig.main([str(x0i) for x0i in args.x0] + [str(p) for p in momentum] + [str(output_interval), "--model", args.__config.NAME] + ["-m"] + [str(m) for m in args.mass] + args.extra_arguments) # generate input. qcle as job type for grid solution is not used
		subprocess.run(["mkdir", "-p", dir_name]) # make directory
		subprocess.run(["cp", "-rL"] + file_list + ["input", dir_name], check=False) # copy elf and input file to dir
		job_set.add(dir_name)
	job_submit_script: typing.Final[Script] = Script()
	script_name: str = f"submit_{len(job_set)}_jobs.sh"
	with open(script_name, "w", encoding=constant.ENC) as f_submit: # generate script
		print(job_submit_script(job_set), file=f_submit)
	result: subprocess.CompletedProcess = subprocess.run(["sbatch", "--wait", script_name], capture_output=True) # submit script
	print(result.stderr.decode(constant.BYTE_ENC))
	print(result.stdout.decode(constant.BYTE_ENC))
	job_array_id: str = result.stdout.strip().split()[-1].decode(constant.BYTE_ENC)
	for array_idx, dir_name in enumerate(job_set):
		subprocess.run(["mv", "slurm-" + job_array_id + "_" + str(array_idx) + ".out", dir_name])


if __name__ == "__main__":
	main()
