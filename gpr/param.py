r"""param
=====
To read the input file and construct quantities on grid
"""
import dataclasses
import math
import typing

import scipy.fft
import sympy.ntheory
import torch

import constant
import pes

torch.set_default_dtype(constant.DTYPE)
torch.set_default_device(constant.DEVICE)


def cutoff(x: float) -> float:
	r"""To calculate the cutoff of the input

	Parameters
	----------
	x : float
		Any double-precision value

	Returns
	-------
	float
		Its cutoff

	Notes
	-----
	The function calculates the cutoff to closest 1eN, 2eN, 5eN. e.g.: 0.11->0.1, 8.2->5, 3626->2000
	"""
	# calculate the number of digits, or N
	log_x: typing.Final[float] = math.log10(abs(x))
	n: typing.Final[int] = int(math.floor(log_x))
	pow_x: typing.Final[int] = 10 ** n
	# the resume
	resume: typing.Final[float] = x / pow_x
	# choose the value: 1, 2, 5
	if resume < 2.0:
		return pow_x
	elif resume < 5.0:
		return 2.0 * pow_x
	else:
		return 5.0 * pow_x


def _read_input(filename: str = constant.INPUT_FILENAME) -> tuple[type[pes.ModelBase], pes.ModelConfig, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float, float]:
	r"""To read input

	Parameters
	----------
	filename : str, optional
		The filename of input, by default `constant.INPUT_FILENAME`

	Returns
	-------
	tuple[type[pes.ModelBase], pes.ModelConfig, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]
		Model, configuration of the model, mass, centers, standard deviation, grid spacing, initial population, initial phase factor, output interval, reoptimization interval, and time step (dt)
	"""
	def match_stuff[T](name: str, dic: dict[str, T]) -> T:
		r"""To match the job name string with a `JobType`

		Parameters
		----------
		name : str
			The name to match
		dic : dict[str, T]
			The dictionary containing all names and its correspondence

		Returns
		-------
		JobType
			The kind of job that matches the name. No matching returns the first job type.
		"""
		if (name := name.lower()) in dic:
			return dic[name]
		return list(dic.values())[0] # qcle

	def line_to_tensor(line: str, target_size: int, to_expand: bool = True) -> torch.Tensor:
		r"""To convert a line of values into `torch.Tensor`, and may tile it to fit the target size

		Parameters
		----------
		line : str
			Containing all values, separated with single space " "
		target_size : int
			The return tensor would be in the shape of (`target_size`,)
		to_expand : bool, optional
			True to expand the tensor from `line` to fit `target_size`,
			False requires the tensor from `line` is already (`target_size`,),
			by default True

		Returns
		-------
		torch.Tensor, dtype of `torch.double`, shape of (`target_size`,)
			The tensor, might repeated, from `line`
		"""
		raw: typing.Final[torch.Tensor] = torch.tensor([float(val) for val in line.split()])
		if to_expand:
			assert target_size % raw.numel() == 0
			return raw.repeat(target_size // raw.numel())
		else:
			assert raw.numel() == target_size
			return raw

	with open(filename, "r", encoding="utf-8") as in_f:
		lines: list[str] = in_f.readlines()
	model: typing.Final[type[pes.ModelBase]] = match_stuff(lines[3].strip(), pes.MODEL_DICT)
	config: typing.Final[pes.ModelConfig] = pes.ModelConfig(model)
	mass: typing.Final[torch.Tensor] = line_to_tensor(lines[5].strip(), config.DIM).abs() # >0
	x0: typing.Final[torch.Tensor] = line_to_tensor(lines[7].strip(), config.DIM)
	p0: typing.Final[torch.Tensor] = line_to_tensor(lines[9].strip(), config.DIM).abs() * -torch.sgn(x0) # diff sign of x0
	sigma_p0: typing.Final[torch.Tensor] = line_to_tensor(lines[11].strip(), config.DIM).abs() # >0
	dx: typing.Final[torch.Tensor] = line_to_tensor(lines[13].strip(), config.DIM).abs() # >0
	init_ppl: typing.Final[torch.Tensor] = line_to_tensor(lines[15].strip(), config.NUM_PES, False).abs() # >0
	init_phase: typing.Final[torch.Tensor] = line_to_tensor(lines[17].strip(), config.NUM_PES, False)
	dt: float = abs(float(lines[23]))
	output_interval: typing.Final[float] = max(round(abs(float(lines[19])) / dt), 1) * dt
	reopt_interval: typing.Final[float] = max(round(abs(float(lines[21])) / output_interval), 1) * output_interval
	# dt = cutoff(min(dt, pes.HBAR / (pes.V_MAX + (math.pi * pes.HBAR) ** 2 / 2.0 * (1.0 / mass / dx ** 2).sum())))
	return model, config, mass, torch.cat((x0.abs() * -torch.sgn(p0), p0)), torch.cat((constant.HBAR / 2.0 / sigma_p0, sigma_p0)), dx, init_ppl / torch.linalg.norm(init_ppl), init_phase, output_interval, reopt_interval, dt


@typing.final
@dataclasses.dataclass(init=False, slots=True, weakref_slot=True)
class Quantity:
	r"""The basic quantities that could be used for all calculation and plots

	Parameters
	----------
	input_filename : str, optional
		The filename of input, by default `constant.INPUT_FILENAME`
	grid_filename : str | None, optional
		The filename of grid solution, generally solution of QCLE, by default None

	Attributes
	----------
	job_type : JobType
		The type of the solution
	model : type[pes.ModelBase]
		The model
	config : pes.ModelConfig
		Configuration of the model
	init_ppl : torch.Tensor
		The initial population on each surface, whose squared sum is normalized. Default to all on the ground state.
	init_phase : torch.Tensor
		The initial phase factor for each surface, in unit of degree
	init_ppl_and_phase : torch.Tensor
		The combination of `init_ppl` and `init_phase`
	mass : torch.Tensor
		Mass of classical degree of freedom
	num_grids_on_each_dimension : torch.Tensor
		The number of grids on each dimension
	num_grids_in_total : int
		The total number of grids, i.e., `num_grids_on_each_dimension.prod()`
	x0 : torch.Tensor
		Initial position of classical degree of freedom
	sigma_x0 : torch.Tensor
		Initial standard deviation of position of classical degree of freedom
	x_min : torch.Tensor
		Lower bound of position of classical degree of freedom for the grids
	x_max : torch.Tensor
		Upper bound of position of classical degree of freedom for the grids
	x_range : torch.Tensor
		Range of position of classical degree of freedom for the grids
	dx : torch.Tensor
		Grid spacing of position of classical degree of freedom
	x_grids_each_dim : list[torch.Tensor]
		Grids for position of a classical degree of freedom
	p0 : torch.Tensor
		Initial momentum of classical degree of freedom
	sigma_p0 : torch.Tensor
		Initial standard deviation of momentum of classical degree of freedom
	p_min : torch.Tensor
		Lower bound of momentum of classical degree of freedom for the grids
	p_max : torch.Tensor
		Upper bound of momentum of classical degree of freedom for the grids
	p_range : torch.Tensor
		Range of momentum of classical degree of freedom for the grids
	dp : torch.Tensor
		Grid spacing of momentum of classical degree of freedom
	p_grids_each_dim : list[torch.Tensor]
		Grids for momentum of a classical degree of freedom
	r0 : torch.Tensor
		Initial center of classical degree of freedom
	sigma_r0 : torch.Tensor
		Initial standard deviation of classical degree of freedom
	r_min : torch.Tensor
		Lower bound of classical degree of freedom for the grids
	r_max : torch.Tensor
		Upper bound of classical degree of freedom for the grids
	r_range : torch.Tensor
		Range of classical degree of freedom for the grids
	dr : torch.Tensor
		Grid spacing of classical degree of freedom
	r_grids_each_dim : list[torch.Tensor]
		Grids of a classical degree of freedom
	dt : float
		The time interval for evolution (in atomic unit time)
	output_ticks : int
		The number of dt between outputs
	output_interval : float
		The time interval between outputs
	reopt_ticks : int
		The number of outputs between hyperparameter reoptimization
	reopt_inteval : float
		The time interval between hyperparameter reoptimization
	total_ticks : int
		The upper bound of the number of outputs to finish the evolution, estimated by :math:`t=\frac{|x_0|-(-|x_0|)}{p_0/m}*coe`
	"""
	@staticmethod
	def __get_fft_grids(N: torch.Tensor) -> torch.Tensor:
		r"""To get the number of grids for each dimension that is optimal for fft

		Parameters
		----------
		N : torch.Tensor, dtype of `torch.int64`
			The number of grids on each dimension

		Returns
		-------
		torch.Tensor, dtype of `torch.int64`
			Optimal number of grids on each dimension
		"""
		result: torch.Tensor = N.cpu().clone()
		result.apply_(scipy.fft.next_fast_len)
		while not torch.all(result % 2 == 1):
			result += torch.where(result % 2 == 1, 0, 1)
			result.apply_(scipy.fft.next_fast_len)
		return result.to(torch.get_default_device())

	__NUM_GRIDS_PER_WAVELENGTH: typing.ClassVar = 2.0
	model: typing.Final[type[pes.ModelBase]]
	config: typing.Final[pes.ModelConfig]
	init_ppl: typing.Final[torch.Tensor]
	init_phase: typing.Final[torch.Tensor]
	init_ppl_and_phase: typing.Final[torch.Tensor]
	mass: typing.Final[torch.Tensor]
	num_grids_on_each_dimension: typing.Final[torch.Tensor]
	num_grids_in_total: typing.Final[int]
	x0: typing.Final[torch.Tensor]
	sigma_x0: typing.Final[torch.Tensor]
	x_min: typing.Final[torch.Tensor]
	x_max: typing.Final[torch.Tensor]
	x_range: typing.Final[torch.Tensor]
	dx: typing.Final[torch.Tensor]
	x_grids_each_dim: typing.Final[list[torch.Tensor]]
	p0: typing.Final[torch.Tensor]
	sigma_p0: typing.Final[torch.Tensor]
	p_min: typing.Final[torch.Tensor]
	p_max: typing.Final[torch.Tensor]
	p_range: typing.Final[torch.Tensor]
	dp: typing.Final[torch.Tensor]
	p_grids_each_dim: typing.Final[list[torch.Tensor]]
	r0: typing.Final[torch.Tensor]
	sigma_r0: typing.Final[torch.Tensor]
	r_min: typing.Final[torch.Tensor]
	r_max: typing.Final[torch.Tensor]
	r_range: typing.Final[torch.Tensor]
	dr: typing.Final[torch.Tensor]
	r_grids_each_dim: typing.Final[list[torch.Tensor]]
	dt: typing.Final[float]
	output_ticks: typing.Final[int]
	output_interval: typing.Final[float]
	reopt_ticks: typing.Final[int]
	reopt_interval: typing.Final[float]
	total_ticks: typing.Final[int]

	def __init__(
		self,
		input_filename: str = constant.INPUT_FILENAME,
		total_grids: int | None = None
	) -> None:
		# directly get from input
		dx: torch.Tensor
		dt: float
		self.model, self.config, self.mass, self.r0, self.sigma_r0, dx, self.init_ppl, self.init_phase, self.output_interval, self.reopt_interval, dt = _read_input(input_filename)
		# non-coord terms
		self.init_ppl_and_phase = self.init_ppl * torch.exp(math.pi * 1.0j * self.init_phase / 180.0)
		# centers and widths
		self.x0 = self.r0[:self.config.DIM]
		self.sigma_x0 = self.sigma_r0[:self.config.DIM]
		self.p0 = self.r0[self.config.DIM:]
		self.sigma_p0 = self.sigma_r0[self.config.DIM:]
		# x
		value_from_total_grids: bool = False
		half_grids: torch.Tensor
		if total_grids is not None:
			n_grids_each_dim = int(round(math.pow(total_grids, 1.0 / self.config.DIM)))
			if n_grids_each_dim ** self.config.PHASEDIM == total_grids:
				self.num_grids_in_total = total_grids
				self.num_grids_on_each_dimension = torch.full((self.config.DIM,), n_grids_each_dim, dtype=torch.int64)
				half_grids = (self.num_grids_on_each_dimension - 1) // 2
				self.x_max = 2.0 * self.x0.abs()
				self.x_min = -self.x_max
				self.x_range = self.x_max - self.x_min
				self.dx = self.x_range / self.num_grids_on_each_dimension
				value_from_total_grids = True
		if not value_from_total_grids:
			# dx_from_wavelength: typing.Final[torch.Tensor] = (2.0 * math.pi * constant.HBAR) / ((self.p0 + 3.0 * self.sigma_p0) * Quantity.__NUM_GRIDS_PER_WAVELENGTH) # N grids per de Broglie wavelength
			# resolution: typing.Final[torch.Tensor] = self.sigma_x0 / dx_from_wavelength # also sigma_p0/dp
			# dp_theory: typing.Final[torch.Tensor] = self.sigma_p0 / resolution # in real, dx*dp=2*pi*hbar/(4*N+1)
			# self.dx = torch.minimum(dx, dx_from_wavelength) # .apply_(cutoff)
			# half_grids_theory: typing.Final[torch.Tensor] = torch.ceil((2.0 * math.pi * constant.HBAR / self.dx / dp_theory - 1) / 4).to(torch.int64)
			# half_grids_x0: typing.Final[torch.Tensor] = torch.ceil(2.0 * self.x0.abs() / self.dx).to(torch.int64)
			# N_lowerbound: typing.Final[torch.Tensor] = 2 * torch.maximum(half_grids_theory, half_grids_x0) + 1
			# self.num_grids_on_each_dimension = Quantity.__get_fft_grids(N_lowerbound)
			# half_grids: typing.Final[torch.Tensor] = (self.num_grids_on_each_dimension - 1) // 2 # since x_max could > from_spacing, n_grids might be adjusted
			# self.dx = dx.abs()
			# self.x_max = 2.0 * self.x0.abs()
			# self.x_min = -self.x_max
			# self.x_range = self.x_max - self.x_min
			# half_grids = torch.round(self.x_max / self.dx).to(torch.int64)
			# self.num_grids_on_each_dimension = half_grids * 2 + 1
			# self.num_grids_in_total = int(self.num_grids_on_each_dimension.prod().item())
			dx_from_wavelength: typing.Final[torch.Tensor] = (2.0 * math.pi * constant.HBAR) / ((self.p0 + 3.0 * self.sigma_p0) * Quantity.__NUM_GRIDS_PER_WAVELENGTH) # N grids per de Broglie wavelength
			resolution: typing.Final[torch.Tensor] = self.sigma_x0 / dx_from_wavelength # also sigma_p0/dp
			dp_theory: typing.Final[torch.Tensor] = self.sigma_p0 / resolution # in real, dx*dp=2*pi*hbar/(4*N+1)
			self.dx = torch.minimum(dx, dx_from_wavelength) # .apply_(cutoff)
			half_grids_theory: typing.Final[torch.Tensor] = torch.ceil((2.0 * math.pi * constant.HBAR / self.dx / dp_theory - 1) / 4).to(torch.int64)
			half_grids_x0: typing.Final[torch.Tensor] = torch.ceil(2.0 * self.x0.abs() / self.dx).to(torch.int64)
			N_lowerbound: typing.Final[torch.Tensor] = 2 * torch.maximum(half_grids_theory, half_grids_x0) + 1
			self.num_grids_on_each_dimension = Quantity.__get_fft_grids(N_lowerbound)
			half_grids = (self.num_grids_on_each_dimension - 1) // 2 # since x_max could > from_spacing, n_grids might be adjusted
			self.x_max = half_grids * self.dx
			self.x_min = -self.x_max
			self.x_range = self.x_max - self.x_min
			self.num_grids_in_total = int(self.num_grids_on_each_dimension.prod().item())
		self.x_grids_each_dim = [torch.linspace(self.x_min[iDim], self.x_max[iDim], int(self.num_grids_on_each_dimension[iDim].item())) for iDim in self.config.DIM_RANGE]
		# p
		self.dp = 2.0 * math.pi * constant.HBAR / (4 * half_grids + 1.0) / self.dx
		self.p_min = self.p0 - self.dp * half_grids
		self.p_max = self.p0 + self.dp * half_grids
		self.p_range = self.p_max - self.p_min
		self.p_grids_each_dim = [torch.linspace(self.p_min[iDim], self.p_max[iDim], int(self.num_grids_on_each_dimension[iDim].item())) for iDim in self.config.DIM_RANGE]
		# r
		self.r_min = torch.cat((self.x_min, self.p_min))
		self.r_max = torch.cat((self.x_max, self.p_max))
		self.r_range = torch.cat((self.x_range, self.p_range))
		self.dr = torch.cat((self.dx, self.dp))
		self.r_grids_each_dim = self.x_grids_each_dim + self.p_grids_each_dim
		# time quantities
		self.dt = cutoff(min(
			dt,
			constant.HBAR / (self.model.max_potential() + 0.5 * (1.0 / self.mass * (constant.HBAR * math.pi / self.dx) ** 2).sum().item()) # hbar/E
		))
		self.output_ticks = max(int(round(self.output_interval / self.dt)), 1)
		self.output_interval = self.output_ticks * self.dt
		self.reopt_ticks = max(int(round(self.reopt_interval / self.output_interval)), 1)
		self.reopt_interval = self.reopt_ticks * self.output_interval
		self.total_ticks = int((self.x_range / (self.p0 / self.mass)).max().item() * 1.5 / self.output_interval) if self.output_ticks != 0 else 0
		# assume a gaussian wavepacket, widening as :math:`\sigma_x^2(t)=\sigma_{x}^2(0)+(\sigma_{p}(0)*t/m)^2`
		# after widening, 5sigma should still be within the box
		# print info
		print(f"Running gpr on {self.config.NAME.lower()}")
		print(f"dt = {self.dt} au, output {"until the end" if self.output_ticks == 0 else f"every {self.output_interval} au"}.")
		print("Initial population\tInitial phase factor")
		for iPES in self.config.PES_RANGE:
			print(f"{self.init_ppl[iPES]:18}\t{self.init_phase[iPES]:20}")
		print("mass x0 sigma_x0 xmin xmax dx p0 sigma_p0 pmin pmax dp ngrids")
		for iDim in self.config.DIM_RANGE:
			print(self.mass[iDim].item(), self.x0[iDim].item(), self.sigma_x0[iDim].item(), self.x_min[iDim].item(), self.x_max[iDim].item(), self.dx[iDim].item(), self.p0[iDim].item(), self.sigma_p0[iDim].item(), self.p_min[iDim].item(), self.p_max[iDim].item(), self.dp[iDim].item(), self.num_grids_on_each_dimension[iDim].item(), sympy.ntheory.factorint(self.num_grids_on_each_dimension[iDim].item()))
		print(end="", flush=True)
