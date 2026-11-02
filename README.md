# GPR-QCLE

Trajectory-supported quantum-classical Liouville dynamics with Gaussian-process reconstruction and adaptive nonadiabatic branch enrichment.

This repository contains research code for representing solutions of the mixed quantum-classical Liouville equation (QCLE) with element-specific phase-space trajectory sets and Gaussian process regression (GPR). The implementation is designed to reproduce QCLE dynamics while avoiding a full phase-space grid wherever possible. Nonadiabatic branch contributions are generated from the QCLE transition structure and incorporated adaptively into the trajectory-supported representation.

The code is intended primarily as a research implementation and reproducibility resource rather than as a general-purpose Python package.

## Method overview

The central object is the partially Wigner-transformed density matrix (PWTDM),

\[
\rho_W^{\alpha\beta}(R,P,t),
\]

represented separately for each independent density-matrix element.

At a high level, the algorithm:

1. propagates the phase-space centers associated with each density-matrix element on the corresponding mean surface;
2. reconstructs density-matrix values at nearby dynamically connected points using GPR;
3. evaluates QCLE nonadiabatic coupling contributions;
4. generates momentum-shifted branch candidates when the trajectory support overlaps the nonadiabatic-coupling region;
5. enriches or reorganizes the active representation when additional branch support is required;
6. retrains the kernel representation and continues the propagation;
7. evaluates populations, phase-space moments, energies, purity, marginals, and optional grid-based errors during the trajectory evolution.

The reconstruction is intended to approximate the QCLE solution itself. It is **not** designed to repair intrinsic limitations of the QCLE approximation. In particular, agreement between GPR-QCLE and a reference QCLE calculation should be distinguished from agreement between the QCLE and exact quantum dynamics.

## Repository layout

```text
gpr-qcle/
├── gpr/                 # GPR-QCLE trajectory implementation
├── exact_solution/      # exact/reference dynamics (Git submodule)
├── .gitmodules
└── ...
```

The main driver is:

```text
gpr/main.py
```

The driver imports the project modules `param`, `pes`, `pred`, `expectation`, and `plot`. The checked-in `constant.py` centralizes numerical defaults, filenames, precision, device selection, and plotting-related constants.

The `exact_solution` directory is configured as a Git submodule.

## Requirements

The current source uses modern Python syntax, including the Python 3.12 `type` statement, so **Python 3.12 or newer** is recommended.

Core runtime dependencies visible in the main driver include:

- [PyTorch](https://pytorch.org/)
- NumPy
- SciPy
- Matplotlib

A recent PyTorch 2.x build is recommended. CUDA is optional.

The code sets

```python
torch.set_default_dtype(torch.double)
```

and automatically selects

```python
"cuda" if torch.cuda.is_available() else "cpu"
```

so the default numerical precision is double precision and an available CUDA device is used automatically.

## Clone

Because the reference solver is a submodule, clone recursively:

```bash
git clone --recurse-submodules https://github.com/kaigu1997/gpr-qcle.git
cd gpr-qcle
```

The current `.gitmodules` file uses an SSH URL for `exact_solution`. If you do not have GitHub SSH access configured, clone the main repository first and switch the submodule URL to HTTPS:

```bash
git clone https://github.com/kaigu1997/gpr-qcle.git
cd gpr-qcle

git config submodule.exact_solution.url https://github.com/kaigu1997/exact_solution.git
git submodule update --init --recursive
```

## Environment setup

A minimal environment can be created with, for example:

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib torch
```

For GPU calculations, install the PyTorch build appropriate for the CUDA runtime on your machine or cluster.

> The repository currently does not expose itself as an installable Python package. Run the driver from the `gpr` directory so that the local modules are importable and output files are written to the intended working directory.

## Running GPR-QCLE

From the repository root:

```bash
cd gpr
python main.py
```

The command-line interface currently exposes two optional arguments:

```text
-p, --plot
    Draw phase-space/marginal output during the run.

-r FILE, --read FILE
    Read a grid-QCLE solution from FILE for comparison.
```

Examples:

```bash
# Trajectory calculation only
python main.py

# Trajectory calculation with plotting
python main.py --plot

# Compare against a grid-QCLE reference file
python main.py --read /path/to/grid_solution.txt

# Compare and plot
python main.py --plot --read /path/to/grid_solution.txt
```

Model, grid, time-step, mass, initial-state, and propagation parameters are handled by the project parameter/input machinery (`param` and the default `input.txt` naming convention). For reproducible calculations, keep a copy of the exact input file together with the generated output.

## Important driver defaults

Several sampling sizes are currently defined directly in `gpr/main.py`:

| Constant | Current value | Role in the driver |
|---|---:|---|
| `NUM_PTS` | 256 | central/inducing trajectory-point count used to initialize the GPR representation |
| `NUM_XTR_RATIO` | 32 | extra-sample ratio used around the trajectory centers |
| `NUM_MC_PTS` | 100000 | Monte Carlo sample count used for observable evaluation |
| `NUM_EVL_MC_PTS` | 10000 | additional evolution/Monte-Carlo-size constant defined by the driver |

If these values are changed for convergence tests, report them together with the physical input parameters. In practice, increasing the number of trajectory centers is not guaranteed to improve stability monotonically because the conditioning and redundancy of the kernel representation also change.

## Numerical behavior

### Precision and device

The code defaults to `torch.double`. GPU execution is selected automatically when CUDA is available; otherwise the CPU is used.

For reproducibility, the current random seed constant is

```python
SEED = 0
```

unless changed in the source.

### GPR reconstruction

The trajectory ensemble provides local support for the PWTDM reconstruction. The method is most reliable when predictions are made near dynamically represented phase-space regions. Kernel predictions far outside the trajectory support should not be interpreted as independently resolved physical structure.

The implementation is therefore best viewed as a trajectory-supported reconstruction rather than a global black-box surrogate for the complete phase-space density.

### Nonadiabatic branches

Nonadiabatic coupling can generate phase-space support that is not represented by simple propagation of the existing centers. The adaptive branch-enrichment machinery is intended to add support for these dynamically generated contributions while keeping the representation compact.

The transition weights used to assess candidate contributions contain the QCLE coupling strength, parent density, and time-step dependence. The branch-activation logic is conceptually separate from the density transferred to a particular candidate: activation detects whether the current trajectory support has reached the nonadiabatic region, while the transition weight determines the importance of the generated branch.

### Reference calculations

The `--read` option allows the GPR-QCLE trajectory solution to be compared with an externally generated grid-QCLE solution. When such a file is supplied, the driver evaluates interpolation-based errors in addition to the standard trajectory observables.

The `exact_solution` submodule provides the companion exact/reference dynamics code used for quantum-dynamical benchmarking.

## Output

The driver writes plain-text data files in the working directory. Filenames defined in `gpr/constant.py` include:

```text
points.txt
belonging.txt
all_grids.txt
marginal.txt
ave.txt
error.txt
parameters.txt
scale.txt
loss.txt
```

The main driver also writes:

```text
density.txt
```

Depending on the calculation and available memory, full-grid predictions may be omitted while marginal predictions and trajectory-based observables are still evaluated.

The output contains, among other quantities:

- trajectory/inducing and additional sample points;
- element membership information;
- reconstructed phase-space values when a full grid is available;
- position and momentum marginals;
- state populations;
- coordinate moments and covariance information;
- potential and kinetic energies;
- purity-related quantities;
- GPR parameters, rescaling factors, and losses;
- errors relative to a supplied grid-QCLE reference.

With `--plot`, the code also generates figure output and packages time-dependent frames at finalization.

## HPC / SLURM behavior

The driver checks the environment variable

```text
SLURM_JOB_END_TIME
```

when it is available. It estimates whether sufficient wall time remains for another output interval and can stop cleanly rather than beginning an iteration that is unlikely to finish before the allocation ends.

This is useful for long production calculations on SLURM-managed clusters.

## Typical benchmarking workflow

A useful validation hierarchy is:

1. **Exact/reference quantum dynamics**  
   establishes the physical benchmark for low-dimensional models;

2. **grid-QCLE**  
   provides a direct numerical solution of the QCLE against which the trajectory representation can be checked;

3. **GPR-QCLE**  
   tests whether the adaptive trajectory/kernel representation reproduces the QCLE solution at substantially reduced phase-space sampling cost.

This distinction matters because a discrepancy between QCLE and exact quantum dynamics is not necessarily a numerical failure of GPR-QCLE.

## Diagnostics worth monitoring

For production and convergence studies, useful diagnostics include:

- total and state-resolved populations;
- Hermiticity of the density matrix;
- position/momentum marginal densities;
- purity;
- energy-related observables;
- GPR loss and optimized kernel parameters;
- rescaling factors;
- number and distribution of active trajectory centers;
- grid-QCLE error when a reference grid solution is available.

When changing the number of centers, kernel parameters, branch-enrichment settings, or time step, convergence should be assessed with physical observables **and** phase-space/marginal diagnostics rather than population alone.

## Reproducibility notes

The implementation contains several choices that can materially affect numerical stability:

- number and distribution of trajectory centers;
- number of auxiliary samples;
- GPR/kernel hyperparameters;
- branch-enrichment criteria;
- QCLE time step;
- momentum-jump branch construction;
- numerical precision and hardware backend.

For a published calculation, archive the source commit, input file, driver constants, software environment, and reference data used for comparison.

## Citation

If you use this repository in scientific work, please cite my thesis (to be published) describing the method and the specific benchmark calculations.

## Contact

For questions about the implementation or the associated QCLE/GPR methodology, please open an issue in this repository.
