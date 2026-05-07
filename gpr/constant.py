r"""constant
============
Some constant that is used everywhere in the project
"""
import collections.abc
import time
import typing

import matplotlib as mpl
import torch

HBAR: typing.Final = 1.0

SEED: typing.Final = 0 # int(time.time() * 1e6) % 0xffff_ffff_ffff_ffff

DTYPE: typing.Final = torch.double
DEVICE: typing.Final = "cuda" if torch.cuda.is_available() else "cpu"
ENC: typing.Final = "latin1"
BYTE_ENC: typing.Final = "utf-8"
FMT: typing.Final = r"%.18e"

type Predictor = collections.abc.Callable[[torch.Tensor, int], torch.Tensor]

DEBUG_MODE: typing.Final[typing.Literal[True, False]] = True

INPUT_FILENAME: typing.Final = "input"

WFN_FILENAME: typing.Final = "wfn"
PWTDM_FILENAME: typing.Final = "pwtdm"
PWTDM_MARGINAL_FILENAME: typing.Final = "pwtdm_marginal"
AVERAGE_FILENAME: typing.Final = "ave"
POINTS_FILENAME: typing.Final = "points"
BELONGING_FILENAME: typing.Final = "belonging"
ALL_GRIDS_FILENAME: typing.Final = "all_grids"
MARGINAL_FILENAME: typing.Final = "marginal"
ERROR_FILENAME: typing.Final = "error"
PARAMETER_FILENAME: typing.Final = "parameters"
SCALE_FILENAME: typing.Final = "scale"
LOSS_FILENAME: typing.Final = "loss"

DATA_EXTENSION: typing.Final = ".txt"
FIGURE_EXTENSION: typing.Final = ".png"
ANIMATION_EXTENSION: typing.Final = ".gif"
TAR_EXTENSION: typing.Final = ".tgz"

FIGSIZE: typing.Final[tuple[float, float]] = tuple(mpl.rcParams["figure.figsize"])[:2]
TIME_TEMPLATE: typing.Final = "Time = {} a.u."
RESCALE_TEMPLATE: typing.Final = "Rescale Factor = {:.6e}"
