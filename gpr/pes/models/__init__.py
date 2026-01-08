r"""pes.model
=============
All models
"""
import inspect

from .base import ModelBase, ModelConfig, model_validation
from .const_model import CONST
from .tully import DAC, ECR, SAC


def _get_subclasses(m: type[ModelBase] = ModelBase) -> dict[str, type[ModelBase]]:
	r"""To get all subclasses of the Model

	Parameters
	----------
	m : type[ModelBase], optional
		A class to search for its subclasses, by default ModelBase

	Returns
	-------
	dict[str, type[ModelBase]]
		Lowercase name and its corresponding model class
	"""
	sc: list[type[ModelBase]] = m.__subclasses__()
	result: dict[str, type[ModelBase]] = {c.__name__.lower(): c for c in sc if not inspect.isabstract(c)}
	for c in sc:
		result |= _get_subclasses(c)
	return result


MODEL_DICT: dict[str, type[ModelBase]] = _get_subclasses()
