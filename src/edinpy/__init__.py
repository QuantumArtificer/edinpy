"""EDinPy: exact diagonalization for discrete finite many-body systems."""

from . import fermion
from .boson import boson

__version__ = "0.3.0.dev0"

__all__ = ["fermion", "boson", "__version__"]
