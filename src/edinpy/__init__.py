"""EDinPy: exact diagonalization for discrete finite many-body systems."""

from . import fermion
from .boson import boson

__version__ = "0.1.0"

__all__ = ["fermion", "boson", "__version__"]
