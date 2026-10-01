"""EDinPy: exact diagonalization for discrete finite many-body systems."""

from . import boson, fermion
from ._version import __version__

__all__ = ["fermion", "boson", "__version__"]
