"""Statistics-independent discrete degree-of-freedom and mode indexing."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from numbers import Integral


@dataclass(frozen=True, slots=True)
class DoF:
    """Discrete degree of freedom used to label single-particle modes.

    Parameters
    ----------
    size : int
        Number of allowed values of the degree of freedom.
    name : str, optional
        Descriptive name such as ``"site"``, ``"spin"``, ``"species"``,
        ``"orbital"``, ``"layer"``, or ``"valley"``.
    labels : iterable of str, optional
        Names for the individual values of the degree of freedom. Their order
        is the same as the canonical integer indices ``0, 1, ..., size - 1``.

    Notes
    -----
    ``DoF`` is statistics-independent. It only specifies discrete labels and
    does not represent a quantum state or Hilbert space by itself.
    """

    size: int
    name: str | None = None
    labels: tuple[str, ...] | None = None

    def __post_init__(self):
        """Validate the degree-of-freedom cardinality, name, and labels."""
        if not isinstance(self.size, Integral):
            raise TypeError("'size' must be an integer.")
        if self.size <= 0:
            raise ValueError("'size' must be positive.")
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError("'name' must be a string or None.")

        size = int(self.size)
        object.__setattr__(self, "size", size)

        if self.labels is None:
            return

        try:
            labels = tuple(self.labels)
        except TypeError as exc:
            raise TypeError("'labels' must be an iterable of strings or None.") from exc
        if len(labels) != size:
            raise ValueError("'labels' must contain exactly 'size' entries.")
        if not all(isinstance(label, str) and label for label in labels):
            raise TypeError("All DoF labels must be non-empty strings.")
        if len(set(labels)) != len(labels):
            raise ValueError("DoF labels must be unique.")
        object.__setattr__(self, "labels", labels)


class DiscreteModes:
    """Statistics-independent mixed-radix indexing of discrete modes.

    Subclasses provide the physical meaning of the modes (for example,
    fermionic or bosonic) while sharing the Cartesian-product indexing logic.
    The first degree of freedom is the fastest-varying index.
    """

    __slots__ = ("_dofs", "_strides", "_n_modes")

    def __init__(self, *dofs: DoF):
        """Construct an ordered mode set from discrete degrees of freedom."""
        if not dofs:
            raise ValueError("At least one DoF is required.")
        if not all(isinstance(dof, DoF) for dof in dofs):
            raise TypeError("All arguments must be DoF instances.")

        strides = []
        stride = 1
        for dof in dofs:
            strides.append(stride)
            stride *= dof.size

        self._dofs = tuple(dofs)
        self._strides = tuple(strides)
        self._n_modes = prod(dof.size for dof in dofs)

    def __len__(self):
        """Return the number of modes."""
        return self._n_modes

    def __repr__(self):
        """Return an unambiguous representation using the public subclass name."""
        args = ", ".join(repr(dof) for dof in self._dofs)
        return f"{type(self).__name__}({args})"

    @property
    def dofs(self):
        """tuple[DoF, ...]: Degrees of freedom defining the modes."""
        return self._dofs

    @property
    def strides(self):
        """tuple[int, ...]: Mixed-radix strides for the DoF indices."""
        return self._strides

    @property
    def n_modes(self):
        """int: Total number of modes."""
        return self._n_modes

    def resolve(self, indices):
        """Map degree-of-freedom coordinates to a zero-based mode index.

        Parameters
        ----------
        indices : int or iterable of int
            Coordinate along each degree of freedom. A single integer is
            accepted when the mode set contains only one degree of freedom.

        Returns
        -------
        int
            Flat mode index in the canonical mixed-radix ordering.
        """
        if isinstance(indices, Integral):
            if len(self._dofs) != 1:
                raise ValueError(
                    f"Expected {len(self._dofs)} indices, received one."
                )
            indices = (int(indices),)
        else:
            indices = tuple(indices)

        if len(indices) != len(self._dofs):
            raise ValueError(
                f"Expected {len(self._dofs)} indices, received {len(indices)}."
            )

        mode = 0
        for index, dof, stride in zip(indices, self._dofs, self._strides):
            if not isinstance(index, Integral):
                raise TypeError("Mode indices must be integers.")
            index = int(index)
            if index < 0 or index >= dof.size:
                label = dof.name or "unnamed DoF"
                raise IndexError(
                    f"Index {index} is outside [0, {dof.size}) for {label}."
                )
            mode += index * stride
        return mode

    def unravel(self, mode):
        """Map a zero-based mode index to degree-of-freedom coordinates.

        Parameters
        ----------
        mode : int
            Flat mode index in ``[0, n_modes)``.

        Returns
        -------
        tuple[int, ...]
            Coordinate along each degree of freedom in the order supplied to
            ``FermionModes`` or ``BosonModes``.
        """
        if not isinstance(mode, Integral):
            raise TypeError("'mode' must be an integer.")
        mode = int(mode)
        if mode < 0 or mode >= self._n_modes:
            raise IndexError(
                f"Mode index {mode} is outside [0, {self._n_modes})."
            )

        return tuple(
            (mode // stride) % dof.size
            for dof, stride in zip(self._dofs, self._strides)
        )
