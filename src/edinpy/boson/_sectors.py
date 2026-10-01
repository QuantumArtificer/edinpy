"""Many-body sectors and direct bosonic sector basis construction."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import product
from numbers import Integral

from ._basis import FockBasis, FockVector, _OccupationCodec
from ._modes import BosonModes


def _modes_with_dof_value(modes, dof_index, value_index):
    """Return mode indices carrying one value of a selected DoF."""
    dof = modes.dofs[dof_index]
    stride = modes.strides[dof_index]
    block = stride * dof.size
    selected = []

    for block_start in range(0, modes.n_modes, block):
        start = block_start + value_index * stride
        selected.extend(range(start, start + stride))

    return tuple(selected)


def _dof_pair_mode_groups(modes, first_index, second_index):
    """Partition modes into intersections of two degrees of freedom."""
    first = modes.dofs[first_index]
    second = modes.dofs[second_index]
    first_stride = modes.strides[first_index]
    second_stride = modes.strides[second_index]
    groups = [[[] for _ in range(second.size)] for _ in range(first.size)]

    for mode in range(modes.n_modes):
        first_value = (mode // first_stride) % first.size
        second_value = (mode // second_stride) % second.size
        groups[first_value][second_value].append(mode)

    return tuple(tuple(tuple(cell) for cell in row) for row in groups)


def _multi_dof_mode_groups(modes, dof_indices):
    """Partition modes by joint values of several projected DoFs."""
    dof_indices = tuple(dof_indices)
    value_ranges = [range(modes.dofs[index].size) for index in dof_indices]
    coordinates = tuple(product(*value_ranges))
    sizes = tuple(modes.dofs[index].size for index in dof_indices)
    groups = [[] for _ in coordinates]

    for mode in range(modes.n_modes):
        group_index = 0
        for dof_index, size in zip(dof_indices, sizes):
            value_index = (mode // modes.strides[dof_index]) % size
            group_index = group_index * size + value_index
        groups[group_index].append(mode)

    return coordinates, tuple(tuple(group) for group in groups)


def _completed_targets(dof, particle_numbers, total):
    """Return per-value targets, inferring a sole unspecified value."""
    targets = [None] * dof.size
    for value_index, number in particle_numbers.items():
        targets[value_index] = number

    unspecified = [index for index, value in enumerate(targets) if value is None]
    if len(unspecified) == 1:
        targets[unspecified[0]] = total - sum(
            value for value in targets if value is not None
        )
    return tuple(targets)


def _bounded_compositions(total, upper_bounds):
    """Yield integer vectors with fixed sum and component-wise upper bounds."""
    upper_bounds = tuple(int(bound) for bound in upper_bounds)
    if total < 0 or total > sum(upper_bounds):
        return

    suffix_capacity = [0] * (len(upper_bounds) + 1)
    for index in range(len(upper_bounds) - 1, -1, -1):
        suffix_capacity[index] = suffix_capacity[index + 1] + upper_bounds[index]

    values = [0] * len(upper_bounds)

    def recurse(index, remaining):
        if index + 1 == len(upper_bounds):
            if 0 <= remaining <= upper_bounds[index]:
                values[index] = remaining
                yield tuple(values)
            return

        lower = max(0, remaining - suffix_capacity[index + 1])
        upper = min(upper_bounds[index], remaining)
        for value in range(lower, upper + 1):
            values[index] = value
            yield from recurse(index + 1, remaining - value)

    if not upper_bounds:
        if total == 0:
            yield ()
        return

    yield from recurse(0, total)


def _two_dof_occupation_tables(capacities, row_targets, column_targets, total):
    """Yield feasible joint boson totals for two projected DoFs."""
    capacities = tuple(tuple(int(value) for value in row) for row in capacities)
    n_rows = len(capacities)
    n_columns = len(capacities[0])

    future_column_capacity = [[0] * n_columns for _ in range(n_rows + 1)]
    for row in range(n_rows - 1, -1, -1):
        for column in range(n_columns):
            future_column_capacity[row][column] = (
                future_column_capacity[row + 1][column] + capacities[row][column]
            )

    row_capacities = [sum(row) for row in capacities]
    future_row_min = [0] * (n_rows + 1)
    future_row_max = [0] * (n_rows + 1)
    for row in range(n_rows - 1, -1, -1):
        target = row_targets[row]
        future_row_min[row] = future_row_min[row + 1] + (
            0 if target is None else target
        )
        future_row_max[row] = future_row_max[row + 1] + (
            row_capacities[row] if target is None else target
        )

    remaining_columns = [
        None if target is None else int(target) for target in column_targets
    ]
    table = [[0] * n_columns for _ in range(n_rows)]

    def recurse(row, remaining_total):
        if row == n_rows:
            if remaining_total != 0:
                return
            if any(value not in (None, 0) for value in remaining_columns):
                return
            yield tuple(tuple(values) for values in table)
            return

        target = row_targets[row]
        if target is None:
            lower_total = max(0, remaining_total - future_row_max[row + 1])
            upper_total = min(
                row_capacities[row],
                remaining_total - future_row_min[row + 1],
            )
            row_totals = range(lower_total, upper_total + 1)
        else:
            if target > remaining_total:
                return
            row_totals = (target,)

        bounds = []
        for column in range(n_columns):
            bound = capacities[row][column]
            if remaining_columns[column] is not None:
                bound = min(bound, remaining_columns[column])
            bounds.append(bound)

        for row_total in row_totals:
            for values in _bounded_compositions(row_total, bounds):
                feasible = True
                for column, value in enumerate(values):
                    table[row][column] = value
                    if remaining_columns[column] is not None:
                        remaining_columns[column] -= value
                        residual = remaining_columns[column]
                        if (
                            residual < 0
                            or residual > future_column_capacity[row + 1][column]
                        ):
                            feasible = False

                new_remaining_total = remaining_total - row_total
                if (
                    new_remaining_total < future_row_min[row + 1]
                    or new_remaining_total > future_row_max[row + 1]
                ):
                    feasible = False

                if feasible:
                    yield from recurse(row + 1, new_remaining_total)

                for column, value in enumerate(values):
                    if remaining_columns[column] is not None:
                        remaining_columns[column] += value

    yield from recurse(0, total)


def _multi_dof_occupation_patterns(capacities, coordinates, targets, total):
    """Yield feasible joint boson totals for three or more projected DoFs."""
    capacities = tuple(int(value) for value in capacities)
    coordinates = tuple(tuple(value for value in item) for item in coordinates)
    targets = tuple(tuple(value for value in target) for target in targets)

    constraints = []
    for axis, axis_targets in enumerate(targets):
        for value_index, target in enumerate(axis_targets):
            if target is not None:
                constraints.append((axis, value_index, int(target)))

    memberships = []
    for coordinate in coordinates:
        memberships.append(
            tuple(
                constraint_index
                for constraint_index, (axis, value_index, _target) in enumerate(
                    constraints
                )
                if coordinate[axis] == value_index
            )
        )

    n_cells = len(capacities)
    suffix_capacity = [0] * (n_cells + 1)
    for cell in range(n_cells - 1, -1, -1):
        suffix_capacity[cell] = suffix_capacity[cell + 1] + capacities[cell]

    suffix_constraint_capacity = [[0] * (n_cells + 1) for _ in constraints]
    for constraint_index, (axis, value_index, _target) in enumerate(constraints):
        suffix = suffix_constraint_capacity[constraint_index]
        for cell in range(n_cells - 1, -1, -1):
            suffix[cell] = suffix[cell + 1]
            if coordinates[cell][axis] == value_index:
                suffix[cell] += capacities[cell]

    remaining_targets = [target for _axis, _value, target in constraints]
    pattern = [0] * n_cells

    def recurse(cell, remaining_total):
        if remaining_total < 0 or remaining_total > suffix_capacity[cell]:
            return
        for constraint_index, remaining in enumerate(remaining_targets):
            if (
                remaining < 0
                or remaining > suffix_constraint_capacity[constraint_index][cell]
            ):
                return

        if cell == n_cells:
            if remaining_total == 0 and all(value == 0 for value in remaining_targets):
                yield tuple(pattern)
            return

        lower = max(0, remaining_total - suffix_capacity[cell + 1])
        upper = min(capacities[cell], remaining_total)

        for constraint_index in memberships[cell]:
            remaining = remaining_targets[constraint_index]
            future_capacity = suffix_constraint_capacity[constraint_index][cell + 1]
            lower = max(lower, remaining - future_capacity)
            upper = min(upper, remaining)

        if lower > upper:
            return

        for occupation in range(lower, upper + 1):
            pattern[cell] = occupation
            for constraint_index in memberships[cell]:
                remaining_targets[constraint_index] -= occupation

            yield from recurse(cell + 1, remaining_total - occupation)

            for constraint_index in memberships[cell]:
                remaining_targets[constraint_index] += occupation

    yield from recurse(0, int(total))


def _group_packed_states(mode_indices, count, codec):
    """Return packed states distributing ``count`` bosons over selected modes."""
    mode_indices = tuple(sorted(int(mode) for mode in mode_indices))
    count = int(count)
    if count < 0:
        return ()
    if not mode_indices:
        return (0,) if count == 0 else ()

    bits = codec.bits_per_mode
    states = []

    def recurse(position, remaining, packed):
        if position == 0:
            mode = mode_indices[0]
            states.append(packed | (remaining << (mode * bits)))
            return

        mode = mode_indices[position]
        shift = mode * bits
        for occupation in range(remaining + 1):
            recurse(
                position - 1,
                remaining - occupation,
                packed | (occupation << shift),
            )

    recurse(len(mode_indices) - 1, count, 0)
    return tuple(states)


def _basis_from_group_occupations(n_modes, N, groups, occupations):
    """Build a sorted bosonic basis from disjoint groups and group totals."""
    groups = tuple(tuple(group) for group in groups)
    occupations = tuple(tuple(int(value) for value in pattern) for pattern in occupations)
    if not occupations:
        raise ValueError("Particle-number constraints are mutually incompatible.")

    codec = _OccupationCodec(n_modes, N)
    cache = {}
    states = []

    for pattern in occupations:
        if len(pattern) != len(groups):
            raise ValueError("Occupation pattern does not match mode groups.")
        pieces = []
        feasible = True
        for group_index, (group, count) in enumerate(zip(groups, pattern)):
            key = (group_index, count)
            group_states = cache.get(key)
            if group_states is None:
                group_states = _group_packed_states(group, count, codec)
                cache[key] = group_states
            if not group_states:
                feasible = False
                break
            pieces.append(group_states)
        if not feasible:
            continue

        if len(pieces) == 1:
            states.extend(pieces[0])
        elif len(pieces) == 2:
            left_states, right_states = pieces
            states.extend(
                left | right
                for left in left_states
                for right in right_states
            )
        else:
            def combine(group_index, packed):
                if group_index == len(pieces):
                    states.append(packed)
                    return
                for piece in pieces[group_index]:
                    combine(group_index + 1, packed | piece)

            combine(0, 0)

    if not states:
        raise ValueError("Particle-number constraints are mutually incompatible.")

    states.sort()
    return FockBasis._from_sorted_packed_states(n_modes, N, states)


class NParticleSector:
    """Fixed-particle-number sector of bosonic Fock space.

    Parameters
    ----------
    modes : BosonModes
        Bosonic modes defining the occupation-number representation.
    N : int
        Number of bosons in the sector.

    Notes
    -----
    Construction records the sector specification but does not generate the
    many-body basis. Additional particle-number constraints associated with
    named degrees of freedom can be imposed with :meth:`project_particles`
    before :meth:`build`
    is called. The constrained basis is generated directly rather than by
    filtering the complete fixed-``N`` sector.
    """

    __slots__ = ("modes", "N", "_basis", "_projected_particle_numbers")

    def __init__(self, modes, N):
        if not isinstance(modes, BosonModes):
            raise TypeError("'modes' must be a BosonModes instance.")
        if not isinstance(N, Integral):
            raise TypeError("'N' must be an integer.")
        N = int(N)
        if N < 0:
            raise ValueError("'N' must be non-negative.")

        self.modes = modes
        self.N = N
        self._basis = None
        self._projected_particle_numbers = {}

    @property
    def is_built(self):
        """bool: Whether the many-body basis has been generated."""
        return self._basis is not None

    def project_particles(
        self, dof_name, particle_numbers=None, /, **named_particle_numbers
    ):
        """Fix particle numbers for values of one degree of freedom.

        Parameters
        ----------
        dof_name : str
            Name of the degree of freedom to constrain.
        particle_numbers : mapping, optional
            Mapping from DoF value indices or string labels to particle
            numbers. Integer keys work even when the :class:`DoF` has no
            descriptive labels.
        **named_particle_numbers : int
            Convenience form for string-labeled DoFs. For example,
            ``project_particles("species", up=2, down=2)`` is equivalent to
            ``project_particles("species", {"up": 2, "down": 2})``.

        Returns
        -------
        NParticleSector
            This sector, so constraints can be chained before :meth:`build`.

        Examples
        --------
        Default integer DoF values can be constrained without inventing
        labels::

            site = DoF(2, name="site")
            modes = BosonModes(site)
            sector = (
                NParticleSector(modes, N=2)
                .project_particles("site", {0: 1, 1: 1})
                .build()
            )

        Descriptive labels retain the keyword shorthand::

            species = DoF(2, name="species", labels=("a", "b"))
            modes = BosonModes(species)
            sector = (
                NParticleSector(modes, N=3)
                .project_particles("species", a=2, b=1)
                .build()
            )
        """
        if self.is_built:
            raise RuntimeError(
                "Particle-number constraints must be specified before sector.build()."
            )
        if not isinstance(dof_name, str):
            raise TypeError("'dof_name' must be a string.")

        matches = [
            index for index, dof in enumerate(self.modes.dofs) if dof.name == dof_name
        ]
        if not matches:
            raise ValueError(f"No degree of freedom named {dof_name!r} exists.")
        if len(matches) > 1:
            raise ValueError(
                f"The degree-of-freedom name {dof_name!r} is ambiguous."
            )

        dof_index = matches[0]
        dof = self.modes.dofs[dof_index]

        if particle_numbers is None:
            entries = []
        else:
            if not isinstance(particle_numbers, Mapping):
                raise TypeError("'particle_numbers' must be a mapping or None.")
            entries = list(particle_numbers.items())
        entries.extend(named_particle_numbers.items())
        if not entries:
            raise ValueError("At least one particle number must be specified.")

        label_to_index = (
            {label: index for index, label in enumerate(dof.labels)}
            if dof.labels is not None
            else {}
        )
        updated = dict(self._projected_particle_numbers.get(dof_index, {}))

        for value, number in entries:
            if isinstance(value, Integral):
                value_index = int(value)
                if value_index < 0 or value_index >= dof.size:
                    raise IndexError(
                        f"Value index {value_index} is outside [0, {dof.size}) "
                        f"for {dof_name!r}."
                    )
                display_value = value_index
            elif isinstance(value, str):
                if value not in label_to_index:
                    if dof.labels is None:
                        raise ValueError(
                            f"The degree of freedom {dof_name!r} has no string "
                            "labels; use integer value indices in a mapping."
                        )
                    allowed = ", ".join(repr(label) for label in dof.labels)
                    raise ValueError(
                        f"Unknown label {value!r} for {dof_name!r}. "
                        f"Expected one of: {allowed}."
                    )
                value_index = label_to_index[value]
                display_value = value
            else:
                raise TypeError(
                    "Projection keys must be integer value indices or string labels."
                )

            if not isinstance(number, Integral):
                raise TypeError("Constrained particle numbers must be integers.")
            number = int(number)
            if number < 0:
                raise ValueError("Constrained particle numbers must be non-negative.")
            if number > self.N:
                raise ValueError(
                    f"Particle number {number} exceeds sector N={self.N}."
                )
            if value_index in updated and updated[value_index] != number:
                raise ValueError(
                    f"A particle number for {dof_name}={display_value!r} "
                    "is already set."
                )
            updated[value_index] = number

        specified = sum(updated.values())
        if specified > self.N:
            raise ValueError(
                "Constrained particle numbers exceed the sector particle number N."
            )

        unspecified_count = dof.size - len(updated)
        remaining = self.N - specified
        if unspecified_count == 0 and remaining != 0:
            raise ValueError(
                "Particle numbers specified for all values must sum to sector N."
            )

        self._projected_particle_numbers[dof_index] = updated
        return self

    def _build_one_dof_projection(self, dof_index, particle_numbers):
        """Build a basis with particle numbers fixed on one projected DoF."""
        dof = self.modes.dofs[dof_index]

        groups = []
        counts = []
        constrained_value_indices = set()

        for value_index, count in particle_numbers.items():
            constrained_value_indices.add(value_index)
            groups.append(_modes_with_dof_value(self.modes, dof_index, value_index))
            counts.append(count)

        unspecified_value_indices = [
            value_index
            for value_index in range(dof.size)
            if value_index not in constrained_value_indices
        ]
        if unspecified_value_indices:
            residual_modes = []
            for value_index in unspecified_value_indices:
                residual_modes.extend(
                    _modes_with_dof_value(self.modes, dof_index, value_index)
                )
            groups.append(tuple(sorted(residual_modes)))
            counts.append(self.N - sum(counts))

        return _basis_from_group_occupations(
            self.modes.n_modes,
            self.N,
            groups,
            (tuple(counts),),
        )

    def _build_two_dof_projection(self, first_index, second_index):
        """Build a basis satisfying projections on two DoFs jointly."""
        first = self.modes.dofs[first_index]
        second = self.modes.dofs[second_index]
        first_numbers = self._projected_particle_numbers[first_index]
        second_numbers = self._projected_particle_numbers[second_index]

        row_targets = _completed_targets(first, first_numbers, self.N)
        column_targets = _completed_targets(second, second_numbers, self.N)

        group_rows = _dof_pair_mode_groups(self.modes, first_index, second_index)
        groups = [group for row in group_rows for group in row]
        capacities = [
            tuple(self.N if group else 0 for group in row) for row in group_rows
        ]

        tables = tuple(
            _two_dof_occupation_tables(
                capacities,
                row_targets,
                column_targets,
                self.N,
            )
        )
        occupations = tuple(
            tuple(value for row in table for value in row) for table in tables
        )

        return _basis_from_group_occupations(
            self.modes.n_modes,
            self.N,
            groups,
            occupations,
        )

    def _build_multi_dof_projection(self, projected_indices):
        """Build a basis satisfying projections on three or more DoFs jointly."""
        projected_indices = tuple(projected_indices)
        targets = tuple(
            _completed_targets(
                self.modes.dofs[dof_index],
                self._projected_particle_numbers[dof_index],
                self.N,
            )
            for dof_index in projected_indices
        )

        coordinates, groups = _multi_dof_mode_groups(self.modes, projected_indices)
        capacities = tuple(self.N if group else 0 for group in groups)
        occupations = tuple(
            _multi_dof_occupation_patterns(
                capacities,
                coordinates,
                targets,
                self.N,
            )
        )

        return _basis_from_group_occupations(
            self.modes.n_modes,
            self.N,
            groups,
            occupations,
        )

    def build(self):
        """Generate the many-body basis and return this sector.

        Returns
        -------
        NParticleSector
            This sector, with its basis available through :attr:`basis`.

        Notes
        -----
        Calling ``build()`` more than once retains the existing basis. When
        particle-number constraints are present, the constrained basis is
        generated directly.
        """
        if self._basis is None:
            projected = tuple(self._projected_particle_numbers)
            if not projected:
                self._basis = FockBasis(self.modes.n_modes, self.N)
            elif len(projected) == 1:
                dof_index = projected[0]
                self._basis = self._build_one_dof_projection(
                    dof_index,
                    self._projected_particle_numbers[dof_index],
                )
            elif len(projected) == 2:
                self._basis = self._build_two_dof_projection(*projected)
            else:
                self._basis = self._build_multi_dof_projection(projected)
        return self

    @property
    def basis(self):
        """FockBasis: Ordered basis of the built sector."""
        if self._basis is None:
            raise RuntimeError(
                "The NParticleSector has not been built. Call sector.build() "
                "before accessing its basis."
            )
        return self._basis

    @property
    def dimension(self):
        """int: Hilbert-space dimension of the built sector."""
        return self.basis.dimension

    def from_vector(self, coefficients):
        """Construct a basis-backed bosonic ket from coefficient coordinates.

        Parameters
        ----------
        coefficients : array_like
            One-dimensional coefficient array in ``self.basis`` ordering.

        Returns
        -------
        FockVector
            Ket represented in this bosonic sector.

        Raises
        ------
        RuntimeError
            If :meth:`build` has not been called.
        """
        if not self.is_built:
            raise RuntimeError(
                "The NParticleSector has not been built. Call sector.build() "
                "before constructing a FockVector."
            )
        return FockVector(coefficients, self)
