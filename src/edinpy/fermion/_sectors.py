"""Many-body sectors and sector basis construction."""

from __future__ import annotations

from itertools import product
from math import comb, prod
from numbers import Integral

import numpy as np

from ._basis import FockBasis, FockVector
from ._modes import FermionModes


def _fixed_count_masks_uint64(mode_indices, count):
    """Return fixed-population masks as a compact ``uint64`` array."""
    mode_indices = tuple(mode_indices)
    local_states = FockBasis._enumerate_states(len(mode_indices), count)

    if not mode_indices:
        return np.asarray(local_states, dtype=np.uint64)

    start = mode_indices[0]
    if mode_indices == tuple(range(start, start + len(mode_indices))):
        masks = np.asarray(local_states, dtype=np.uint64)
        if start:
            masks = masks << np.uint64(start)
        return masks

    bit_values = [np.uint64(1) << np.uint64(mode) for mode in mode_indices]
    masks = np.empty(len(local_states), dtype=np.uint64)
    for index, local_state in enumerate(local_states):
        state = np.uint64(0)
        while local_state:
            lowest = local_state & -local_state
            local_index = lowest.bit_length() - 1
            state |= bit_values[local_index]
            local_state ^= lowest
        masks[index] = state
    return masks


def _python_states_to_words(states, n_bits):
    """Convert non-negative Python integers to little-endian ``uint64`` words."""
    states = tuple(states)
    n_words = max(1, (int(n_bits) + 63) // 64)
    words = np.empty((len(states), n_words), dtype=np.uint64)
    mask = (1 << 64) - 1

    for word in range(n_words):
        shift = 64 * word
        words[:, word] = np.fromiter(
            ((state >> shift) & mask for state in states),
            dtype=np.uint64,
            count=len(states),
        )
    return words


def _fixed_count_masks_words(mode_indices, count, n_words):
    """Return fixed-population masks as little-endian ``uint64`` words."""
    mode_indices = tuple(mode_indices)
    local_states = FockBasis._enumerate_states(len(mode_indices), count)
    masks = np.zeros((len(local_states), n_words), dtype=np.uint64)

    if not mode_indices:
        return masks

    local_word_count = max(1, (len(mode_indices) + 63) // 64)
    if len(mode_indices) <= 64:
        local_words = np.asarray(local_states, dtype=np.uint64)[:, np.newaxis]
    else:
        local_words = _python_states_to_words(local_states, len(mode_indices))

    start = mode_indices[0]
    if mode_indices == tuple(range(start, start + len(mode_indices))):
        destination_word = start // 64
        offset = start % 64
        for local_word in range(local_word_count):
            source = local_words[:, local_word]
            target = destination_word + local_word
            masks[:, target] |= source << np.uint64(offset)
            if offset and target + 1 < n_words:
                masks[:, target + 1] |= source >> np.uint64(64 - offset)
        return masks

    for local_index, mode in enumerate(mode_indices):
        source_word = local_index // 64
        source_bit = local_index % 64
        occupied = (
            local_words[:, source_word] >> np.uint64(source_bit)
        ) & np.uint64(1)
        masks[:, mode // 64] |= occupied << np.uint64(mode % 64)
    return masks


def _sorted_python_states_from_words(words):
    """Return integer states sorted by the numeric value of multiword masks."""
    words = np.asarray(words, dtype=np.uint64)
    if words.ndim != 2 or words.shape[1] == 0:
        raise ValueError("'words' must have shape (n_states, n_words).")

    order = np.lexsort(tuple(words[:, word] for word in range(words.shape[1])))
    words = words[order]

    states = words[:, 0].astype(object)
    for word in range(1, words.shape[1]):
        states += words[:, word].astype(object) << (64 * word)
    return tuple(states.tolist())


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

    return tuple(
        tuple(tuple(cell) for cell in row)
        for row in groups
    )


def _multi_dof_mode_groups(modes, dof_indices):
    """Partition modes by the joint values of several projected DoFs."""
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
    """Return per-value targets, inferring the only unspecified value when possible."""
    label_to_index = {label: index for index, label in enumerate(dof.labels)}
    targets = [None] * dof.size
    for label, number in particle_numbers.items():
        targets[label_to_index[label]] = number

    unspecified = [index for index, value in enumerate(targets) if value is None]
    if len(unspecified) == 1:
        targets[unspecified[0]] = total - sum(
            value for value in targets if value is not None
        )
    return tuple(targets)


def _bounded_compositions(total, upper_bounds):
    """Yield integer vectors with a fixed sum and component-wise upper bounds."""
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
    """Yield feasible cell occupations for two projected degrees of freedom."""
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
                        if residual < 0 or residual > future_column_capacity[row + 1][column]:
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
    """Yield feasible cell occupations for three or more projected DoFs."""
    capacities = tuple(int(value) for value in capacities)
    coordinates = tuple(
        tuple(value for value in coordinate) for coordinate in coordinates
    )
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

    suffix_constraint_capacity = [
        [0] * (n_cells + 1) for _ in constraints
    ]
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


def _basis_from_group_occupations(n_modes, N, groups, occupations):
    """Build a sorted basis from disjoint mode groups and allowed occupations."""
    groups = tuple(tuple(group) for group in groups)
    occupations = tuple(tuple(int(value) for value in pattern) for pattern in occupations)
    if not occupations:
        raise ValueError("Particle projections are mutually incompatible.")

    capacities = tuple(len(group) for group in groups)
    dimensions = [
        prod(comb(capacity, count) for capacity, count in zip(capacities, pattern))
        for pattern in occupations
    ]
    dimension = sum(dimensions)

    if n_modes <= 64:
        cache = {}
        states = np.empty(dimension, dtype=np.uint64)
        offset = 0

        for pattern, pattern_dimension in zip(occupations, dimensions):
            pieces = np.asarray([0], dtype=np.uint64)
            for group_index, (group, count) in enumerate(zip(groups, pattern)):
                key = (group_index, count)
                masks = cache.get(key)
                if masks is None:
                    masks = _fixed_count_masks_uint64(group, count)
                    cache[key] = masks
                pieces = np.bitwise_or(
                    pieces[:, np.newaxis],
                    masks[np.newaxis, :],
                ).reshape(-1)
            states[offset : offset + pattern_dimension] = pieces
            offset += pattern_dimension

        states.sort()
        return FockBasis._from_sorted_states(n_modes, N, states)

    n_words = (n_modes + 63) // 64
    cache = {}
    states = np.empty((dimension, n_words), dtype=np.uint64)
    offset = 0

    for pattern, pattern_dimension in zip(occupations, dimensions):
        pieces = np.zeros((1, n_words), dtype=np.uint64)
        for group_index, (group, count) in enumerate(zip(groups, pattern)):
            key = (group_index, count)
            masks = cache.get(key)
            if masks is None:
                masks = _fixed_count_masks_words(group, count, n_words)
                cache[key] = masks
            pieces = np.bitwise_or(
                pieces[:, np.newaxis, :],
                masks[np.newaxis, :, :],
            ).reshape(-1, n_words)

        states[offset : offset + pattern_dimension] = pieces
        offset += pattern_dimension

    sorted_states = _sorted_python_states_from_words(states)
    return FockBasis._from_sorted_states(n_modes, N, sorted_states)


class NParticleSector:
    """Fixed-particle-number sector of fermionic Fock space.

    Parameters
    ----------
    modes : FermionModes
        Fermionic modes defining the occupation-number representation.
    N : int
        Number of fermions in the sector.

    Notes
    -----
    Construction records the sector specification but does not generate the
    many-body basis. Particle numbers associated with named degrees of
    freedom can be fixed with :meth:`project_particles` before :meth:`build`
    is called.
    """

    __slots__ = (
        "modes",
        "N",
        "_basis",
        "_projected_particle_numbers",
    )

    def __init__(self, modes, N):
        """Create an unbuilt fixed-particle-number sector specification."""
        if not isinstance(modes, FermionModes):
            raise TypeError("'modes' must be a FermionModes instance.")
        if not isinstance(N, Integral):
            raise TypeError("'N' must be an integer.")
        N = int(N)
        if N < 0 or N > modes.n_modes:
            raise ValueError("'N' must satisfy 0 <= N <= modes.n_modes.")

        self.modes = modes
        self.N = N
        self._basis = None
        self._projected_particle_numbers = {}

    @property
    def is_built(self):
        """bool: Whether the many-body basis has been generated."""
        return self._basis is not None

    def project_particles(self, dof_name, /, **particle_numbers):
        """Fix particle numbers for labeled values of a degree of freedom.

        Parameters
        ----------
        dof_name : str
            Name of a degree of freedom in ``self.modes``.
        **particle_numbers : int
            Required particle number for each labeled value of the selected
            degree of freedom. The labels must have been supplied when the
            :class:`DoF` was constructed.

        Returns
        -------
        NParticleSector
            This unbuilt sector, allowing method chaining before :meth:`build`.

        Examples
        --------
        Particle numbers can be fixed for one degree of freedom::

            sector = (
                NParticleSector(modes, N=4)
                .project_particles("spin", up=2, down=2)
                .build()
            )

        Multiple projections are solved jointly rather than applied
        sequentially::

            sector = (
                NParticleSector(modes, N=4)
                .project_particles("spin", up=2, down=2)
                .project_particles("layer", top=2, bottom=2)
                .project_particles("orbital", a=2, b=2)
                .build()
            )

        Notes
        -----
        The projected basis is generated directly at :meth:`build` time. The
        complete fixed-``N`` basis is not constructed and filtered.
        """
        if self.is_built:
            raise RuntimeError(
                "Particle projections must be specified before sector.build()."
            )
        if not isinstance(dof_name, str):
            raise TypeError("'dof_name' must be a string.")
        if not particle_numbers:
            raise ValueError("At least one particle number must be specified.")

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
        if dof.labels is None:
            raise ValueError(
                f"The degree of freedom {dof_name!r} has no labels. "
                "Define labels on the DoF before using project_particles()."
            )

        label_to_index = {label: index for index, label in enumerate(dof.labels)}
        capacity = self.modes.n_modes // dof.size
        updated = dict(self._projected_particle_numbers.get(dof_index, {}))

        for label, number in particle_numbers.items():
            if label not in label_to_index:
                allowed = ", ".join(repr(value) for value in dof.labels)
                raise ValueError(
                    f"Unknown label {label!r} for {dof_name!r}. "
                    f"Expected one of: {allowed}."
                )
            if not isinstance(number, Integral):
                raise TypeError("Projected particle numbers must be integers.")
            number = int(number)
            if number < 0:
                raise ValueError("Projected particle numbers must be non-negative.")
            if number > capacity:
                raise ValueError(
                    f"Particle number {number} exceeds the {capacity} modes "
                    f"available for {dof_name}={label!r}."
                )
            if label in updated and updated[label] != number:
                raise ValueError(
                    f"A particle number for {dof_name}={label!r} is already set."
                )
            updated[label] = number

        specified = sum(updated.values())
        if specified > self.N:
            raise ValueError(
                "Projected particle numbers exceed the sector particle number N."
            )

        unspecified_count = dof.size - len(updated)
        remaining = self.N - specified
        if unspecified_count == 0 and remaining != 0:
            raise ValueError(
                "Particle numbers specified for all values must sum to sector N."
            )

        remaining_capacity = unspecified_count * capacity
        if remaining > remaining_capacity:
            raise ValueError(
                "The unspecified values of the projected degree of freedom do "
                "not have enough modes for the remaining particles."
            )

        self._projected_particle_numbers[dof_index] = updated
        return self

    def _build_one_dof_projection(self, dof_index, particle_numbers):
        """Build a basis with particle numbers fixed on one projected DoF."""
        dof = self.modes.dofs[dof_index]
        label_to_index = {label: index for index, label in enumerate(dof.labels)}

        groups = []
        counts = []
        constrained_value_indices = set()

        for label, count in particle_numbers.items():
            value_index = label_to_index[label]
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
        """Build a basis satisfying particle projections on two DoFs jointly."""
        first = self.modes.dofs[first_index]
        second = self.modes.dofs[second_index]
        first_numbers = self._projected_particle_numbers[first_index]
        second_numbers = self._projected_particle_numbers[second_index]

        row_targets = _completed_targets(first, first_numbers, self.N)
        column_targets = _completed_targets(second, second_numbers, self.N)

        group_rows = _dof_pair_mode_groups(
            self.modes,
            first_index,
            second_index,
        )
        groups = [group for row in group_rows for group in row]
        capacities = [tuple(len(group) for group in row) for row in group_rows]

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

        coordinates, groups = _multi_dof_mode_groups(
            self.modes,
            projected_indices,
        )
        capacities = tuple(len(group) for group in groups)
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
        Calling ``build()`` more than once is safe. The existing basis is
        retained rather than regenerated.
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
        """FockBasis: Ordered basis of the built sector.

        Raises
        ------
        RuntimeError
            If :meth:`build` has not been called.
        """
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
        """Construct a basis-backed Fock vector from coefficient coordinates.

        Parameters
        ----------
        coefficients : array_like
            One-dimensional coefficient array in ``self.basis`` ordering.

        Returns
        -------
        FockVector
            Ket represented in the fixed-``N`` Fock basis of this sector.

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
