"""Shared literal-algebra builders for statistics-independent operators."""

from __future__ import annotations


def conjugate(value):
    """Return the complex conjugate of a numerical coupling."""
    method = getattr(value, "conjugate", None)
    return method() if method is not None else value


def hopping(creation, annihilation, i, j, t, modes):
    """Build a Hermitian one-particle hopping expression."""
    forward = creation(i, modes=modes) * annihilation(j, modes=modes)
    reverse = creation(j, modes=modes) * annihilation(i, modes=modes)
    return t * forward + conjugate(t) * reverse


def onsite(number, i, epsilon, modes):
    """Build an onsite one-body energy."""
    return epsilon * number(i, modes=modes)


def density_density(number, i, j, coupling, modes):
    """Build a density-density interaction."""
    return (
        coupling
        * number(i, modes=modes)
        * number(j, modes=modes)
    )


def spin_plus(creation, annihilation, up, down, modes):
    """Build the standard two-component raising operator."""
    return creation(up, modes=modes) * annihilation(down, modes=modes)


def spin_minus(creation, annihilation, up, down, modes):
    """Build the standard two-component lowering operator."""
    return creation(down, modes=modes) * annihilation(up, modes=modes)


def spin_x(spin_plus_builder, spin_minus_builder, up, down, modes):
    """Build Sx from raising and lowering operators."""
    return 0.5 * (
        spin_plus_builder(up, down, modes)
        + spin_minus_builder(up, down, modes)
    )


def spin_y(spin_plus_builder, spin_minus_builder, up, down, modes):
    """Build Sy from raising and lowering operators."""
    return (-0.5j) * (
        spin_plus_builder(up, down, modes)
        - spin_minus_builder(up, down, modes)
    )


def spin_z(number, up, down, modes):
    """Build Sz from two number operators."""
    return 0.5 * (
        number(up, modes=modes) - number(down, modes=modes)
    )


def heisenberg_exchange(
    spin_plus_builder,
    spin_minus_builder,
    spin_z_builder,
    up_i,
    down_i,
    up_j,
    down_j,
    coupling,
    modes,
):
    """Build an isotropic two-component Heisenberg exchange expression."""
    transverse = 0.5 * (
        spin_plus_builder(up_i, down_i, modes)
        * spin_minus_builder(up_j, down_j, modes)
        + spin_minus_builder(up_i, down_i, modes)
        * spin_plus_builder(up_j, down_j, modes)
    )
    longitudinal = (
        spin_z_builder(up_i, down_i, modes)
        * spin_z_builder(up_j, down_j, modes)
    )
    return coupling * (longitudinal + transverse)
