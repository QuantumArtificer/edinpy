"""Generate all scientific figures used by the EDinPy documentation.

Run from the repository root with::

    PYTHONPATH=src python docs/scripts/generate_figures.py

Many-body model figures are produced with the public EDinPy API. Analytical
curves are used where the corresponding result is stated in the documentation.
Performance figures read versioned benchmark snapshots from ``docs/data``.
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.special import j0, j1

from edinpy import boson as edb
from edinpy import fermion as edf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "source" / "_static" / "figures"
DATA = ROOT / "docs" / "data"
OUT.mkdir(parents=True, exist_ok=True)

# Make committed SVG assets reproducible across repeated local builds.
mpl.rcParams["svg.hashsalt"] = "edinpy-docs"
mpl.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.0,
    }
)

UP, DOWN = 0, 1


def save(fig, name: str):
    """Save one documentation figure as a whitespace-clean SVG."""
    path = OUT / name
    fig.savefig(path,
                bbox_inches="tight",
                metadata={
                    "Date": None,
                    "Creator": "EDinPy documentation",
                         },
                )
    plt.close(fig)
    text = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in text.splitlines()) + "\n",
        encoding="utf-8",
    )
    print(path.relative_to(ROOT))


def spinful_chain(
    L: int,
    *,
    N: int | None = None,
    t: float = 1.0,
    U: float = 0.0,
    V: float = 0.0,
    periodic: bool = True,
):
    """Return a spinful Hubbard or extended-Hubbard chain."""
    if N is None:
        N = L

    site = edf.DoF(L, name="site")
    spin = edf.DoF(2, name="spin")
    modes = edf.FermionModes(site, spin)
    sector = edf.NParticleSector(modes, N=N).build()
    c = edf.set_notation(edf.Annihilation, modes)
    cd = edf.set_notation(edf.Creation, modes)
    n = edf.set_notation(edf.Number, modes)

    bonds = [(i, i + 1) for i in range(L - 1)]
    if periodic and L > 2:
        bonds.append((L - 1, 0))

    H = 0
    for i, j in bonds:
        for sigma in (UP, DOWN):
            hop = cd(i, sigma) * c(j, sigma)
            H += -t * (hop + hop.dag)
    for i in range(L):
        H += U * n(i, UP) * n(i, DOWN)
    for i, j in bonds:
        n_i = n(i, UP) + n(i, DOWN)
        n_j = n(j, UP) + n(j, DOWN)
        H += V * n_i * n_j

    return {
        "modes": modes,
        "sector": sector,
        "c": c,
        "cd": cd,
        "n": n,
        "H": H,
        "hamiltonian": edf.Hamiltonian(H, sector),
        "bonds": bonds,
    }


def spin_operators(model):
    """Return local S^z operators for a spinful chain model."""
    modes = model["modes"]
    L = model["sector"].modes.dofs[0].size
    return [
        edf.SpinZ((i, UP), (i, DOWN), modes)
        for i in range(L)
    ]


def charge_operators(model, filling: float = 1.0):
    """Return local connected density operators n_i-filling."""
    n = model["n"]
    L = model["sector"].modes.dofs[0].size
    return [n(i, UP) + n(i, DOWN) - filling for i in range(L)]


def pair_operators(model):
    """Return local on-site singlet-pair annihilation operators."""
    c = model["c"]
    L = model["sector"].modes.dofs[0].size
    return [c(i, DOWN) * c(i, UP) for i in range(L)]


def bond_operators(model):
    """Return kinetic bond operators B_i for the chain bonds."""
    c = model["c"]
    cd = model["cd"]
    bonds = model["bonds"]
    result = []
    for i, j in bonds:
        B = 0
        for sigma in (UP, DOWN):
            hop = cd(i, sigma) * c(j, sigma)
            B += hop + hop.dag
        result.append(B)
    return result


def structure_factor(psi, local_operators, q):
    r"""Return L^{-1}<O_q^dag O_q> for O_q=sum_j exp(-iqj) O_j."""
    L = len(local_operators)
    O_q = 0
    for j, operator in enumerate(local_operators):
        O_q += np.exp(-1j * q * j) * operator
    return float(np.real(psi.dag * O_q.dag * O_q * psi) / L)


def translational_correlation(psi, local_operators, r):
    """Return the translationally averaged equal-time two-point correlator."""
    L = len(local_operators)
    value = 0.0
    for i, left in enumerate(local_operators):
        right = local_operators[(i + r) % L]
        value += np.real(psi.dag * left * right * psi)
    return float(value / L)


def connected_structure_factor(psi, local_operators, q):
    """Return the connected structure factor for local operators."""
    raw = structure_factor(psi, local_operators, q)
    mean_q = 0.0j
    for j, operator in enumerate(local_operators):
        mean = psi.dag * operator * psi
        mean_q += np.exp(-1j * q * j) * mean
    return float(raw - abs(mean_q) ** 2 / len(local_operators))


def dimer_site_label(state: int):
    """Return a compact physical site-occupation label for the dimer basis."""
    site_labels = []
    for site in range(2):
        up = bool(state & (1 << site))
        down = bool(state & (1 << (site + 2)))
        if up and down:
            label = "↑↓"
        elif up:
            label = "↑"
        elif down:
            label = "↓"
        else:
            label = "0"
        site_labels.append(label)
    return f"|{site_labels[0]}, {site_labels[1]}⟩"


def hubbard_dimer_observables():
    """Generate single-purpose Hubbard-dimer observables versus U/t."""
    U_values = np.linspace(0.0, 12.0, 49)
    d_ed = []
    spin_ed = []
    gap_ed = []

    for U in U_values:
        model = spinful_chain(2, U=U, periodic=False)
        ham = model["hamiltonian"]
        energies, _ = ham.eigsolve(k=None)
        psi0 = ham.eigenstate(0)
        n = model["n"]
        modes = model["modes"]

        D = n(0, UP) * n(0, DOWN) + n(1, UP) * n(1, DOWN)
        Sdot = edf.HeisenbergExchange(
            (0, UP), (0, DOWN), (1, UP), (1, DOWN), 1.0, modes
        )
        d_ed.append(np.real(psi0.dag * D * psi0))
        spin_ed.append(np.real(psi0.dag * Sdot * psi0))
        gap_ed.append(energies[1] - energies[0])

    U_exact = np.linspace(0.0, 12.0, 500)
    root = np.sqrt(U_exact**2 + 16.0)
    d_exact = 0.5 * (1.0 - U_exact / root)
    spin_exact = -0.75 * (1.0 - d_exact)
    gap_exact = 0.5 * (root - U_exact)

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.plot(U_exact, d_exact, label="exact")
    ax.plot(U_values, d_ed, "o", label="ED")
    ax.set_xlabel(r"$U/t$")
    ax.set_ylabel(r"$\langle D\rangle$")
    ax.legend(frameon=True)
    save(fig, "example_hubbard_dimer_double_occupancy.svg")

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.plot(U_exact, spin_exact, label="exact")
    ax.plot(U_values, spin_ed, "o", label="ED")
    ax.axhline(-0.75, linestyle="--", linewidth=1.0, label=r"$-3/4$")
    ax.set_xlabel(r"$U/t$")
    ax.set_ylabel(r"$\langle\mathbf{S}_0\!\cdot\!\mathbf{S}_1\rangle$")
    ax.legend(frameon=True)
    save(fig, "example_hubbard_dimer_spin_correlation.svg")

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.plot(U_exact, gap_exact, label="exact")
    mask = U_exact >= 2.0
    ax.plot(U_exact[mask], 4.0 / U_exact[mask], "--", label=r"$4t^2/U$")
    ax.plot(U_values[1:], np.asarray(gap_ed)[1:], "o", label="ED")
    ax.set_xlabel(r"$U/t$")
    ax.set_ylabel(r"$\Delta_{ST}/t$")
    ax.legend(frameon=True)
    save(fig, "example_hubbard_dimer_gap.svg")



def hubbard_dimer_spectrum():
    """Complete half-filled Hubbard-dimer spectrum as a function of U/t."""
    U_values = np.linspace(0.0, 12.0, 97)
    spectra = []
    for U in U_values:
        model = spinful_chain(2, U=U, periodic=False)
        energies, _ = model["hamiltonian"].eigsolve(k=None)
        spectra.append(energies)
    spectra = np.asarray(spectra)

    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    ax.plot(U_values, spectra[:, 0], label=r"$E_0$")
    ax.plot(U_values, spectra[:, 1], label=r"$S=1$")
    ax.plot(U_values, spectra[:, 4], label=r"$E_U$")
    ax.plot(U_values, spectra[:, 5], label=r"$E_+$")
    ax.set_xlabel(r"$U/t$")
    ax.set_ylabel(r"$E/t$")
    ax.legend(frameon=True)
    save(fig, "hubbard_dimer_spectrum.svg")

def hubbard_dimer_wavefunction():
    """Ground-state probabilities in the six-dimensional dimer basis."""
    model = spinful_chain(2, U=4.0, periodic=False)
    ham = model["hamiltonian"]
    ham.eigsolve(k=None)
    psi0 = ham.eigenstate(0)
    states = model["sector"].basis.states
    labels = [dimer_site_label(state) for state in states]
    probabilities = np.abs(psi0.coefficients) ** 2

    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    ax.bar(labels, probabilities)
    ax.set_ylabel(r"$|\langle\alpha|\psi_0\rangle|^2$")
    ax.tick_params(axis="x", rotation=25)
    save(fig, "hubbard_dimer_wavefunction.svg")


def hubbard_chain_spectrum_observables():
    """Generate Hubbard-ring spectra and local observables versus U/t."""
    L = 6
    U_values = np.linspace(0.0, 8.0, 33)
    levels = []
    double_occupancy = []
    local_moment = []

    for U in U_values:
        model = spinful_chain(L, U=U, periodic=True)
        ham = model["hamiltonian"]
        energies, _ = ham.eigsolve(k=8, which="SA", v0=np.ones(ham.sector.dimension))
        psi0 = ham.eigenstate(0)
        n = model["n"]
        levels.append(energies - energies[0])
        double_occupancy.append(
            sum(
                np.real(psi0.dag * n(i, UP) * n(i, DOWN) * psi0)
                for i in range(L)
            )
            / L
        )
        local_moment.append(
            sum(
                np.real(
                    psi0.dag
                    * (n(i, UP) - n(i, DOWN))
                    * (n(i, UP) - n(i, DOWN))
                    * psi0
                )
                for i in range(L)
            )
            / L
        )

    levels = np.asarray(levels)

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for level in range(1, levels.shape[1]):
        ax.plot(U_values, levels[:, level], linewidth=1.1)
    ax.set_xlabel(r"$U/t$")
    ax.set_ylabel(r"$(E_n-E_0)/t$")
    save(fig, "example_hubbard_chain_spectrum.svg")

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.plot(U_values, double_occupancy, "o-")
    ax.set_xlabel(r"$U/t$")
    ax.set_ylabel(r"$D$")
    save(fig, "example_hubbard_chain_double_occupancy.svg")

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.plot(U_values, local_moment, "o-")
    ax.set_xlabel(r"$U/t$")
    ax.set_ylabel(r"$\mu^2$")
    save(fig, "example_hubbard_chain_local_moment.svg")



def hubbard_chain_correlations():
    """Generate separate real-space spin and charge correlations."""
    L = 6
    distances = np.arange(L // 2 + 1)
    data = {}
    for U in (0.0, 4.0, 8.0):
        model = spinful_chain(L, U=U, periodic=True)
        ham = model["hamiltonian"]
        ham.eigsolve(k=1, which="SA", v0=np.ones(ham.sector.dimension))
        psi0 = ham.eigenstate(0)
        spins = spin_operators(model)
        charges = charge_operators(model)
        spin_corr = [translational_correlation(psi0, spins, int(r)) for r in distances]
        charge_corr = [translational_correlation(psi0, charges, int(r)) for r in distances]
        data[U] = (spin_corr, charge_corr)

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for U, (spin_corr, _charge_corr) in data.items():
        ax.plot(distances, spin_corr, "o-", label=fr"$U/t={U:g}$")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xticks(distances)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$C_s(r)$")
    ax.legend(frameon=True)
    save(fig, "example_hubbard_chain_spin_correlations.svg")

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for U, (_spin_corr, charge_corr) in data.items():
        ax.plot(distances, charge_corr, "o-", label=fr"$U/t={U:g}$")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xticks(distances)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$C_c(r)$")
    ax.legend(frameon=True)
    save(fig, "example_hubbard_chain_charge_correlations.svg")


def hubbard_chain_structure_factors():
    """Generate separate momentum-resolved spin, charge, and pair structure factors."""
    L = 6
    q_values = 2.0 * np.pi * np.arange(L) / L
    q_over_pi = q_values / np.pi
    data = {}
    for U in (0.0, 4.0, 8.0):
        model = spinful_chain(L, U=U, periodic=True)
        ham = model["hamiltonian"]
        ham.eigsolve(k=1, which="SA", v0=np.ones(ham.sector.dimension))
        psi0 = ham.eigenstate(0)
        data[U] = (
            [structure_factor(psi0, spin_operators(model), q) for q in q_values],
            [structure_factor(psi0, charge_operators(model), q) for q in q_values],
            [structure_factor(psi0, pair_operators(model), q) for q in q_values],
        )

    for filename, ylabel, index in (
        ("example_hubbard_chain_spin_structure.svg", r"$S_s(q)$", 0),
        ("example_hubbard_chain_charge_structure.svg", r"$S_c(q)$", 1),
        ("example_hubbard_chain_pair_structure_q.svg", r"$P(q)$", 2),
    ):
        fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
        for U, values in data.items():
            ax.plot(q_over_pi, values[index], "o-", label=fr"$U/t={U:g}$")
        ax.set_xlabel(r"$q/\pi$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(q_over_pi)
        ax.legend(frameon=True)
        save(fig, filename)


def extended_hubbard_competition():
    """Generate staggered extended-Hubbard diagnostics versus V/t."""
    L = 6
    U = 4.0
    V_values = np.linspace(0.0, 4.0, 33)
    spin_pi = []
    charge_pi = []
    bond_pi = []
    double_occupancy = []

    for V in V_values:
        model = spinful_chain(L, U=U, V=V, periodic=True)
        ham = model["hamiltonian"]
        ham.eigsolve(k=1, which="SA", v0=np.ones(ham.sector.dimension))
        psi0 = ham.eigenstate(0)
        n = model["n"]
        spin_pi.append(structure_factor(psi0, spin_operators(model), np.pi))
        charge_pi.append(structure_factor(psi0, charge_operators(model), np.pi))
        bond_pi.append(connected_structure_factor(psi0, bond_operators(model), np.pi))
        double_occupancy.append(
            sum(
                np.real(psi0.dag * n(i, UP) * n(i, DOWN) * psi0)
                for i in range(L)
            )
            / L
        )

    for filename, values, ylabel in (
        ("example_extended_hubbard_spin_pi.svg", spin_pi, r"$S_s(\pi)$"),
        ("example_extended_hubbard_charge_pi.svg", charge_pi, r"$S_c(\pi)$"),
        ("example_extended_hubbard_bond_pi.svg", bond_pi, r"$S_B^{\mathrm{conn}}(\pi)$"),
        ("example_extended_hubbard_double_occupancy.svg", double_occupancy, r"$D$"),
    ):
        fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
        ax.plot(V_values, values, "o-")
        ax.axvline(U / 2.0, linestyle="--", linewidth=1.0, label=r"$U/2$")
        ax.set_xlabel(r"$V/t$")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=True)
        save(fig, filename)


def extended_hubbard_structure_profiles():
    """Generate separate momentum profiles across the extended-Hubbard crossover."""
    L = 6
    U = 4.0
    q_values = 2.0 * np.pi * np.arange(L) / L
    q_over_pi = q_values / np.pi
    V_samples = (0.5, 2.25, 3.5)
    data = {}

    for V in V_samples:
        model = spinful_chain(L, U=U, V=V, periodic=True)
        ham = model["hamiltonian"]
        ham.eigsolve(k=1, which="SA", v0=np.ones(ham.sector.dimension))
        psi0 = ham.eigenstate(0)
        bonds = bond_operators(model)
        data[V] = (
            [structure_factor(psi0, spin_operators(model), q) for q in q_values],
            [structure_factor(psi0, charge_operators(model), q) for q in q_values],
            [connected_structure_factor(psi0, bonds, q) for q in q_values],
        )

    for filename, ylabel, index in (
        ("example_extended_hubbard_spin_q.svg", r"$S_s(q)$", 0),
        ("example_extended_hubbard_charge_q.svg", r"$S_c(q)$", 1),
        ("example_extended_hubbard_bond_q.svg", r"$S_B^{\mathrm{conn}}(q)$", 2),
    ):
        fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
        for V, values in data.items():
            ax.plot(q_over_pi, values[index], "o-", label=fr"$V/t={V:g}$")
        ax.set_xlabel(r"$q/\pi$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(q_over_pi)
        ax.legend(frameon=True)
        save(fig, filename)


def spin_exchange_spectrum():
    """Generate the constrained two-spin Heisenberg spectrum and total spin."""
    site = edf.DoF(2, name="site", labels=("left", "right"))
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    modes = edf.FermionModes(site, spin)
    sector = (
        edf.NParticleSector(modes, N=2)
        .project_particles("site", left=1, right=1)
        .build()
    )
    sdot = edf.HeisenbergExchange(
        (0, UP), (0, DOWN), (1, UP), (1, DOWN), 1.0, modes
    )
    splus = sum(
        (edf.SpinPlus((i, UP), (i, DOWN), modes) for i in range(2)), start=0
    )
    sminus = sum(
        (edf.SpinMinus((i, UP), (i, DOWN), modes) for i in range(2)), start=0
    )
    sz = sum(
        (edf.SpinZ((i, UP), (i, DOWN), modes) for i in range(2)), start=0
    )
    s2 = sz * sz + 0.5 * (splus * sminus + sminus * splus)

    J_values = np.linspace(-2.0, 2.0, 81)
    spectra = []
    for J in J_values:
        ham = edf.Hamiltonian(J * sdot, sector)
        energies, _ = ham.eigsolve(k=None)
        spectra.append(energies)
    spectra = np.asarray(spectra)

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for level in range(spectra.shape[1]):
        ax.plot(J_values, spectra[:, level], linewidth=1.0)
    ax.plot(J_values, -0.75 * J_values, "--", label=r"$S=0$")
    ax.plot(J_values, 0.25 * J_values, "--", label=r"$S=1$")
    ax.set_xlabel(r"$J$")
    ax.set_ylabel(r"$E$")
    ax.legend(frameon=True)
    save(fig, "example_fermion_exchange_spectrum.svg")

    ham = edf.Hamiltonian(sdot, sector)
    energies, _ = ham.eigsolve(k=None)
    spin_squared = [
        np.real(ham.eigenstate(i).dag * s2 * ham.eigenstate(i))
        for i in range(sector.dimension)
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.bar(np.arange(sector.dimension), spin_squared)
    ax.set_xticks(np.arange(sector.dimension))
    ax.set_xlabel("eigenstate")
    ax.set_ylabel(r"$\langle S_{\mathrm{tot}}^2\rangle$")
    save(fig, "example_fermion_exchange_total_spin.svg")


def flux_threaded_ring():
    """Generate level flow, ground-state energy, and persistent current."""
    L = 4
    N = 2
    t = 1.0
    phi_values = np.linspace(-2.0 * np.pi, 2.0 * np.pi, 161)
    k_values = 2.0 * np.pi * np.arange(L) / L

    site = edf.DoF(L, name="site")
    modes = edf.FermionModes(site)
    sector = edf.NParticleSector(modes, N=N).build()
    c = edf.set_notation(edf.Annihilation, modes)
    cd = edf.set_notation(edf.Creation, modes)

    single_particle = []
    e_ed = []
    e_exact = []
    i_ed = []
    i_exact = []

    for phi in phi_values:
        theta = phi / L
        phase = np.exp(1j * theta)
        H = 0
        current = 0
        for i in range(L):
            j = (i + 1) % L
            hop = cd(i) * c(j)
            H += -t * (phase * hop + phase.conjugate() * hop.dag)
            current += (t / L) * (
                1j * phase * hop - 1j * phase.conjugate() * hop.dag
            )

        ham = edf.Hamiltonian(H, sector)
        ham.eigsolve(k=None)
        psi0 = ham.eigenstate(0)
        e_ed.append(ham.eigvals[0])
        i_ed.append(np.real(psi0.dag * current * psi0))

        eps = -2.0 * t * np.cos(k_values - theta)
        single_particle.append(eps)
        occupied = np.argsort(eps)[:N]
        e_exact.append(np.sum(eps[occupied]))
        i_exact.append((2.0 * t / L) * np.sum(np.sin(k_values[occupied] - theta)))

    single_particle = np.asarray(single_particle)
    x = phi_values / np.pi

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for m in range(L):
        ax.plot(x, single_particle[:, m], label=fr"$m={m}$")
    ax.set_xlabel(r"$\phi/\pi$")
    ax.set_ylabel(r"$\varepsilon_m/t$")
    ax.legend(frameon=True)
    save(fig, "example_flux_single_particle.svg")

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.plot(x, e_exact, label="exact")
    ax.plot(x[::8], np.asarray(e_ed)[::8], "o", label="ED")
    ax.set_xlabel(r"$\phi/\pi$")
    ax.set_ylabel(r"$E_0/t$")
    ax.legend(frameon=True)
    save(fig, "example_flux_ground_energy.svg")

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.plot(x, i_exact, label="exact")
    ax.plot(x[::8], np.asarray(i_ed)[::8], "o", label="ED")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel(r"$\phi/\pi$")
    ax.set_ylabel(r"$I/t$")
    ax.legend(frameon=True)
    save(fig, "example_flux_current.svg")


def hilbert_space_growth():
    """Compare representative fermionic and bosonic fixed-N dimensions."""
    L_values = np.arange(1, 17)
    fermion_half_filled = np.asarray(
        [comb(2 * L, L) for L in L_values], dtype=float
    )
    boson_unit_filled = np.asarray(
        [comb(2 * L - 1, L) for L in L_values], dtype=float
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.semilogy(
        L_values,
        fermion_half_filled,
        "o-",
        label="fermion",
    )
    ax.semilogy(
        L_values,
        boson_unit_filled,
        "o-",
        label="boson",
    )
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"sector dimension $D$")
    ax.legend(frameon=True)
    save(fig, "hilbert_space_growth.svg")



def bose_hubbard_dimer():
    """Generate single-purpose Bose-Hubbard dimer figures."""
    N = 8
    J = 1.0
    modes = edb.BosonModes(edb.DoF(2, name="well"))
    sector = edb.NParticleSector(modes, N=N).build()
    sx = edb.SpinX(0, 1, modes)
    sz = edb.SpinZ(0, 1, modes)

    U_values = np.linspace(0.0, 16.0, 41)
    levels = []
    coherence = []
    imbalance_variance = []
    distributions = {}

    for U in U_values:
        H = (
            edb.Hopping(0, 1, -J, modes)
            + edb.Hubbard(0, U, modes)
            + edb.Hubbard(1, U, modes)
        )
        ham = edb.Hamiltonian(H, sector)
        energies, _ = ham.eigsolve(k=None)
        psi0 = ham.eigenstate(0)
        levels.append(energies[:5] - energies[0])
        coherence.append(np.real(psi0.dag * sx * psi0) / (N / 2))
        imbalance_variance.append(
            np.real(psi0.dag * sz * sz * psi0) / (N / 2) ** 2
        )
        for target in (0.0, 4.0, 12.0):
            if np.isclose(U, target):
                n_left = np.asarray(
                    [sector.basis.state_at(i).occupations[0] for i in range(sector.dimension)]
                )
                order = np.argsort(n_left)
                distributions[target] = (
                    n_left[order],
                    np.abs(psi0.coefficients[order]) ** 2,
                )

    levels = np.asarray(levels)

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for level in range(1, levels.shape[1]):
        ax.plot(U_values, levels[:, level], linewidth=1.1)
    ax.set_xlabel(r"$U/J$")
    ax.set_ylabel(r"$(E_n-E_0)/J$")
    save(fig, "example_bose_dimer_spectrum.svg")

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.plot(U_values, coherence, "o-")
    ax.set_xlabel(r"$U/J$")
    ax.set_ylabel(r"$\langle S_x\rangle/(N/2)$")
    save(fig, "example_bose_dimer_coherence.svg")

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.plot(U_values, imbalance_variance, "o-")
    ax.set_xlabel(r"$U/J$")
    ax.set_ylabel(r"$\langle S_z^2\rangle/(N/2)^2$")
    save(fig, "example_bose_dimer_imbalance.svg")

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for U, (n_left, probabilities) in sorted(distributions.items()):
        ax.plot(n_left, probabilities, "o-", label=fr"$U/J={U:g}$")
    ax.set_xlabel(r"$n_L$")
    ax.set_ylabel(r"$P(n_L)$")
    ax.legend(frameon=True)
    save(fig, "example_bose_dimer_probabilities.svg")


def _bose_hubbard_ring(L: int, N: int, U: float, J: float = 1.0):
    """Return a periodic Bose-Hubbard ring and its basic operator notation."""
    modes = edb.BosonModes(edb.DoF(L, name="site"))
    sector = edb.NParticleSector(modes, N=N).build()
    b = edb.set_notation(edb.Annihilation, modes)
    bd = edb.set_notation(edb.Creation, modes)
    n = edb.set_notation(edb.Number, modes)
    H = 0
    for i in range(L):
        H += edb.Hopping(i, (i + 1) % L, -J, modes)
        H += edb.Hubbard(i, U, modes)
    return modes, sector, b, bd, n, edb.Hamiltonian(H, sector)


def _boson_one_body_density_matrix(psi, b, bd, L: int):
    """Return rho_ij=<b_i^dag b_j> for one bosonic eigenstate."""
    return np.asarray(
        [[psi.dag * bd(i) * b(j) * psi for j in range(L)] for i in range(L)],
        dtype=complex,
    )


def _boson_density_structure_factor(psi, n, L: int, N: int, q: float):
    """Return the connected density structure factor at lattice momentum q."""
    delta_n_q = sum(
        (
            np.exp(-1j * q * j) * (n(j) - N / L)
            for j in range(L)
        ),
        start=0,
    )
    return float(np.real(psi.dag * delta_n_q.dag * delta_n_q * psi) / L)


def bose_hubbard_chain():
    """Generate finite-ring Bose-Hubbard coherence and density diagnostics."""
    L = 6
    N = 6
    U_values = np.linspace(0.0, 16.0, 17)
    condensate_fraction = []
    number_variance = []
    selected = {}

    for U in U_values:
        _modes, sector, b, bd, n, ham = _bose_hubbard_ring(L, N, U)
        ham.eigsolve(k=1, which="SA", v0=np.ones(sector.dimension))
        psi0 = ham.eigenstate(0)
        rho = _boson_one_body_density_matrix(psi0, b, bd, L)
        occupations = np.linalg.eigvalsh(rho)
        condensate_fraction.append(occupations[-1] / N)
        local_variances = []
        for i in range(L):
            mean = np.real(psi0.dag * n(i) * psi0)
            mean_square = np.real(psi0.dag * n(i) * n(i) * psi0)
            local_variances.append(mean_square - mean**2)
        number_variance.append(np.mean(local_variances))

        for target in (0.0, 4.0, 12.0):
            if np.isclose(U, target):
                g1 = []
                for r in range(L // 2 + 1):
                    value = 0.0j
                    for i in range(L):
                        value += psi0.dag * bd(i) * b((i + r) % L) * psi0
                    g1.append(np.real(value / L))
                q_values = 2.0 * np.pi * np.arange(L) / L
                structure = [
                    _boson_density_structure_factor(psi0, n, L, N, q)
                    for q in q_values
                ]
                selected[target] = (np.asarray(g1), q_values, np.asarray(structure))

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.plot(U_values, condensate_fraction, "o-")
    ax.set_xlabel(r"$U/J$")
    ax.set_ylabel(r"$n_0/N$")
    save(fig, "example_bose_ring_condensate_fraction.svg")

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.plot(U_values, number_variance, "o-")
    ax.set_xlabel(r"$U/J$")
    ax.set_ylabel(r"$\Delta n^2$")
    save(fig, "example_bose_ring_number_variance.svg")

    distances = np.arange(L // 2 + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for U, (g1, _q, _structure) in sorted(selected.items()):
        ax.plot(distances, g1, "o-", label=fr"$U/J={U:g}$")
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$g^{(1)}(r)$")
    ax.set_xticks(distances)
    ax.legend(frameon=True)
    save(fig, "example_bose_ring_g1.svg")

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for U, (_g1, q_values, structure) in sorted(selected.items()):
        ax.plot(q_values / np.pi, structure, "o-", label=fr"$U/J={U:g}$")
    ax.set_xlabel(r"$q/\pi$")
    ax.set_ylabel(r"$S_n(q)$")
    ax.set_xticks(2.0 * np.arange(L) / L)
    ax.legend(frameon=True)
    save(fig, "example_bose_ring_density_structure.svg")


def boson_spin_exchange():
    """Generate the locally constrained Schwinger-boson Heisenberg spectrum."""
    site = edb.DoF(2, name="site", labels=("left", "right"))
    component = edb.DoF(2, name="component", labels=("up", "down"))
    modes = edb.BosonModes(site, component)
    sector = (
        edb.NParticleSector(modes, N=2)
        .project_particles("site", left=1, right=1)
        .build()
    )
    up, down = 0, 1
    sdot = edb.HeisenbergExchange(
        (0, up), (0, down), (1, up), (1, down), 1.0, modes
    )
    splus = sum(
        (edb.SpinPlus((i, up), (i, down), modes) for i in range(2)), start=0
    )
    sminus = sum(
        (edb.SpinMinus((i, up), (i, down), modes) for i in range(2)), start=0
    )
    sz = sum(
        (edb.SpinZ((i, up), (i, down), modes) for i in range(2)), start=0
    )
    s2 = sz * sz + 0.5 * (splus * sminus + sminus * splus)

    J_values = np.linspace(-2.0, 2.0, 81)
    spectra = []
    for J in J_values:
        ham = edb.Hamiltonian(J * sdot, sector)
        energies, _ = ham.eigsolve(k=None)
        spectra.append(energies)
    spectra = np.asarray(spectra)

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for level in range(spectra.shape[1]):
        ax.plot(J_values, spectra[:, level], linewidth=1.0)
    ax.plot(J_values, -0.75 * J_values, "--", label=r"$S=0$")
    ax.plot(J_values, 0.25 * J_values, "--", label=r"$S=1$")
    ax.set_xlabel(r"$J$")
    ax.set_ylabel(r"$E$")
    ax.legend(frameon=True)
    save(fig, "example_boson_exchange_spectrum.svg")

    ham = edb.Hamiltonian(sdot, sector)
    energies, _ = ham.eigsolve(k=None)
    spin_squared = [
        np.real(ham.eigenstate(i).dag * s2 * ham.eigenstate(i))
        for i in range(sector.dimension)
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.bar(np.arange(sector.dimension), spin_squared)
    ax.set_xticks(np.arange(sector.dimension))
    ax.set_xlabel("eigenstate")
    ax.set_ylabel(r"$\langle S_{\mathrm{tot}}^2\rangle$")
    save(fig, "example_boson_exchange_total_spin.svg")



def guide_fermion_ring_current():
    """Persistent current of a half-filled four-site spinless fermion ring."""
    L = 4
    N = 2
    t = 1.0
    phi_values = np.linspace(-2.0 * np.pi, 2.0 * np.pi, 161)
    k_values = 2.0 * np.pi * np.arange(L) / L

    site = edf.DoF(L, name="site")
    modes = edf.FermionModes(site)
    sector = edf.NParticleSector(modes, N=N).build()
    c = edf.set_notation(edf.Annihilation, modes)
    cd = edf.set_notation(edf.Creation, modes)

    current_ed = []
    current_exact = []
    for phi in phi_values:
        theta = phi / L
        phase = np.exp(1j * theta)
        H = 0
        current = 0
        for i in range(L):
            j = (i + 1) % L
            hop = cd(i) * c(j)
            H += -t * (phase * hop + phase.conjugate() * hop.dag)
            current += (t / L) * (
                1j * phase * hop - 1j * phase.conjugate() * hop.dag
            )

        ham = edf.Hamiltonian(H, sector)
        ham.eigsolve(k=None)
        psi0 = ham.eigenstate(0)
        current_ed.append(np.real(psi0.dag * current * psi0))

        eps = -2.0 * t * np.cos(k_values - theta)
        occupied = np.argsort(eps)[:N]
        current_exact.append(
            (2.0 * t / L) * np.sum(np.sin(k_values[occupied] - theta))
        )

    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    x = phi_values / np.pi
    ax.plot(x, current_exact, label="exact")
    ax.plot(x[::8], np.asarray(current_ed)[::8], "o", markersize=3.4, label="ED")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel(r"$\phi/\pi$")
    ax.set_ylabel(r"$I/t$")
    ax.legend(frameon=True)
    save(fig, "guide_fermion_ring_current.svg")



def landing_hubbard_ground_state():
    """Hubbard-dimer reference curves and ED points for the landing page."""
    t = 1.0
    U_reference = np.linspace(0.0, 12.0, 500)
    U_ed = np.linspace(0.0, 12.0, 13)

    root = np.sqrt(U_reference**2 + 16.0 * t**2)
    energy_reference = 0.5 * (U_reference - root) / t
    double_reference = 0.5 * (1.0 - U_reference / root)
    moment_reference = 1.0 - double_reference

    energy_ed = []
    double_ed = []
    moment_ed = []

    for U in U_ed:
        model = spinful_chain(2, U=U, t=t, periodic=False)
        ham = model["hamiltonian"]
        ham.eigsolve(k=None)
        psi0 = ham.eigenstate(0)
        n = model["n"]

        double = (
            n(0, UP) * n(0, DOWN)
            + n(1, UP) * n(1, DOWN)
        )
        moment = 0.5 * (
            (n(0, UP) - n(0, DOWN)) * (n(0, UP) - n(0, DOWN))
            + (n(1, UP) - n(1, DOWN)) * (n(1, UP) - n(1, DOWN))
        )

        energy_ed.append(float(ham.eigvals[0] / t))
        double_ed.append(float(np.real(psi0.dag * double * psi0)))
        moment_ed.append(float(np.real(psi0.dag * moment * psi0)))

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(6.4, 7.0),
        sharex=True,
        constrained_layout=True,
    )

    series = (
        (energy_reference, energy_ed, r"$E_0/t$"),
        (double_reference, double_ed, r"$D$"),
        (moment_reference, moment_ed, r"$\mu^2$"),
    )

    for ax, (reference, ed_values, ylabel) in zip(axes, series):
        ax.plot(U_reference / t, reference, label="analytic")
        ax.plot(
            U_ed / t,
            ed_values,
            linestyle="none",
            marker="o",
            markerfacecolor="none",
            label="ED",
        )
        ax.set_ylabel(ylabel)

    axes[-1].set_xlabel(r"$U/t$")
    axes[0].legend(frameon=True)
    save(fig, "landing_hubbard_ground_state.svg")


def guide_hubbard_low_energy():
    """Low-energy excitation spectrum of a six-site half-filled Hubbard ring."""
    L = 6
    U_values = np.linspace(0.0, 8.0, 33)
    levels = []
    for U in U_values:
        model = spinful_chain(L, U=U, periodic=True)
        ham = model["hamiltonian"]
        energies, _ = ham.eigsolve(
            k=8,
            which="SA",
            v0=np.ones(ham.sector.dimension),
        )
        levels.append(energies - energies[0])
    levels = np.asarray(levels)

    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    for level in range(1, levels.shape[1]):
        ax.plot(U_values, levels[:, level], linewidth=1.1)
    ax.set_xlabel(r"$U/t$")
    ax.set_ylabel(r"$(E_n-E_0)/t$")
    save(fig, "guide_hubbard_low_energy.svg")


def guide_hubbard_double_occupancy():
    """Total double occupancy of the half-filled Hubbard dimer."""
    U_values = np.linspace(0.0, 12.0, 49)
    double_ed = []
    for U in U_values:
        model = spinful_chain(2, U=U, periodic=False)
        ham = model["hamiltonian"]
        ham.eigsolve(k=None)
        psi0 = ham.eigenstate(0)
        n = model["n"]
        D = n(0, UP) * n(0, DOWN) + n(1, UP) * n(1, DOWN)
        double_ed.append(np.real(psi0.dag * D * psi0))

    U_exact = np.linspace(0.0, 12.0, 500)
    double_exact = 0.5 * (
        1.0 - U_exact / np.sqrt(U_exact**2 + 16.0)
    )

    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    ax.plot(U_exact, double_exact, label="exact")
    ax.plot(U_values, double_ed, "o", markersize=3.4, label="ED")
    ax.set_xlabel(r"$U/t$")
    ax.set_ylabel(r"$D$")
    ax.legend(frameon=True)
    save(fig, "guide_hubbard_double_occupancy.svg")


def guide_hubbard_correlations():
    """Separate spin and charge correlation plots for the user guide."""
    L = 6
    distances = np.arange(L // 2 + 1)
    spin_data = {}
    charge_data = {}

    for U in (0.0, 4.0, 8.0):
        model = spinful_chain(L, U=U, periodic=True)
        ham = model["hamiltonian"]
        ham.eigsolve(k=1, which="SA", v0=np.ones(ham.sector.dimension))
        psi0 = ham.eigenstate(0)
        spins = spin_operators(model)
        charges = charge_operators(model)
        spin_data[U] = [
            translational_correlation(psi0, spins, int(r)) for r in distances
        ]
        charge_data[U] = [
            translational_correlation(psi0, charges, int(r)) for r in distances
        ]

    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    for U, values in spin_data.items():
        ax.plot(distances, values, "o-", label=fr"$U/t={U:g}$")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xticks(distances)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$C_s(r)$")
    ax.legend(frameon=True)
    save(fig, "guide_hubbard_spin_correlations.svg")

    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    for U, values in charge_data.items():
        ax.plot(distances, values, "o-", label=fr"$U/t={U:g}$")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xticks(distances)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$C_c(r)$")
    ax.legend(frameon=True)
    save(fig, "guide_hubbard_charge_correlations.svg")


def guide_hubbard_structure_factors():
    """Separate spin and charge structure-factor plots for the user guide."""
    L = 6
    q_values = 2.0 * np.pi * np.arange(L) / L
    q_over_pi = q_values / np.pi
    spin_data = {}
    charge_data = {}

    for U in (0.0, 4.0, 8.0):
        model = spinful_chain(L, U=U, periodic=True)
        ham = model["hamiltonian"]
        ham.eigsolve(k=1, which="SA", v0=np.ones(ham.sector.dimension))
        psi0 = ham.eigenstate(0)
        spin_data[U] = [
            structure_factor(psi0, spin_operators(model), q) for q in q_values
        ]
        charge_data[U] = [
            structure_factor(psi0, charge_operators(model), q) for q in q_values
        ]

    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    for U, values in spin_data.items():
        ax.plot(q_over_pi, values, "o-", label=fr"$U/t={U:g}$")
    ax.set_xticks(q_over_pi)
    ax.set_xlabel(r"$q/\pi$")
    ax.set_ylabel(r"$S_s(q)$")
    ax.legend(frameon=True)
    save(fig, "guide_hubbard_spin_structure_factor.svg")

    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    for U, values in charge_data.items():
        ax.plot(q_over_pi, values, "o-", label=fr"$U/t={U:g}$")
    ax.set_xticks(q_over_pi)
    ax.set_xlabel(r"$q/\pi$")
    ax.set_ylabel(r"$S_c(q)$")
    ax.legend(frameon=True)
    save(fig, "guide_hubbard_charge_structure_factor.svg")


def guide_bose_hubbard_observables():
    """Condensate fraction and onsite number variance of a finite Bose-Hubbard ring."""
    L = 6
    N = 6
    U_values = np.linspace(0.0, 16.0, 17)
    condensate_fraction = []
    number_variance = []

    for U in U_values:
        _modes, sector, b, bd, n, ham = _bose_hubbard_ring(L, N, U)
        ham.eigsolve(k=1, which="SA", v0=np.ones(sector.dimension))
        psi0 = ham.eigenstate(0)
        rho = _boson_one_body_density_matrix(psi0, b, bd, L)
        occupations = np.linalg.eigvalsh(rho)
        condensate_fraction.append(occupations[-1] / N)

        local_variances = []
        for i in range(L):
            mean = np.real(psi0.dag * n(i) * psi0)
            mean_square = np.real(psi0.dag * n(i) * n(i) * psi0)
            local_variances.append(mean_square - mean**2)
        number_variance.append(np.mean(local_variances))

    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    ax.plot(U_values, condensate_fraction, "o-")
    ax.set_xlabel(r"$U/J$")
    ax.set_ylabel(r"$\lambda_{\max}/N$")
    save(fig, "guide_bose_hubbard_condensate_fraction.svg")

    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    ax.plot(U_values, number_variance, "o-")
    ax.set_xlabel(r"$U/J$")
    ax.set_ylabel(r"$\Delta n^2$")
    save(fig, "guide_bose_hubbard_number_variance.svg")


def hubbard_finite_size_energy():
    """Ground-state energy density of half-filled open Hubbard chains."""
    sizes = np.asarray([4, 6, 8, 10], dtype=int)
    U = 4.0
    t = 1.0
    energy_density = []

    for L in sizes:
        site = edf.DoF(int(L), name="site")
        spin = edf.DoF(2, name="spin", labels=("up", "down"))
        modes = edf.FermionModes(site, spin)
        sector = (
            edf.NParticleSector(modes, N=int(L))
            .project_particles("spin", up=int(L // 2), down=int(L // 2))
            .build()
        )
        c = edf.set_notation(edf.Annihilation, modes)
        cd = edf.set_notation(edf.Creation, modes)
        n = edf.set_notation(edf.Number, modes)

        H = 0
        for i in range(int(L) - 1):
            j = i + 1
            for sigma in (UP, DOWN):
                hop = cd(i, sigma) * c(j, sigma)
                H += -t * (hop + hop.dag)
        for i in range(int(L)):
            H += U * n(i, UP) * n(i, DOWN)

        ham = edf.Hamiltonian(H, sector)
        energies, _ = ham.eigsolve(
            k=1,
            which="SA",
            tol=1e-10,
            v0=np.ones(sector.dimension),
        )
        energy_density.append(float(energies[0] / (L * t)))

    def lieb_wu_integrand(omega):
        if omega == 0.0:
            return 0.25
        return (
            j0(omega)
            * j1(omega)
            / (omega * (1.0 + np.exp(U * omega / (2.0 * t))))
        )

    e_infinite = -4.0 * quad(
        lieb_wu_integrand,
        0.0,
        40.0,
        epsabs=1e-12,
        epsrel=1e-12,
        limit=500,
    )[0]

    inverse_size = 1.0 / sizes.astype(float)
    order = np.argsort(inverse_size)

    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    ax.plot(
        inverse_size[order],
        np.asarray(energy_density)[order],
        "o-",
        label="ED",
    )
    ax.axhline(e_infinite, linestyle="--", label="Lieb-Wu")
    ax.set_xlim(0.0, 0.27)
    ax.set_xlabel(r"$1/L$")
    ax.set_ylabel(r"$E_0/(Lt)$")
    ax.legend(frameon=True)
    save(fig, "example_hubbard_chain_finite_size_energy.svg")


def validation_parallel_scaling():
    """Reference thread scaling of the fermion matrix-free executor."""
    data = json.loads(
        (DATA / "fermion_parallel_reference.json").read_text(encoding="utf-8")
    )
    threads = np.asarray([1, 2, 4, 8], dtype=int)

    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    for case in data["cases"]:
        speedups = case["geometric_mean_speedup_vs_numba_serial"]
        values = [speedups[str(thread)] for thread in threads]
        ax.plot(
            threads,
            values,
            "o-",
            label=fr"$D={case['dimension']:,}$",
        )
    ax.axhline(1.0, linewidth=0.8, linestyle="--")
    ax.set_xticks(threads)
    ax.set_xlabel("threads")
    ax.set_ylabel("speedup")
    ax.legend(frameon=True)
    save(fig, "validation_fermion_parallel_scaling.svg")

def hubbard_matrix_sparsity():
    """Plot the sparse structure of a four-site half-filled Hubbard matrix."""
    model = spinful_chain(4, U=4.0, periodic=False)
    matrix = model["hamiltonian"].matrix

    fig, ax = plt.subplots(figsize=(5.2, 5.0), constrained_layout=True)
    ax.spy(matrix, markersize=2.4)
    ax.set_xlabel("basis column")
    ax.set_ylabel("basis row")
    save(fig, "hubbard_matrix_sparsity.svg")


def main():
    """Generate every documentation figure."""
    obsolete = (
        "hubbard_chain_spectrum_observables.svg",
        "hubbard_chain_correlations.svg",
        "hubbard_chain_structure_factors.svg",
        "extended_hubbard_competition.svg",
        "extended_hubbard_structure_profiles.svg",
        "spin_exchange_spectrum.svg",
        "flux_threaded_ring.svg",
        "bose_hubbard_dimer.svg",
        "bose_hubbard_chain.svg",
        "hubbard_dimer_observables.svg",
        "boson_spin_exchange.svg",
        "example_hubbard_chain_pair_structure.svg",
    )
    for name in obsolete:
        (OUT / name).unlink(missing_ok=True)
    hubbard_dimer_observables()
    hubbard_dimer_spectrum()
    hubbard_dimer_wavefunction()
    hubbard_chain_spectrum_observables()
    hubbard_chain_correlations()
    hubbard_chain_structure_factors()
    extended_hubbard_competition()
    extended_hubbard_structure_profiles()
    spin_exchange_spectrum()
    flux_threaded_ring()
    bose_hubbard_dimer()
    bose_hubbard_chain()
    boson_spin_exchange()
    hilbert_space_growth()
    hubbard_matrix_sparsity()
    guide_fermion_ring_current()
    landing_hubbard_ground_state()
    guide_hubbard_low_energy()
    guide_hubbard_double_occupancy()
    guide_hubbard_correlations()
    guide_hubbard_structure_factors()
    guide_bose_hubbard_observables()
    hubbard_finite_size_energy()
    validation_parallel_scaling()


if __name__ == "__main__":
    main()
