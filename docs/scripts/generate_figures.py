"""Generate all scientific figures used by the EDinPy documentation.

Run from the repository root with::

    PYTHONPATH=src python docs/scripts/generate_figures.py

Every numerical data point is produced with the public EDinPy API. Analytical
curves are used only where the corresponding closed-form result is stated in
the documentation.
"""

from __future__ import annotations

from math import comb
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from edinpy import fermion as edf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "source" / "_static" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Make committed SVG assets reproducible across repeated local builds.
mpl.rcParams["svg.hashsalt"] = "edinpy-docs"

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
    sector = edf.NParticleSector(modes, N=N)
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
    """Analytic and ED observables for the symmetric half-filled dimer."""
    U_values = np.linspace(0.0, 12.0, 49)
    e_ed = []
    d_ed = []
    moment_ed = []
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
        local_moment = 0.5 * (
            (n(0, UP) - n(0, DOWN)) * (n(0, UP) - n(0, DOWN))
            + (n(1, UP) - n(1, DOWN)) * (n(1, UP) - n(1, DOWN))
        )
        Sdot = edf.HeisenbergExchange(
            (0, UP), (0, DOWN), (1, UP), (1, DOWN), 1.0, modes
        )

        e_ed.append(energies[0])
        d_ed.append(np.real(psi0.dag * D * psi0))
        moment_ed.append(np.real(psi0.dag * local_moment * psi0))
        spin_ed.append(np.real(psi0.dag * Sdot * psi0))
        gap_ed.append(energies[1] - energies[0])

    U_exact = np.linspace(0.0, 12.0, 500)
    root = np.sqrt(U_exact**2 + 16.0)
    e_exact = 0.5 * (U_exact - root)
    d_exact = 0.5 * (1.0 - U_exact / root)
    moment_exact = 1.0 - d_exact
    spin_exact = -0.75 * moment_exact
    gap_exact = 0.5 * (root - U_exact)

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6), constrained_layout=True)

    axes[0, 0].plot(U_exact, e_exact, label="analytic")
    axes[0, 0].plot(U_values, e_ed, "o", markersize=3.2, label="EDinPy")
    axes[0, 0].set_xlabel(r"$U/t$")
    axes[0, 0].set_ylabel(r"$E_0/t$")
    axes[0, 0].set_title("Ground-state energy")
    axes[0, 0].legend()

    axes[0, 1].plot(U_exact, d_exact, label=r"double occupancy $\langle D\rangle$")
    axes[0, 1].plot(U_exact, moment_exact, label=r"local moment $\mu^2$")
    axes[0, 1].plot(U_values, d_ed, "o", markersize=3.0)
    axes[0, 1].plot(U_values, moment_ed, "o", markersize=3.0)
    axes[0, 1].set_xlabel(r"$U/t$")
    axes[0, 1].set_ylabel("dimensionless expectation value")
    axes[0, 1].set_title("Charge suppression and moment formation")
    axes[0, 1].legend()

    axes[1, 0].plot(U_exact, spin_exact, label="analytic")
    axes[1, 0].plot(U_values, spin_ed, "o", markersize=3.2, label="EDinPy")
    axes[1, 0].axhline(-0.75, linestyle="--", linewidth=1.0, label="pure singlet")
    axes[1, 0].set_xlabel(r"$U/t$")
    axes[1, 0].set_ylabel(r"$\langle\mathbf{S}_0\cdot\mathbf{S}_1\rangle$")
    axes[1, 0].set_title("Antiferromagnetic correlation")
    axes[1, 0].legend()

    mask = U_exact >= 2.0
    axes[1, 1].plot(U_exact, gap_exact, label="exact gap")
    axes[1, 1].plot(U_exact[mask], 4.0 / U_exact[mask], "--", label=r"$4t^2/U$")
    axes[1, 1].plot(U_values[1:], np.asarray(gap_ed)[1:], "o", markersize=3.0, label="EDinPy")
    axes[1, 1].set_xlabel(r"$U/t$")
    axes[1, 1].set_ylabel(r"$\Delta_{ST}/t$")
    axes[1, 1].set_ylim(0.0, 2.1)
    axes[1, 1].set_title("Singlet-triplet gap")
    axes[1, 1].legend()

    save(fig, "hubbard_dimer_observables.svg")


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
    ax.set_title(r"Hubbard-dimer ground state at $U/t=4$")
    ax.tick_params(axis="x", rotation=25)
    save(fig, "hubbard_dimer_wavefunction.svg")


def hubbard_chain_spectrum_observables():
    """Spectrum and several ground-state observables for a six-site ring."""
    L = 6
    U_values = np.linspace(0.0, 8.0, 33)
    levels = []
    double_occupancy = []
    local_moment = []
    spin_pi = []
    charge_pi = []
    pair_zero = []

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
        spin_pi.append(structure_factor(psi0, spin_operators(model), np.pi))
        charge_pi.append(structure_factor(psi0, charge_operators(model), np.pi))
        pair_zero.append(structure_factor(psi0, pair_operators(model), 0.0))

    levels = np.asarray(levels)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), constrained_layout=True)

    for level in range(levels.shape[1]):
        axes[0, 0].plot(U_values, levels[:, level], linewidth=1.1)
    axes[0, 0].set_xlabel(r"$U/t$")
    axes[0, 0].set_ylabel(r"$(E_n-E_0)/t$")
    axes[0, 0].set_title("Lowest eight many-body levels")

    axes[0, 1].plot(
        U_values,
        double_occupancy,
        label=r"$\langle n_{i\uparrow}n_{i\downarrow}\rangle$",
    )
    axes[0, 1].plot(
        U_values,
        local_moment,
        label=r"$\mu^2=\langle(n_{i\uparrow}-n_{i\downarrow})^2\rangle$",
    )
    axes[0, 1].set_xlabel(r"$U/t$")
    axes[0, 1].set_ylabel("site average")
    axes[0, 1].set_title("Local charge and spin diagnostics")
    axes[0, 1].legend()

    axes[1, 0].plot(U_values, spin_pi, label=r"$S_s(\pi)$")
    axes[1, 0].plot(U_values, charge_pi, label=r"$S_c(\pi)$")
    axes[1, 0].set_xlabel(r"$U/t$")
    axes[1, 0].set_ylabel("structure factor")
    axes[1, 0].set_title(r"Spin and charge response at $q=\pi$")
    axes[1, 0].legend()

    axes[1, 1].plot(U_values, pair_zero)
    axes[1, 1].set_xlabel(r"$U/t$")
    axes[1, 1].set_ylabel(r"$P(q=0)$")
    axes[1, 1].set_title("On-site pair structure factor")

    save(fig, "hubbard_chain_spectrum_observables.svg")


def hubbard_chain_correlations():
    """Real-space spin and connected charge correlations of a six-site ring."""
    L = 6
    distances = np.arange(L // 2 + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)

    for U in (0.0, 4.0, 8.0):
        model = spinful_chain(L, U=U, periodic=True)
        ham = model["hamiltonian"]
        ham.eigsolve(k=1, which="SA", v0=np.ones(ham.sector.dimension))
        psi0 = ham.eigenstate(0)
        spins = spin_operators(model)
        charges = charge_operators(model)
        spin_corr = [translational_correlation(psi0, spins, int(r)) for r in distances]
        charge_corr = [translational_correlation(psi0, charges, int(r)) for r in distances]
        axes[0].plot(distances, spin_corr, "o-", label=fr"$U/t={U:g}$")
        axes[1].plot(distances, charge_corr, "o-", label=fr"$U/t={U:g}$")

    axes[0].axhline(0.0, linewidth=0.8)
    axes[0].set_xticks(distances)
    axes[0].set_xlabel("separation $r$")
    axes[0].set_ylabel(r"$C_s(r)$")
    axes[0].set_title(r"$\langle S_i^z S_{i+r}^z\rangle$")
    axes[0].legend()

    axes[1].axhline(0.0, linewidth=0.8)
    axes[1].set_xticks(distances)
    axes[1].set_xlabel("separation $r$")
    axes[1].set_ylabel(r"$C_c(r)$")
    axes[1].set_title(r"$\langle\delta n_i\,\delta n_{i+r}\rangle$")
    axes[1].legend()

    save(fig, "hubbard_chain_correlations.svg")


def hubbard_chain_structure_factors():
    """Momentum-resolved spin, charge, and pair structure factors."""
    L = 6
    q_values = 2.0 * np.pi * np.arange(L) / L
    q_over_pi = q_values / np.pi
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True)

    for U in (0.0, 4.0, 8.0):
        model = spinful_chain(L, U=U, periodic=True)
        ham = model["hamiltonian"]
        ham.eigsolve(k=1, which="SA", v0=np.ones(ham.sector.dimension))
        psi0 = ham.eigenstate(0)
        spins = spin_operators(model)
        charges = charge_operators(model)
        pairs = pair_operators(model)
        s_spin = [structure_factor(psi0, spins, q) for q in q_values]
        s_charge = [structure_factor(psi0, charges, q) for q in q_values]
        s_pair = [structure_factor(psi0, pairs, q) for q in q_values]
        label = fr"$U/t={U:g}$"
        axes[0].plot(q_over_pi, s_spin, "o-", label=label)
        axes[1].plot(q_over_pi, s_charge, "o-", label=label)
        axes[2].plot(q_over_pi, s_pair, "o-", label=label)

    titles = ("Spin structure factor", "Charge structure factor", "Pair structure factor")
    ylabels = (r"$S_s(q)$", r"$S_c(q)$", r"$P(q)$")
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_xlabel(r"$q/\pi$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(q_over_pi)
        ax.legend()

    save(fig, "hubbard_chain_structure_factors.svg")


def extended_hubbard_competition():
    """Spin, charge, and bond diagnostics across the extended-Hubbard crossover."""
    L = 6
    U = 4.0
    V_values = np.linspace(0.0, 4.0, 33)
    spin_pi = []
    charge_pi = []
    bond_pi = []
    double_occupancy = []
    local_moment = []

    for V in V_values:
        model = spinful_chain(L, U=U, V=V, periodic=True)
        ham = model["hamiltonian"]
        ham.eigsolve(k=1, which="SA", v0=np.ones(ham.sector.dimension))
        psi0 = ham.eigenstate(0)
        n = model["n"]

        spin_pi.append(structure_factor(psi0, spin_operators(model), np.pi))
        charge_pi.append(structure_factor(psi0, charge_operators(model), np.pi))
        bond_pi.append(structure_factor(psi0, bond_operators(model), np.pi))
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

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), constrained_layout=True)
    axes[0].plot(V_values, spin_pi, label=r"spin $S_s(\pi)$")
    axes[0].plot(V_values, charge_pi, label=r"charge $S_c(\pi)$")
    axes[0].plot(V_values, bond_pi, label=r"bond $S_B(\pi)$")
    axes[0].axvline(U / 2.0, linestyle="--", linewidth=1.0, label=r"$V=U/2$")
    axes[0].set_xlabel(r"$V/t$ at $U/t=4$")
    axes[0].set_ylabel("structure factor")
    axes[0].set_title("Competing staggered correlations")
    axes[0].legend()

    axes[1].plot(V_values, double_occupancy, label="double occupancy")
    axes[1].plot(V_values, local_moment, label="local moment")
    axes[1].axvline(U / 2.0, linestyle="--", linewidth=1.0)
    axes[1].set_xlabel(r"$V/t$ at $U/t=4$")
    axes[1].set_ylabel("site average")
    axes[1].set_title("Local reorganization of the ground state")
    axes[1].legend()

    save(fig, "extended_hubbard_competition.svg")


def extended_hubbard_structure_profiles():
    """Momentum profiles on the SDW-like, crossover, and CDW-like sides."""
    L = 6
    U = 4.0
    q_values = 2.0 * np.pi * np.arange(L) / L
    q_over_pi = q_values / np.pi
    V_samples = (0.5, 2.25, 3.5)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True)

    for V in V_samples:
        model = spinful_chain(L, U=U, V=V, periodic=True)
        ham = model["hamiltonian"]
        ham.eigsolve(k=1, which="SA", v0=np.ones(ham.sector.dimension))
        psi0 = ham.eigenstate(0)
        spin_values = [structure_factor(psi0, spin_operators(model), q) for q in q_values]
        charge_values = [structure_factor(psi0, charge_operators(model), q) for q in q_values]
        bonds = bond_operators(model)
        bond_values = [connected_structure_factor(psi0, bonds, q) for q in q_values]
        label = fr"$V/t={V:g}$"
        axes[0].plot(q_over_pi, spin_values, "o-", label=label)
        axes[1].plot(q_over_pi, charge_values, "o-", label=label)
        axes[2].plot(q_over_pi, bond_values, "o-", label=label)

    titles = ("Spin", "Charge", "Bond")
    ylabels = (r"$S_s(q)$", r"$S_c(q)$", r"$S_B(q)$")
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_xlabel(r"$q/\pi$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} structure factor")
        ax.set_xticks(q_over_pi)
        ax.legend()

    save(fig, "extended_hubbard_structure_profiles.svg")


def spin_exchange_spectrum():
    """Two-site Heisenberg exchange spectrum and ground-state correlation."""
    site = edf.DoF(2, name="site")
    spin = edf.DoF(2, name="spin")
    modes = edf.FermionModes(site, spin)
    sector = edf.NParticleSector(modes, N=2)
    Sdot = edf.HeisenbergExchange(
        (0, UP), (0, DOWN), (1, UP), (1, DOWN), 1.0, modes
    )

    J_values = np.linspace(-2.0, 2.0, 81)
    spectra = []
    correlations = []
    for J in J_values:
        H = J * Sdot
        ham = edf.Hamiltonian(H, sector)
        energies, _ = ham.eigsolve(k=None)
        spectra.append(energies)
        if abs(J) < 1e-12:
            correlations.append(np.nan)
        else:
            psi0 = ham.eigenstate(0)
            correlations.append(np.real(psi0.dag * Sdot * psi0))

    spectra = np.asarray(spectra)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
    for level in range(spectra.shape[1]):
        axes[0].plot(J_values, spectra[:, level], linewidth=1.0)
    axes[0].plot(J_values, -0.75 * J_values, "--", label="singlet $-3J/4$")
    axes[0].plot(J_values, 0.25 * J_values, "--", label="triplet $J/4$")
    axes[0].axhline(0.0, linestyle=":", linewidth=1.0, label="doublon states")
    axes[0].set_xlabel(r"$J$")
    axes[0].set_ylabel("energy")
    axes[0].set_title("Fermionic two-site exchange spectrum")
    axes[0].legend()

    axes[1].plot(J_values, correlations)
    axes[1].axhline(-0.75, linestyle="--", linewidth=1.0)
    axes[1].axhline(0.25, linestyle="--", linewidth=1.0)
    axes[1].set_xlabel(r"$J$")
    axes[1].set_ylabel(r"$\langle\mathbf{S}_0\cdot\mathbf{S}_1\rangle$")
    axes[1].set_title("Ground-state spin character")

    save(fig, "spin_exchange_spectrum.svg")


def flux_threaded_ring():
    """Level flow, many-body energy, and persistent current of a spinless ring."""
    L = 4
    N = 2
    t = 1.0
    phi_values = np.linspace(-2.0 * np.pi, 2.0 * np.pi, 161)
    k_values = 2.0 * np.pi * np.arange(L) / L

    site = edf.DoF(L, name="site")
    modes = edf.FermionModes(site)
    sector = edf.NParticleSector(modes, N=N)
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
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)

    for m in range(L):
        axes[0].plot(x, single_particle[:, m])
    axes[0].set_xlabel(r"$\phi/\pi$")
    axes[0].set_ylabel(r"$\varepsilon_m/t$")
    axes[0].set_title("Single-particle level flow")

    axes[1].plot(x, e_exact, label="free-fermion formula")
    axes[1].plot(x[::8], np.asarray(e_ed)[::8], "o", markersize=3.2, label="EDinPy")
    axes[1].set_xlabel(r"$\phi/\pi$")
    axes[1].set_ylabel(r"$E_0/t$")
    axes[1].set_title("Many-body ground-state energy")
    axes[1].legend()

    axes[2].plot(x, i_exact, label="analytic branch")
    axes[2].plot(x[::8], np.asarray(i_ed)[::8], "o", markersize=3.2, label="EDinPy")
    axes[2].axhline(0.0, linewidth=0.8)
    axes[2].set_xlabel(r"$\phi/\pi$")
    axes[2].set_ylabel(r"$I/t$")
    axes[2].set_title(r"$I=-\partial E_0/\partial\phi$")
    axes[2].legend()

    save(fig, "flux_threaded_ring.svg")


def hilbert_space_growth():
    """Show full and fixed-N Hilbert-space growth for a spinful chain."""
    L_values = np.arange(1, 17)
    full = np.asarray([4**L for L in L_values], dtype=float)
    half_filled = np.asarray([comb(2 * L, L) for L in L_values], dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 4.1), constrained_layout=True)
    ax.semilogy(L_values, full, "o-", label=r"full Fock space $4^L$")
    ax.semilogy(L_values, half_filled, "o-", label=r"fixed $N=L$: $\binom{2L}{L}$")
    ax.set_xlabel("spinful sites $L$")
    ax.set_ylabel("Hilbert-space dimension")
    ax.set_title("Combinatorial growth of a spinful fermion problem")
    ax.legend()
    save(fig, "hilbert_space_growth.svg")


def hubbard_matrix_sparsity():
    """Plot the sparse structure of a four-site half-filled Hubbard matrix."""
    model = spinful_chain(4, U=4.0, periodic=False)
    matrix = model["hamiltonian"].matrix

    fig, ax = plt.subplots(figsize=(5.2, 5.0), constrained_layout=True)
    ax.spy(matrix, markersize=2.4)
    ax.set_xlabel("basis column")
    ax.set_ylabel("basis row")
    ax.set_title(f"Hubbard matrix: D={matrix.shape[0]}, nnz={matrix.nnz}")
    save(fig, "hubbard_matrix_sparsity.svg")


def main():
    """Generate every documentation figure."""
    hubbard_dimer_observables()
    hubbard_dimer_wavefunction()
    hubbard_chain_spectrum_observables()
    hubbard_chain_correlations()
    hubbard_chain_structure_factors()
    extended_hubbard_competition()
    extended_hubbard_structure_profiles()
    spin_exchange_spectrum()
    flux_threaded_ring()
    hilbert_space_growth()
    hubbard_matrix_sparsity()


if __name__ == "__main__":
    main()
