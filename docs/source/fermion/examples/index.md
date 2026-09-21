# Guided examples

Each example is written as a short computational-physics note. It starts from the Hamiltonian, defines the physical quantities appearing in it, prints the complete symbolic EDinPy expression, and uses the eigenstates to calculate observables that address a specific physics question.

```{toctree}
:maxdepth: 1

hubbard_dimer
hubbard_chain
extended_hubbard_chain
spin_exchange
flux_threaded_ring
```

## Suggested reading order

Start with {doc}`hubbard_dimer`. The complete spectrum and several ground-state observables are available analytically, so every numerical result can be checked against a closed-form expression. It introduces double occupancy, local-moment formation, spin correlations, the singlet-triplet gap, and the strong-coupling scale $4t^2/U$.

{doc}`hubbard_chain` moves to a six-site periodic system with no closed-form many-body spectrum. It shows how to extract real-space spin and charge correlations, momentum-space structure factors, on-site pairing correlations, and low-energy spectral evolution from the same `FockVector` ground state. The finite-size results are interpreted in the context of the exact one-dimensional Hubbard solution of Lieb and Wu.

{doc}`extended_hubbard_chain` adds a nearest-neighbor repulsion and compares competing spin, charge, and bond correlations. The example constructs spin, charge, and bond structure factors directly from the operator algebra and compares the finite-ring trends with the established SDW-BOW-CDW physics of the half-filled one-dimensional extended Hubbard model.

{doc}`spin_exchange` isolates the two-site Heisenberg exchange interaction. Because the spectrum is known analytically, the example cleanly separates singlet, triplet, and fermionic doublon states and shows how $\mathbf S_{\mathrm{tot}}^2$ can be used as a quantum-number diagnostic.

{doc}`flux_threaded_ring` introduces a Peierls phase and a genuinely complex Hermitian Hamiltonian. Differentiating that Hamiltonian with respect to flux gives a persistent-current operator. EDinPy reproduces the exact noninteracting energy and current while local densities remain uniform.

## A common workflow

All five examples use the same numerical sequence:

1. define the fermionic modes and Hilbert-space sector
2. write and print the Hamiltonian in literal second-quantized form
3. solve only the part of the eigensystem required by the calculation
4. convert the required eigenvector to a basis-aware `FockVector`
5. build observables from the same primitive operators used in the Hamiltonian
6. evaluate expressions such as `psi.dag * O * psi`
7. organize two-point observables into correlations or structure factors when spatial information matters
8. compare with an analytic result, a controlled limit, or an appropriate literature benchmark.

The examples construct observables directly from the literal Fock algebra. Model-specific observable functions are therefore unnecessary for these calculations.
