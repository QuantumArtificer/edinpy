# Correctness

Correctness is checked at several independent levels: primitive operator algebra, sector dimensions, Hamiltonian matrix elements, analytic few-body spectra, sparse versus matrix-free action, and eigensolver residuals.

## Operator algebra and sectors

Small occupation spaces are compared directly with the canonical fermionic and bosonic relations,

$$
\{c_p,c_q^\dagger\}=\delta_{pq},
\qquad
[b_p,b_q^\dagger]=\delta_{pq}.
$$

Fixed-particle-number dimensions are checked against

$$
D_F(M,N)=\binom{M}{N},
\qquad
D_B(M,N)=\binom{M+N-1}{N},
$$

and symmetry-constrained sectors are checked against the corresponding products of combinatorial factors. These tests establish the basis and primitive operator action before any model-specific Hamiltonian is constructed.

## Analytic Hubbard dimer

For the symmetric half-filled Hubbard dimer,

$$
E_0=\frac{U-\sqrt{U^2+16t^2}}{2},
$$

and the total double occupancy is

$$
\langle D\rangle
=\frac12\left(1-\frac{U}{\sqrt{U^2+16t^2}}\right).
$$

The numerical values coincide with the analytic curve across the plotted interaction range.

```{figure} ../_static/figures/example_hubbard_dimer_double_occupancy.svg
:width: 74%
:alt: Analytic and exact-diagonalization double occupancy of the Hubbard dimer

Total Hubbard-dimer double occupancy. The analytic curve and values obtained from the finite Fock-space Hamiltonian coincide across the plotted interaction range.
```

The singlet-triplet gap provides a second check involving the low-energy spectrum,

$$
\Delta_{ST}
=\frac{\sqrt{U^2+16t^2}-U}{2}
\xrightarrow[U/t\gg1]{}\frac{4t^2}{U}.
$$

```{figure} ../_static/figures/example_hubbard_dimer_gap.svg
:width: 74%
:alt: Analytic and exact-diagonalization singlet-triplet gap of the Hubbard dimer

Hubbard-dimer singlet-triplet gap. The finite-matrix eigensolution follows the closed-form result and approaches the strong-coupling superexchange scale.
```

The complete derivation and operator definitions are given in {doc}`../examples/fermion/hubbard_dimer`.

## Constrained Schwinger-boson spin space

For one Schwinger boson on each of two sites, the constrained bosonic Fock space is four dimensional and represents two spin-$1/2$ degrees of freedom. The Heisenberg Hamiltonian

$$
H=J\mathbf S_0\cdot\mathbf S_1
$$

has

$$
E_{S=0}=-\frac{3J}{4},
\qquad
E_{S=1}=\frac{J}{4}.
$$

```{figure} ../_static/figures/example_boson_exchange_spectrum.svg
:width: 74%
:alt: Analytic singlet and triplet energies represented with constrained Schwinger bosons

Constrained Schwinger-boson Heisenberg dimer. The four-state bosonic sector reproduces the analytic singlet and triplet branches.
```

An independently constructed total-spin operator gives $\langle S_{\mathrm{tot}}^2\rangle=0$ for the singlet and $2$ for each triplet.

```{figure} ../_static/figures/example_boson_exchange_total_spin.svg
:width: 70%
:alt: Total spin squared for the Schwinger-boson Heisenberg-dimer eigenstates

Total-spin expectation values at $J=1$. The result identifies one singlet and the three states of the triplet manifold.
```

The construction is shown in {doc}`../examples/boson/spin_exchange`.

## Independent Hamiltonian actions

For a test vector $x$, explicit sparse construction and matrix-free application are compared through

$$
\epsilon_\infty
=\frac{\|H_{\mathrm{CSC}}x-H_{\mathrm{mf}}x\|_\infty}
{\max(1,\|H_{\mathrm{CSC}}x\|_\infty)}.
$$

Real and complex operator families are tested in both statistics namespaces, including symmetry-constrained sectors. Compiled execution paths are compared against the NumPy implementation using the same represented Hamiltonian.

## Eigensolver residuals

Each returned Ritz pair can be checked with

$$
r_n=\|Hv_n-E_nv_n\|_2.
$$

Analytic agreement and small residuals test different failure modes: the former checks the model and operator algebra against an external result, while the latter checks convergence of the numerical eigenproblem actually solved.
