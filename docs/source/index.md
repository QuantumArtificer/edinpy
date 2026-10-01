# EDinPy

Exact diagonalization for interacting fermionic and bosonic quantum systems

EDinPy is a Python library for exact diagonalization of finite interacting many-body systems in fermionic and bosonic Fock spaces. Hamiltonians and observables are written directly in second quantization, with support for arbitrary products and sums of creation, annihilation, number, spin, and user-constructed operators.

The same interface covers fixed-particle-number and symmetry-constrained sectors, many-body spectra and eigenstates, expectation values, correlation functions, structure factors, one-body density matrices, natural occupations, and transition matrix elements. Standard model terms are available through compact helpers, while the literal operator algebra remains available whenever a Hamiltonian or observable does not fit a predefined form.

```{figure} _static/figures/landing_hubbard_ground_state.svg
:width: 76%
:alt: Analytic and exact-diagonalization results for the half-filled Hubbard dimer

Half-filled Hubbard dimer. Solid curves show the analytic solution and open circles show EDinPy output for the ground-state energy $E_0/t$, total double occupancy $\langle D\rangle=\sum_i\langle n_{i\uparrow}n_{i\downarrow}\rangle$, and site-averaged local moment $\mu^2=\frac{1}{2}\sum_i\langle(n_{i\uparrow}-n_{i\downarrow})^2\rangle$ as functions of $U/t$.
```

## Second-quantized models

For the half-filled Hubbard dimer,

$$
\begin{aligned}
H ={}& -t \sum_{\sigma}
\left(
    c_{0\sigma}^{\dagger}c_{1\sigma}
    + c_{1\sigma}^{\dagger}c_{0\sigma}
\right) \\
&+ U\left(
    n_{0\uparrow}n_{0\downarrow}
    + n_{1\uparrow}n_{1\downarrow}
\right).
\end{aligned}
$$

The same Hamiltonian can be written directly with the operator algebra:

```python
H = (
     -t * sum(
              cd(0, sigma) * c(1, sigma)
              + cd(1, sigma) * c(0, sigma)
              for sigma in (UP, DOWN)
              )
     + U * (
            n(0, UP) * n(0, DOWN)
            + n(1, UP) * n(1, DOWN)
            )
     )
```

Common model terms also have compact equivalents such as `Hopping(...) + Hubbard(...)`.

::::{grid} 2
:::{grid-item-card} Getting started
:link: getting_started
:link-type: doc
A complete Hubbard-dimer calculation from Hamiltonian construction to observables.
:::
:::{grid-item-card} User guide
:link: user_guide/index
:link-type: doc
Fock spaces, symmetry sectors, operator algebra, diagonalization, observables, and numerical practice.
:::
:::{grid-item-card} API reference
:link: reference/index
:link-type: doc
Core, fermionic, and bosonic public interfaces.
:::
:::{grid-item-card} Examples
:link: examples/index
:link-type: doc
Worked fermionic and bosonic many-body calculations.
:::
::::

{doc}`Theory and methods <theory/index>` develops the mathematical and numerical foundations. {doc}`Validation <validation/index>` collects analytic checks, finite-size tests, and benchmark evidence. Contributor material is under {doc}`Development <development/index>`, and the bibliography is collected in {doc}`References <references>`.

## Scope of exact diagonalization

Exact diagonalization is primarily a finite-system method. The growth of the many-body Hilbert space limits the system sizes that can be treated directly, even when conserved symmetries and sparse representations reduce the numerical cost.

Its main uses include finite-cluster calculations, exploratory studies, controlled reference spectra and observables, and quantitative benchmarks for approximate methods. Exact results are particularly useful for testing analytical approximations, comparing the physical accuracy of cluster and many-body methods, and validating approximations introduced by techniques designed for larger systems. Finite-size sequences can additionally be used to assess how cluster results evolve with system size.[^lin]

## Installation

Install from a source checkout:

```bash
git clone https://github.com/QuantumArtificer/edinpy.git
cd edinpy
python -m pip install -e .
```

Citation information is available in `CITATION.cff` and in {doc}`References <references>`.

```{toctree}
:maxdepth: 2
:hidden:

getting_started
user_guide/index
reference/index
examples/index
theory/index
validation/index
development/index
references
```

[^lin]: H. Q. Lin, J. E. Gubernatis, H. Gould, and J. Tobochnik, "Exact Diagonalization Methods for Quantum Systems," Computers in Physics 7, 400-407 (1993), [doi:10.1063/1.4823192](https://doi.org/10.1063/1.4823192).
