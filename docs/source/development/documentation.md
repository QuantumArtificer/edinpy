# Documentation conventions

EDinPy documentation is scientific documentation for an exact-diagonalization library. The physics, mathematical objects, and research workflow determine the exposition. Source layout and implementation details enter where they help a reader use or interpret the calculation.

## Page roles

The landing page states the scientific scope of the package, shows the literal Fock-algebra style, and directs readers to the main documentation paths.

Getting Started develops one complete physical calculation from Hamiltonian to observables. It should be small enough that numerical results can be checked directly and rich enough to introduce the main workflow.

The User Guide follows the general research workflow: define the one-particle space, choose a finite many-body sector, construct operators and Hamiltonians, diagonalize, recover states, and evaluate observables. The presentation should remain statistics agnostic when the concepts and public interface agree. When a concrete namespace is needed, state which statistics is being used and explain the statistics-specific rule at the point where it enters.

Worked examples begin from a physical problem rather than a software feature. Each example should motivate the model, define the Hamiltonian and observables mathematically, show the implementation, present numerical results, and discuss the physics. Numerical performance can be mentioned briefly when it affects the calculation.

The API reference mirrors the public source organization through Core API, Fermionic API, and Bosonic API pages. Shared concepts are documented once where possible. Statistics-specific objects are documented under their corresponding namespace.

Theory and numerical-method pages contain derivations, sign conventions, numerical representations, eigensolver behavior, and scaling arguments that would interrupt the main research workflow in the User Guide.

## Scientific exposition

Introduce a physical quantity before presenting code that computes it. A useful order is

1. physical motivation or question
2. mathematical definition
3. definition of every symbol and parameter
4. physical interpretation
5. EDinPy implementation
6. numerical result
7. discussion of the result

Every quantity shown in a figure must be defined mathematically in the surrounding text. If a plotted quantity is built from a custom operator or helper function, show the implementation that constructs it.

Standard helper functions should appear naturally where they simplify a physical term. Literal operator construction should remain visible whenever it illustrates how a Hamiltonian, correlation function, density matrix, structure factor, spin operator, or other observable is assembled.

Basis-aware state algebra is a central part of the public interface. Expressions such as

```python
psi.dag * O * psi
```

should be used where they make the connection to Dirac notation clearer. State-operator products, Hermitian conjugation, custom sums and products, and model-specific helper functions should be introduced through calculations rather than feature lists.

## Numerical choices

The User Guide should explain numerical representations and execution choices at the point where they affect a normal calculation. Important options include dense versus sparse eigensolution, matrix materialization, matrix-free action, `execution="numpy"`, `execution="numba-serial"`, and `execution="numba-parallel"`.

The discussion should answer practical questions such as when a sparse representation is sufficient, when matrix-free execution saves memory, when Numba compilation is likely to amortize, and when thread overhead can dominate a small calculation. Keep the explanation tied to basis size, operator structure, requested eigenpairs, and repeated Hamiltonian action.

Worked physics examples should keep these remarks brief. A short phrase about lowered operator execution, JIT compilation, solver time, or memory use is enough when it helps explain why a calculation is feasible.

## Figures

Figures should have the visual standard of a scientific paper.

- Use meaningful physical quantities on the axes, with units or normalized variables where applicable.
- Use mathematical notation for observables when it is clearer than prose.
- Do not place panel labels in figures.
- Avoid explanatory sentences inside the plotting area.
- Keep titles absent or very short when the axes and caption already identify the content.
- Legends must be boxed.
- Legend entries should be short and intuitive. Prefer symbols, parameter values, or compact model labels over sentences and long phrases.
- Use line styles, markers, and axes that allow the plotted quantities to be distinguished without relying on prose embedded in the figure.
- Keep visual density appropriate for the rendered documentation width.

Each figure must have a condensed but informative caption. The caption identifies the system, parameters needed to read the plot, and the main physical quantity or trend. The surrounding text carries the fuller interpretation.

The reader should be able to identify the plotted quantities from the axes, legend, and caption without searching the code.

## Examples and output

Executable scripts live under `examples/fermion/` and `examples/boson/`. Documentation pages should show the code required to understand the calculation rather than reproducing every line of the script.

Numerical output should be included when it supports the scientific discussion, verifies an analytic result, or makes an API operation concrete. Avoid terminal-style output that only demonstrates that a command ran.

Reference problems should cite the literature used for the model, analytic result, or physical interpretation. Finite-size calculations should distinguish finite-system crossovers and level structure from thermodynamic phase transitions.

## Prose

Use direct scientific prose.

- Avoid metanarrative about documentation organization, implementation history, refactors, previous versions, or development conversations.
- Avoid defensive constructions that explain an object primarily by saying what it is not.
- Do not use bold or italic emphasis in explanatory prose.
- Do not use em dashes.
- Prefer physical nouns and verbs over software-oriented abstractions when discussing a model or result.
- Introduce backend, compiler, kernel, dispatch, ranking, and storage terminology only where the numerical method requires it.
- Keep paragraphs compact enough to read alongside equations, code, and figures.

## Validation

Before merging documentation changes:

1. execute every affected example
2. regenerate every affected figure from source
3. compare numerical values with analytic results or independent checks where available
4. verify equations against the implementation
5. build Sphinx with warnings treated as errors
6. inspect rendered pages at normal documentation width
7. inspect every figure for clipped labels, overlapping elements, legend placement, and readable axes
8. run the package test suite

A documentation change is complete when the scientific explanation, executable calculation, generated figures, and public API agree.
