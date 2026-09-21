# Validation and benchmarks

Correctness and performance are tested separately. A fast matrix builder is useful only if it reproduces the intended fermionic algebra, while a correct small-system calculation does not by itself establish practical scaling.

The fermionic validation suite covers canonical anticommutation relations, fixed-$N$ basis construction, symbolic operator action, sparse matrix construction, eigensolvers, basis-backed eigenstates, observables, and analytic small-system results.

The bosonic module does not yet have the same validation coverage. Fermionic test results should not be assumed to apply to the older bosonic implementation.

```{toctree}
:maxdepth: 1

fermion_correctness
fermion_performance
```
