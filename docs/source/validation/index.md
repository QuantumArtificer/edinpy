# Validation and benchmarks

EDinPy separates correctness tests from performance measurements. A fast benchmark is not evidence that an operator algebra is correct, and a passing algebra test does not establish practical scaling.

The current quantitative validation pages cover the fermionic backend. They test canonical anticommutation relations, fixed-$N$ basis construction, symbolic and compiled operator action, sparse matrix construction, eigensolvers, basis-backed eigenstates, observable algebra, and analytic many-body benchmarks.

The bosonic implementation does not yet have a corresponding validation suite. Fermionic validation results are not assumed to apply to the bosonic algebra or Hilbert-space construction.

```{toctree}
:maxdepth: 1

fermion_correctness
fermion_performance
```
