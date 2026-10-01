# Validation

Numerical validation is organized around independent checks of the algebra, basis construction, Hamiltonian action, and eigensolution. Analytic few-body models test several layers at once, while direct comparisons between sparse and matrix-free actions test numerical representations without relying on a particular physical spectrum.

Performance measurements are kept separate from correctness because absolute timings and memory usage depend on the machine, thread configuration, optional compiled execution, and the physical workload.

```{toctree}
:maxdepth: 2

correctness
performance
```
