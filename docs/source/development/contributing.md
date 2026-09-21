# Contributing

Changes should preserve three separations in the design:

1. physical labels and sectors are explicit objects
2. the public Hamiltonian is literal second-quantized algebra
3. performance-specific bit masks and sparse-emission structures remain backend implementation details

New public functions and methods should use NumPy-style docstrings with parameters, returns, raises, notes, examples, and references where those sections are relevant. Published algorithms or externally derived implementation ideas must also be recorded in `PROVENANCE.md`.
