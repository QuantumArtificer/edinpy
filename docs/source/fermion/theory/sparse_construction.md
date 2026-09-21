# Symbolic compilation and sparse construction

EDinPy separates the public symbolic algebra from the matrix-construction backend.

A user expression first forms an operator tree. The compiler then flattens sums, separates scalar coefficients from primitive operators, and recognizes algebraic structures that can be evaluated without applying the full expression object state by state.

Recognized terms are lowered to compact intermediate representations containing mode indices and bit masks. Specialized execution kernels handle number products, simple hopping terms, and general number-conserving fermionic monomials. The resulting matrix entries are emitted directly in compressed sparse-column order.

This design keeps the input close to the second-quantized Hamiltonian while allowing the backend to use implementation-specific representations where they are advantageous. The optimized representation is therefore a compilation target, not the user-facing model description.
