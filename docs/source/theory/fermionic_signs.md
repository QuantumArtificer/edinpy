# Fermionic signs

Fermionic signs follow from the canonical anticommutation relations and the chosen ordering of the one-particle modes. For

$$
|\mathbf n\rangle=|n_0,n_1,\ldots,n_{M-1}\rangle,
$$

define the number of occupied modes preceding $p$ as

$$
\eta_p=\sum_{q<p} n_q.
$$

Annihilation acts as

$$
c_p|\mathbf n\rangle
=(-1)^{\eta_p} n_p
|n_0,\ldots,n_p-1,\ldots,n_{M-1}\rangle,
$$

while creation gives

$$
c_p^\dagger|\mathbf n\rangle
=(-1)^{\eta_p}(1-n_p)
|n_0,\ldots,n_p+1,\ldots,n_{M-1}\rangle.
$$

The parity factor counts how many fermionic operators must be anticommuted past occupied modes before acting on the target mode. It depends on the ordering convention, while physical matrix elements remain consistent when the same ordering is used for basis states and operators.

For a hopping bilinear $c_p^\dagger c_q$, the sign is obtained by applying the annihilation operator first and the creation operator second. This automatically accounts for the occupation parity between the two modes; no separate sign convention is attached to a hopping term.

With several discrete labels, such as site and spin, the flattened mode order fixes the parity count. Changing the order of the degrees of freedom changes intermediate occupation strings and signs, so a single mode convention must be used throughout one calculation.

Bosonic creation and annihilation operators carry occupation amplitudes $\sqrt{n_p+1}$ and $\sqrt{n_p}$ instead of fermionic parity factors.
