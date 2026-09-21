# Fermionic signs and mode ordering

The creation operator acting on an occupation-number state obeys

$$
c_p^\dagger|n_0,\ldots,n_p,\ldots\rangle
=(1-n_p)(-1)^{\sum_{q<p}n_q}
|n_0,\ldots,1_p,\ldots\rangle,
$$

and annihilation obeys

$$
c_p|n_0,\ldots,n_p,\ldots\rangle
=n_p(-1)^{\sum_{q<p}n_q}
|n_0,\ldots,0_p,\ldots\rangle.
$$

The sign depends on a canonical ordering of the fermionic modes. `FermionModes` supplies this ordering and is carried explicitly by primitive operators. Mode ownership is therefore part of the operator definition and is not stored as global mutable state.

In the bit-string representation, the parity exponent is obtained from the population of occupied bits below mode $p$. Optimized kernels reduce this parity with bit operations. The resulting sign is mathematically identical to the explicit occupation sum above.
