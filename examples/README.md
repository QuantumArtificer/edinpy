# Examples

The scripts in this directory are complete, executable EDinPy calculations. They use only the public package interface and are kept small enough to inspect alongside the corresponding Hamiltonian.

Run an example from the repository root after installing EDinPy:

```bash
python examples/hubbard_dimer.py
```

The examples cover:

- [`hubbard_dimer.py`](hubbard_dimer.py): the half-filled two-site Hubbard model, including analytic checks of the ground-state energy and observables
- [`hubbard_chain.py`](hubbard_chain.py): a spinful Hubbard chain with local moments and charge, spin, and pairing diagnostics
- [`extended_hubbard_chain.py`](extended_hubbard_chain.py): nearest-neighbor density interactions and competing charge and spin correlations
- [`flux_threaded_ring.py`](flux_threaded_ring.py): complex hopping amplitudes, Hermiticity, flux dependence, and persistent current
- [`spin_exchange.py`](spin_exchange.py): the fermionic representation of two-spin Heisenberg exchange

The documentation develops the same calculations in more detail in the [worked examples](https://quantumartificer.github.io/edinpy/fermion/examples/index.html).
