# Operator and state algebra

EDinPy keeps the many-body notation close to the mathematics. Creation, annihilation, and number operators are first-class symbolic objects. Ordinary Python `+`, `-`, and `*` build operator expressions, `.dag` takes the Hermitian adjoint, and the same binary operations act on Fock kets and bras.

## Primitive operators

For one four-mode problem:

```python
from edinpy import fermion as edf

mode = edf.DoF(4, name="mode")
modes = edf.FermionModes(mode)

c = edf.set_notation(edf.Annihilation, modes)
cd = edf.set_notation(edf.Creation, modes)
n = edf.set_notation(edf.Number, modes)

print(c(2))
print(cd(2))
print(n(2))
```

```text
c[2]
c†[2]
n[2]
```

The variable names `c`, `cd`, and `n` are chosen by the script. EDinPy only binds the operator class to the `FermionModes` object.

## Products and sums are literal expressions

```python
hop = cd(0) * c(3)
hermitian_hop = hop + hop.dag
interaction = 2.5 * n(0) * n(3)
H = -1.0 * hermitian_hop + interaction

print(hop)
print(hop.dag)
print(H)
```

```text
c†[0] c[3]
c†[3] c[0]
-1.0 c†[0] c[3] + -1.0 c†[3] c[0] + 2.5 n[0] n[3]
```

Products act from right to left, exactly as they do on paper. A product such as `cd(0) * c(3)` therefore means $c_0^\dagger c_3$.

## Fermionic signs are visible in state action

Consider

$$
|\psi\rangle=|1010\rangle,
$$

where modes 1 and 3 are occupied and mode 0 is the rightmost bit. Acting on the upper occupied mode crosses one occupied lower-index mode and therefore produces a minus sign:

```python
ket = edf.FockState(0b1010, n_modes=4)

print("ket:       ", ket)
print("c[1] ket:  ", c(1) * ket)
print("c[3] ket:  ", c(3) * ket)
print("c†[0] ket: ", cd(0) * ket)
```

```text
ket:        1 |1010>
c[1] ket:   1 |1000>
c[3] ket:   -1 |0010>
c†[0] ket:  1 |1011>
```

The composite action remains literal:

```python
print(cd(0) * c(3) * ket)
```

```text
-1 |0011>
```

This low-level operator-state action is useful when checking a sign convention or deriving a matrix element by hand.

## Canonical anticommutation relations

The primitive action implements

$$
\{c_p,c_q^\dagger\}=\delta_{pq},\qquad
\{c_p,c_q\}=\{c_p^\dagger,c_q^\dagger\}=0.
$$

The sign of an action on mode $p$ is

$$
(-1)^{\sum_{q<p} n_q},
$$

where the order $q<p$ is the `FermionModes` order described in {doc}`modes_and_sectors`.

## Bras, kets, and ordinary binary operations

The same one-dimensional mode set can be used to build a small fixed-$N$ problem:

```python
sector = edf.NParticleSector(modes, N=2).build()
H = n(0) + 2 * n(1) + 3 * n(2) + 4 * n(3)

hamiltonian = edf.Hamiltonian(H, sector)
energies, _ = hamiltonian.eigsolve(k=None)

print(H)
print(energies)
```

```text
n[0] + 2 n[1] + 3 n[2] + 4 n[3]
[3. 4. 5. 5. 6. 7.]
```

Recover two eigenstates as basis-aware kets:

```python
psi = hamiltonian.eigenstate(0)
phi = hamiltonian.eigenstate(1)

print(psi.dag * psi)
print(psi.dag * phi)
```

```text
1.0
0.0
```

An expectation value and a transition matrix element use the same binary operations:

```python
O = n(0)
print(O)
print(psi.dag * O * psi)
print(phi.dag * O * psi)
```

```text
n[0]
1.0
0.0
```

No special observable container is required. The method

```python
print(psi.inner(phi))
```

is the convenience form of `psi.dag * phi` and prints

```text
0.0
```

## Literal state action versus an expectation value

These two expressions ask related but different questions:

```python
removed = c(0) * psi
expectation = psi.dag * c(0) * psi

print(removed)
print(expectation)
```

```text
1.0 |0010>
0.0
```

`c(0) * psi` is the actual one-particle ket produced by annihilating mode 0. It lies outside the original two-particle sector, but the state action remains well defined. The scalar expectation value is zero because the one- and two-particle sectors are orthogonal.

For basis-backed vectors in the same fixed-$N$ sector, `psi.dag * O * psi` uses the compiled sparse operator representation directly. Ordinary Dirac notation therefore also uses the optimized path for expectation values.

## Symbolic front end, compiled back end

The symbolic tree is the public representation of the physics. The compiler recognizes structures such as number products, one-body hopping, and number-conserving monomials and lowers them to optimized sparse execution kernels. The optimized path therefore does not require tables, integer transition maps, or a separate model-specific language.
