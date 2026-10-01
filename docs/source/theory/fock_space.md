# Fock space and symmetry sectors

Let $p=0,\ldots,M-1$ label an ordered set of one-particle modes. A Fock state is specified by its occupations,

$$
|\mathbf n\rangle=|n_0,n_1,\ldots,n_{M-1}\rangle,
$$

and a many-body state in a finite sector is

$$
|\psi\rangle=\sum_{\alpha=1}^{D} A_\alpha|\alpha\rangle,
\qquad
\sum_\alpha |A_\alpha|^2=1.
$$

The basis states $|\alpha\rangle$ are occupation configurations consistent with the particle statistics and the conserved quantum numbers chosen for the calculation.

## Fermions

For fermions,

$$
n_p\in\{0,1\},
$$

and the canonical anticommutation relations are

$$
\{c_p,c_q^\dagger\}=\delta_{pq},
\qquad
\{c_p,c_q\}=\{c_p^\dagger,c_q^\dagger\}=0.
$$

The total number operator is

$$
\hat N=\sum_{p=0}^{M-1} n_p,
\qquad n_p=c_p^\dagger c_p.
$$

A fixed-$N$ sector contains

$$
D_F(M,N)=\binom{M}{N}
$$

basis states.

## Bosons

For bosons,

$$
n_p=0,1,2,\ldots,
$$

with

$$
[b_p,b_q^\dagger]=\delta_{pq},
\qquad
[b_p,b_q]=[b_p^\dagger,b_q^\dagger]=0.
$$

At fixed total particle number,

$$
\sum_{p=0}^{M-1}n_p=N,
$$

so the sector dimension is the number of weak compositions of $N$ into $M$ parts,

$$
D_B(M,N)=\binom{N+M-1}{N}.
$$

A fixed-$N$ bosonic sector is finite without imposing an additional local occupation cutoff.

## Conserved quantum numbers

If a Hermitian operator $Q$ commutes with the Hamiltonian,

$$
[H,Q]=0,
$$

then $H$ can be block diagonalized in eigenspaces of $Q$. Particle number is the most common example. If several commuting number operators are conserved, the basis may be restricted to their simultaneous eigenspace.

For a spin-conserving fermion model,

$$
\hat N_\uparrow=\sum_i n_{i\uparrow},
\qquad
\hat N_\downarrow=\sum_i n_{i\downarrow},
$$

and a sector with fixed $(N_\uparrow,N_\downarrow)$ has

$$
D=\binom{L}{N_\uparrow}\binom{L}{N_\downarrow}
$$

states on $L$ sites. Analogous constraints apply to separately conserved layers, orbitals, species, or other labeled subspaces. These are symmetry-constrained sectors: the constraint is valid only when its number operator commutes with the Hamiltonian.

## Matrix elements

Once a basis has been chosen, the Hamiltonian is the finite matrix

$$
H_{\alpha\beta}=\langle\alpha|H|\beta\rangle.
$$

The exact-diagonalization problem is therefore exact for the represented finite Hamiltonian up to floating-point error and eigensolver tolerance. Its practical limitation is combinatorial growth of $D$, even after symmetry reduction. The scaling of representative fermionic and bosonic sectors is shown in {doc}`../user_guide/modes_and_sectors`.
