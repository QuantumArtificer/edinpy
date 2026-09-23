import pytest

from edinpy import fermion as edf


def test_n_particle_sector_is_unbuilt_on_construction():
    modes = edf.FermionModes(edf.DoF(6))
    sector = edf.NParticleSector(modes, N=3)

    assert not sector.is_built
    with pytest.raises(RuntimeError, match="sector.build"):
        _ = sector.basis
    with pytest.raises(RuntimeError, match="sector.build"):
        _ = sector.dimension


def test_n_particle_sector_builds_its_basis():
    modes = edf.FermionModes(edf.DoF(6))
    sector = edf.NParticleSector(modes, N=3)

    returned = sector.build()

    assert returned is sector
    assert sector.is_built
    assert sector.dimension == 20
    assert sector.basis.n_modes == 6
    assert sector.basis.N == 3


def test_n_particle_sector_build_is_idempotent():
    modes = edf.FermionModes(edf.DoF(6))
    sector = edf.NParticleSector(modes, N=3).build()
    basis = sector.basis

    assert sector.build() is sector
    assert sector.basis is basis


def test_n_particle_sector_from_vector_requires_build():
    modes = edf.FermionModes(edf.DoF(4))
    sector = edf.NParticleSector(modes, N=2)

    with pytest.raises(RuntimeError, match="sector.build"):
        sector.from_vector([1, 0, 0, 0, 0, 0])


def test_independent_sectors_do_not_share_state():
    modes_a = edf.FermionModes(edf.DoF(5))
    modes_b = edf.FermionModes(edf.DoF(7))
    sector_a = edf.NParticleSector(modes_a, 2).build()
    sector_b = edf.NParticleSector(modes_b, 3).build()

    assert sector_a.dimension == 10
    assert sector_b.dimension == 35
    assert sector_a.basis.n_modes == 5
    assert sector_b.basis.n_modes == 7


def test_large_vacuum_sector_preserves_mode_count():
    modes = edf.FermionModes(edf.DoF(70))
    sector = edf.NParticleSector(modes, 0).build()

    assert sector.basis.states == (0,)
    assert sector.basis.n_modes == 70


def _count_dof_value(state, modes, dof_index, value_index):
    count = 0
    for mode in range(modes.n_modes):
        if modes.unravel(mode)[dof_index] == value_index:
            count += (state >> mode) & 1
    return count


def test_one_dof_projection_builds_direct_fixed_population_sector():
    site = edf.DoF(4, name="site")
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    modes = edf.FermionModes(site, spin)

    sector = (
        edf.NParticleSector(modes, N=4)
        .project_particles("spin", up=2, down=2)
        .build()
    )

    assert sector.dimension == 36
    assert all(state.bit_count() == 4 for state in sector.basis.states)
    assert all(_count_dof_value(state, modes, 1, 0) == 2 for state in sector.basis)
    assert all(_count_dof_value(state, modes, 1, 1) == 2 for state in sector.basis)


def test_one_dof_projection_matches_filtering_reference_basis():
    site = edf.DoF(4, name="site")
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    modes = edf.FermionModes(site, spin)

    sector = (
        edf.NParticleSector(modes, N=4)
        .project_particles("spin", up=2, down=2)
        .build()
    )
    reference = tuple(
        state
        for state in edf.FockBasis(modes.n_modes, 4)
        if _count_dof_value(state, modes, 1, 0) == 2
        and _count_dof_value(state, modes, 1, 1) == 2
    )

    assert sector.basis.states == reference


def test_one_dof_projection_does_not_enumerate_full_fixed_n_basis(monkeypatch):
    site = edf.DoF(4, name="site")
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    modes = edf.FermionModes(site, spin)

    original = edf.FockBasis._enumerate_states

    def reject_full_sector(n_modes, N):
        if (n_modes, N) == (modes.n_modes, 4):
            raise AssertionError("full fixed-N enumeration was used")
        return original(n_modes, N)

    monkeypatch.setattr(edf.FockBasis, "_enumerate_states", reject_full_sector)

    sector = (
        edf.NParticleSector(modes, N=4)
        .project_particles("spin", up=2, down=2)
        .build()
    )

    assert sector.dimension == 36


def test_one_dof_projection_can_leave_labels_unspecified():
    site = edf.DoF(3, name="site")
    flavor = edf.DoF(3, name="flavor", labels=("a", "b", "c"))
    modes = edf.FermionModes(site, flavor)

    sector = edf.NParticleSector(modes, N=4).project_particles("flavor", a=2).build()

    assert sector.dimension == 45
    assert all(_count_dof_value(state, modes, 1, 0) == 2 for state in sector.basis)


def test_one_dof_projection_can_be_accumulated_before_build():
    site = edf.DoF(3, name="site")
    flavor = edf.DoF(3, name="flavor", labels=("a", "b", "c"))
    modes = edf.FermionModes(site, flavor)

    sector = (
        edf.NParticleSector(modes, N=3)
        .project_particles("flavor", a=1)
        .project_particles("flavor", b=1, c=1)
        .build()
    )

    assert sector.dimension == 27
    for value_index in range(3):
        assert all(
            _count_dof_value(state, modes, 1, value_index) == 1
            for state in sector.basis
        )


def test_project_particles_requires_labeled_named_dof():
    modes = edf.FermionModes(edf.DoF(4, name="site"), edf.DoF(2, name="spin"))
    sector = edf.NParticleSector(modes, N=2)

    with pytest.raises(ValueError, match="has no labels"):
        sector.project_particles("spin", up=1)


def test_project_particles_rejects_unknown_dof_and_label():
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    modes = edf.FermionModes(edf.DoF(2, name="site"), spin)
    sector = edf.NParticleSector(modes, N=2)

    with pytest.raises(ValueError, match="No degree of freedom"):
        sector.project_particles("layer", top=1)
    with pytest.raises(ValueError, match="Unknown label"):
        sector.project_particles("spin", left=1)


def test_project_particles_rejects_inconsistent_particle_numbers():
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    modes = edf.FermionModes(edf.DoF(2, name="site"), spin)

    with pytest.raises(ValueError, match="exceed the sector particle number"):
        edf.NParticleSector(modes, N=2).project_particles("spin", up=2, down=1)

    with pytest.raises(ValueError, match="must sum to sector N"):
        edf.NParticleSector(modes, N=3).project_particles("spin", up=1, down=1)

    with pytest.raises(ValueError, match="exceeds the 2 modes"):
        edf.NParticleSector(modes, N=3).project_particles("spin", up=3)


def test_project_particles_cannot_modify_built_sector():
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    modes = edf.FermionModes(edf.DoF(2, name="site"), spin)
    sector = edf.NParticleSector(modes, N=2).build()

    with pytest.raises(RuntimeError, match="before sector.build"):
        sector.project_particles("spin", up=1, down=1)


def test_one_dof_projection_above_64_modes_uses_multiword_backend():
    site = edf.DoF(35, name="site")
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    modes = edf.FermionModes(site, spin)

    sector = (
        edf.NParticleSector(modes, N=2)
        .project_particles("spin", up=1, down=1)
        .build()
    )

    assert modes.n_modes == 70
    assert sector.dimension == 35 * 35
    assert all(state.bit_count() == 2 for state in sector.basis.states)
    assert sector.basis.states[-1].bit_length() == 70


@pytest.mark.parametrize("dofs, projected_index", [
    (
        (
            edf.DoF(2, name="spin", labels=("up", "down")),
            edf.DoF(4, name="site"),
        ),
        0,
    ),
    (
        (
            edf.DoF(2, name="site"),
            edf.DoF(2, name="spin", labels=("up", "down")),
            edf.DoF(2, name="layer"),
        ),
        1,
    ),
])
def test_one_dof_projection_is_independent_of_dof_position(dofs, projected_index):
    modes = edf.FermionModes(*dofs)
    sector = (
        edf.NParticleSector(modes, N=4)
        .project_particles("spin", up=2, down=2)
        .build()
    )
    reference = tuple(
        state
        for state in edf.FockBasis(modes.n_modes, 4)
        if _count_dof_value(state, modes, projected_index, 0) == 2
        and _count_dof_value(state, modes, projected_index, 1) == 2
    )

    assert sector.basis.states == reference


def test_two_dof_projection_builds_joint_intersection_sector():
    site = edf.DoF(4, name="site")
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    layer = edf.DoF(2, name="layer", labels=("top", "bottom"))
    modes = edf.FermionModes(site, spin, layer)

    sector = (
        edf.NParticleSector(modes, N=4)
        .project_particles("spin", up=2, down=2)
        .project_particles("layer", top=2, bottom=2)
        .build()
    )

    assert sector.dimension == 328
    assert all(_count_dof_value(state, modes, 1, 0) == 2 for state in sector.basis)
    assert all(_count_dof_value(state, modes, 1, 1) == 2 for state in sector.basis)
    assert all(_count_dof_value(state, modes, 2, 0) == 2 for state in sector.basis)
    assert all(_count_dof_value(state, modes, 2, 1) == 2 for state in sector.basis)


def test_two_dof_projection_matches_filtering_reference_basis():
    site = edf.DoF(3, name="site")
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    layer = edf.DoF(2, name="layer", labels=("top", "bottom"))
    modes = edf.FermionModes(site, spin, layer)

    sector = (
        edf.NParticleSector(modes, N=4)
        .project_particles("spin", up=2, down=2)
        .project_particles("layer", top=3, bottom=1)
        .build()
    )
    reference = tuple(
        state
        for state in edf.FockBasis(modes.n_modes, 4)
        if _count_dof_value(state, modes, 1, 0) == 2
        and _count_dof_value(state, modes, 1, 1) == 2
        and _count_dof_value(state, modes, 2, 0) == 3
        and _count_dof_value(state, modes, 2, 1) == 1
    )

    assert sector.basis.states == reference


def test_two_dof_projection_supports_partial_constraints():
    site = edf.DoF(3, name="site")
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    layer = edf.DoF(3, name="layer", labels=("top", "middle", "bottom"))
    modes = edf.FermionModes(site, spin, layer)

    sector = (
        edf.NParticleSector(modes, N=4)
        .project_particles("spin", up=2)
        .project_particles("layer", top=1)
        .build()
    )
    reference = tuple(
        state
        for state in edf.FockBasis(modes.n_modes, 4)
        if _count_dof_value(state, modes, 1, 0) == 2
        and _count_dof_value(state, modes, 2, 0) == 1
    )

    assert sector.basis.states == reference


def test_two_dof_projection_does_not_enumerate_full_fixed_n_basis(monkeypatch):
    site = edf.DoF(3, name="site")
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    layer = edf.DoF(2, name="layer", labels=("top", "bottom"))
    modes = edf.FermionModes(site, spin, layer)

    original = edf.FockBasis._enumerate_states

    def reject_full_sector(n_modes, N):
        if (n_modes, N) == (modes.n_modes, 4):
            raise AssertionError("full fixed-N enumeration was used")
        return original(n_modes, N)

    monkeypatch.setattr(edf.FockBasis, "_enumerate_states", reject_full_sector)

    sector = (
        edf.NParticleSector(modes, N=4)
        .project_particles("spin", up=2, down=2)
        .project_particles("layer", top=2, bottom=2)
        .build()
    )

    assert sector.dimension > 0


def test_two_dof_projection_detects_cross_constraint_incompatibility():
    site = edf.DoF(1, name="site")
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    layer = edf.DoF(2, name="layer", labels=("top", "bottom"))
    modes = edf.FermionModes(site, spin, layer)

    sector = (
        edf.NParticleSector(modes, N=2)
        .project_particles("spin", up=2, down=0)
        .project_particles("layer", top=2, bottom=0)
    )

    with pytest.raises(ValueError, match="mutually incompatible"):
        sector.build()


def test_two_dof_projection_is_independent_of_projected_dof_order():
    layer = edf.DoF(2, name="layer", labels=("top", "bottom"))
    site = edf.DoF(3, name="site")
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    modes = edf.FermionModes(layer, site, spin)

    first = (
        edf.NParticleSector(modes, N=4)
        .project_particles("spin", up=2, down=2)
        .project_particles("layer", top=2, bottom=2)
        .build()
    )
    second = (
        edf.NParticleSector(modes, N=4)
        .project_particles("layer", top=2, bottom=2)
        .project_particles("spin", up=2, down=2)
        .build()
    )

    assert first.basis.states == second.basis.states


def test_two_dof_projection_above_64_modes_uses_multiword_backend():
    site = edf.DoF(17, name="site")
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    layer = edf.DoF(2, name="layer", labels=("top", "bottom"))
    modes = edf.FermionModes(site, spin, layer)

    sector = (
        edf.NParticleSector(modes, N=2)
        .project_particles("spin", up=1, down=1)
        .project_particles("layer", top=1, bottom=1)
        .build()
    )

    assert modes.n_modes == 68
    assert sector.dimension == 2 * 17 * 17
    assert sector.basis.states[-1].bit_length() == 68


def test_three_dof_projection_builds_joint_intersection_sector():
    site = edf.DoF(2, name="site")
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    layer = edf.DoF(2, name="layer", labels=("top", "bottom"))
    orbital = edf.DoF(2, name="orbital", labels=("a", "b"))
    modes = edf.FermionModes(site, spin, layer, orbital)

    sector = (
        edf.NParticleSector(modes, N=4)
        .project_particles("spin", up=2, down=2)
        .project_particles("layer", top=2, bottom=2)
        .project_particles("orbital", a=2, b=2)
        .build()
    )

    assert sector.dimension == 132
    for dof_index in (1, 2, 3):
        assert all(
            _count_dof_value(state, modes, dof_index, 0) == 2
            for state in sector.basis
        )
        assert all(
            _count_dof_value(state, modes, dof_index, 1) == 2
            for state in sector.basis
        )


def test_three_dof_projection_matches_filtering_reference_basis():
    site = edf.DoF(2, name="site")
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    layer = edf.DoF(2, name="layer", labels=("top", "bottom"))
    orbital = edf.DoF(2, name="orbital", labels=("a", "b"))
    modes = edf.FermionModes(site, spin, layer, orbital)

    sector = (
        edf.NParticleSector(modes, N=4)
        .project_particles("spin", up=2, down=2)
        .project_particles("layer", top=3, bottom=1)
        .project_particles("orbital", a=2, b=2)
        .build()
    )
    reference = tuple(
        state
        for state in edf.FockBasis(modes.n_modes, 4)
        if _count_dof_value(state, modes, 1, 0) == 2
        and _count_dof_value(state, modes, 1, 1) == 2
        and _count_dof_value(state, modes, 2, 0) == 3
        and _count_dof_value(state, modes, 2, 1) == 1
        and _count_dof_value(state, modes, 3, 0) == 2
        and _count_dof_value(state, modes, 3, 1) == 2
    )

    assert sector.basis.states == reference


def test_three_dof_projection_supports_partial_constraints():
    site = edf.DoF(2, name="site")
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    layer = edf.DoF(3, name="layer", labels=("top", "middle", "bottom"))
    orbital = edf.DoF(2, name="orbital", labels=("a", "b"))
    modes = edf.FermionModes(site, spin, layer, orbital)

    sector = (
        edf.NParticleSector(modes, N=4)
        .project_particles("spin", up=2)
        .project_particles("layer", top=1)
        .project_particles("orbital", a=2)
        .build()
    )
    reference = tuple(
        state
        for state in edf.FockBasis(modes.n_modes, 4)
        if _count_dof_value(state, modes, 1, 0) == 2
        and _count_dof_value(state, modes, 2, 0) == 1
        and _count_dof_value(state, modes, 3, 0) == 2
    )

    assert sector.basis.states == reference


def test_four_dof_projection_is_supported():
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    layer = edf.DoF(2, name="layer", labels=("top", "bottom"))
    orbital = edf.DoF(2, name="orbital", labels=("a", "b"))
    valley = edf.DoF(2, name="valley", labels=("K", "Kp"))
    modes = edf.FermionModes(spin, layer, orbital, valley)

    sector = (
        edf.NParticleSector(modes, N=4)
        .project_particles("spin", up=2, down=2)
        .project_particles("layer", top=2, bottom=2)
        .project_particles("orbital", a=2, b=2)
        .project_particles("valley", K=2, Kp=2)
        .build()
    )
    reference = tuple(
        state
        for state in edf.FockBasis(modes.n_modes, 4)
        if all(
            _count_dof_value(state, modes, dof_index, 0) == 2
            and _count_dof_value(state, modes, dof_index, 1) == 2
            for dof_index in range(4)
        )
    )

    assert sector.basis.states == reference


def test_multi_dof_projection_does_not_enumerate_full_fixed_n_basis(monkeypatch):
    site = edf.DoF(2, name="site")
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    layer = edf.DoF(2, name="layer", labels=("top", "bottom"))
    orbital = edf.DoF(2, name="orbital", labels=("a", "b"))
    modes = edf.FermionModes(site, spin, layer, orbital)

    original = edf.FockBasis._enumerate_states

    def reject_full_sector(n_modes, N):
        if (n_modes, N) == (modes.n_modes, 4):
            raise AssertionError("full fixed-N enumeration was used")
        return original(n_modes, N)

    monkeypatch.setattr(edf.FockBasis, "_enumerate_states", reject_full_sector)

    sector = (
        edf.NParticleSector(modes, N=4)
        .project_particles("spin", up=2, down=2)
        .project_particles("layer", top=2, bottom=2)
        .project_particles("orbital", a=2, b=2)
        .build()
    )

    assert sector.dimension == 132


def test_multi_dof_projection_detects_cross_constraint_incompatibility():
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    layer = edf.DoF(2, name="layer", labels=("top", "bottom"))
    orbital = edf.DoF(2, name="orbital", labels=("a", "b"))
    modes = edf.FermionModes(spin, layer, orbital)

    sector = (
        edf.NParticleSector(modes, N=2)
        .project_particles("spin", up=2, down=0)
        .project_particles("layer", top=2, bottom=0)
        .project_particles("orbital", a=2, b=0)
    )

    with pytest.raises(ValueError, match="mutually incompatible"):
        sector.build()


def test_multi_dof_projection_is_independent_of_projection_order():
    orbital = edf.DoF(2, name="orbital", labels=("a", "b"))
    site = edf.DoF(2, name="site")
    layer = edf.DoF(2, name="layer", labels=("top", "bottom"))
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    modes = edf.FermionModes(orbital, site, layer, spin)

    first = (
        edf.NParticleSector(modes, N=4)
        .project_particles("spin", up=2, down=2)
        .project_particles("layer", top=2, bottom=2)
        .project_particles("orbital", a=2, b=2)
        .build()
    )
    second = (
        edf.NParticleSector(modes, N=4)
        .project_particles("orbital", a=2, b=2)
        .project_particles("spin", up=2, down=2)
        .project_particles("layer", top=2, bottom=2)
        .build()
    )

    assert first.basis.states == second.basis.states


def test_multi_dof_projection_above_64_modes_uses_multiword_backend():
    site = edf.DoF(9, name="site")
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    layer = edf.DoF(2, name="layer", labels=("top", "bottom"))
    orbital = edf.DoF(2, name="orbital", labels=("a", "b"))
    modes = edf.FermionModes(site, spin, layer, orbital)

    sector = (
        edf.NParticleSector(modes, N=2)
        .project_particles("spin", up=1, down=1)
        .project_particles("layer", top=1, bottom=1)
        .project_particles("orbital", a=1, b=1)
        .build()
    )

    assert modes.n_modes == 72
    assert sector.dimension == 4 * 9 * 9
    assert sector.basis.states[-1].bit_length() == 72


def test_one_dof_projection_above_128_modes_matches_reference_basis():
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    site = edf.DoF(65, name="site")
    modes = edf.FermionModes(spin, site)

    sector = (
        edf.NParticleSector(modes, N=2)
        .project_particles("spin", up=1, down=1)
        .build()
    )
    reference = tuple(
        state
        for state in edf.FockBasis(modes.n_modes, 2)
        if _count_dof_value(state, modes, 0, 0) == 1
        and _count_dof_value(state, modes, 0, 1) == 1
    )

    assert modes.n_modes == 130
    assert sector.dimension == 65 * 65
    assert sector.basis.states == reference
    assert sector.basis.states[-1].bit_length() == 130



def test_multiword_backend_handles_interleaved_modes_across_word_boundary():
    site = edf.DoF(35, name="site")
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    modes = edf.FermionModes(site, spin)

    sector = (
        edf.NParticleSector(modes, N=2)
        .project_particles("spin", up=1, down=1)
        .build()
    )
    reference = tuple(
        state
        for state in edf.FockBasis(modes.n_modes, 2)
        if _count_dof_value(state, modes, 1, 0) == 1
        and _count_dof_value(state, modes, 1, 1) == 1
    )

    assert modes.n_modes == 70
    assert sector.basis.states == reference
