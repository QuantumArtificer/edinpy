from edinpy import boson as edb


def _count_dof_value(occupations, modes, dof_index, value_index):
    return sum(
        occupations[mode]
        for mode in range(modes.n_modes)
        if modes.unravel(mode)[dof_index] == value_index
    )


def test_sector_is_lazy_and_build_is_idempotent():
    modes = edb.BosonModes(edb.DoF(4, name="site"))
    sector = edb.NParticleSector(modes, N=3)

    assert not sector.is_built
    first = sector.build()
    basis = sector.basis
    second = sector.build()

    assert first is sector
    assert second is sector
    assert sector.basis is basis
    assert sector.dimension == 20


def test_one_dof_projection_builds_direct_fixed_populations():
    site = edb.DoF(3, name="site")
    species = edb.DoF(2, name="species", labels=("a", "b"))
    modes = edb.BosonModes(site, species)

    sector = (
        edb.NParticleSector(modes, N=3)
        .project_particles("species", a=2, b=1)
        .build()
    )

    assert sector.dimension == 18
    assert all(
        _count_dof_value(state, modes, 1, 0) == 2
        and _count_dof_value(state, modes, 1, 1) == 1
        for state in sector.basis.states
    )


def test_one_dof_partial_projection_merges_unconstrained_values():
    site = edb.DoF(2, name="site")
    species = edb.DoF(3, name="species", labels=("a", "b", "c"))
    modes = edb.BosonModes(site, species)

    sector = (
        edb.NParticleSector(modes, N=3)
        .project_particles("species", a=1)
        .build()
    )

    assert sector.dimension == 20
    assert all(
        _count_dof_value(state, modes, 1, 0) == 1
        for state in sector.basis.states
    )


def test_one_dof_projection_does_not_enumerate_full_fixed_n_basis(monkeypatch):
    site = edb.DoF(3, name="site")
    species = edb.DoF(2, name="species", labels=("a", "b"))
    modes = edb.BosonModes(site, species)

    original = edb.FockBasis._enumerate_packed_states

    def reject_full_sector(n_modes, N):
        if (n_modes, N) == (modes.n_modes, 3):
            raise AssertionError("full fixed-N enumeration was used")
        return original(n_modes, N)

    monkeypatch.setattr(edb.FockBasis, "_enumerate_packed_states", reject_full_sector)

    sector = (
        edb.NParticleSector(modes, N=3)
        .project_particles("species", a=2, b=1)
        .build()
    )
    assert sector.dimension == 18


def test_two_dof_projection_matches_filtering_reference_basis():
    site = edb.DoF(2, name="site")
    species = edb.DoF(2, name="species", labels=("a", "b"))
    layer = edb.DoF(2, name="layer", labels=("top", "bottom"))
    modes = edb.BosonModes(site, species, layer)

    sector = (
        edb.NParticleSector(modes, N=3)
        .project_particles("species", a=2, b=1)
        .project_particles("layer", top=1, bottom=2)
        .build()
    )
    reference = tuple(
        state
        for state in edb.FockBasis(modes.n_modes, 3).states
        if _count_dof_value(state, modes, 1, 0) == 2
        and _count_dof_value(state, modes, 1, 1) == 1
        and _count_dof_value(state, modes, 2, 0) == 1
        and _count_dof_value(state, modes, 2, 1) == 2
    )

    assert sector.basis.states == reference


def test_three_dof_projection_matches_filtering_reference_basis():
    site = edb.DoF(2, name="site")
    species = edb.DoF(2, name="species", labels=("a", "b"))
    layer = edb.DoF(2, name="layer", labels=("top", "bottom"))
    orbital = edb.DoF(2, name="orbital", labels=("x", "y"))
    modes = edb.BosonModes(site, species, layer, orbital)

    sector = (
        edb.NParticleSector(modes, N=3)
        .project_particles("species", a=2, b=1)
        .project_particles("layer", top=1, bottom=2)
        .project_particles("orbital", x=2, y=1)
        .build()
    )
    reference = tuple(
        state
        for state in edb.FockBasis(modes.n_modes, 3).states
        if _count_dof_value(state, modes, 1, 0) == 2
        and _count_dof_value(state, modes, 1, 1) == 1
        and _count_dof_value(state, modes, 2, 0) == 1
        and _count_dof_value(state, modes, 2, 1) == 2
        and _count_dof_value(state, modes, 3, 0) == 2
        and _count_dof_value(state, modes, 3, 1) == 1
    )

    assert sector.basis.states == reference


def test_projection_order_is_irrelevant():
    site = edb.DoF(2, name="site")
    species = edb.DoF(2, name="species", labels=("a", "b"))
    layer = edb.DoF(2, name="layer", labels=("top", "bottom"))
    modes = edb.BosonModes(site, species, layer)

    first = (
        edb.NParticleSector(modes, N=3)
        .project_particles("species", a=2, b=1)
        .project_particles("layer", top=1, bottom=2)
        .build()
    )
    second = (
        edb.NParticleSector(modes, N=3)
        .project_particles("layer", top=1, bottom=2)
        .project_particles("species", a=2, b=1)
        .build()
    )

    assert first.basis.packed_states == second.basis.packed_states



def test_two_dof_projection_does_not_enumerate_full_fixed_n_basis(monkeypatch):
    site = edb.DoF(2, name="site")
    species = edb.DoF(2, name="species", labels=("a", "b"))
    layer = edb.DoF(2, name="layer", labels=("top", "bottom"))
    modes = edb.BosonModes(site, species, layer)

    original = edb.FockBasis._enumerate_packed_states

    def reject_full_sector(n_modes, N):
        if (n_modes, N) == (modes.n_modes, 3):
            raise AssertionError("full fixed-N enumeration was used")
        return original(n_modes, N)

    monkeypatch.setattr(edb.FockBasis, "_enumerate_packed_states", reject_full_sector)

    sector = (
        edb.NParticleSector(modes, N=3)
        .project_particles("species", a=2, b=1)
        .project_particles("layer", top=1, bottom=2)
        .build()
    )
    assert sector.dimension > 0


def test_multi_dof_projection_does_not_enumerate_full_fixed_n_basis(monkeypatch):
    site = edb.DoF(2, name="site")
    species = edb.DoF(2, name="species", labels=("a", "b"))
    layer = edb.DoF(2, name="layer", labels=("top", "bottom"))
    orbital = edb.DoF(2, name="orbital", labels=("x", "y"))
    modes = edb.BosonModes(site, species, layer, orbital)

    original = edb.FockBasis._enumerate_packed_states

    def reject_full_sector(n_modes, N):
        if (n_modes, N) == (modes.n_modes, 3):
            raise AssertionError("full fixed-N enumeration was used")
        return original(n_modes, N)

    monkeypatch.setattr(edb.FockBasis, "_enumerate_packed_states", reject_full_sector)

    sector = (
        edb.NParticleSector(modes, N=3)
        .project_particles("species", a=2, b=1)
        .project_particles("layer", top=1, bottom=2)
        .project_particles("orbital", x=2, y=1)
        .build()
    )
    assert sector.dimension > 0


def test_projected_sector_can_span_more_than_64_packed_bits():
    site = edb.DoF(17, name="site")
    species = edb.DoF(2, name="species", labels=("a", "b"))
    modes = edb.BosonModes(site, species)

    sector = (
        edb.NParticleSector(modes, N=3)
        .project_particles("species", a=2, b=1)
        .build()
    )

    assert sector.basis.packed_width == 68
    assert sector.dimension == 153 * 17
    assert sector.basis.packed_states[-1].bit_length() > 64


def test_projection_validation_rejects_inconsistent_complete_marginal():
    species = edb.DoF(2, name="species", labels=("a", "b"))
    modes = edb.BosonModes(species)

    sector = edb.NParticleSector(modes, N=3)
    try:
        sector.project_particles("species", a=1, b=1)
    except ValueError as exc:
        assert "must sum to sector N" in str(exc)
    else:
        raise AssertionError("inconsistent complete marginal was accepted")


def test_projected_basis_finalization_does_not_redecode_every_state(monkeypatch):
    import edinpy.boson._basis as basis_module

    site = edb.DoF(4, name="site")
    species = edb.DoF(2, name="species", labels=("a", "b"))
    modes = edb.BosonModes(site, species)

    def reject_unpack(self, state):
        raise AssertionError("projected basis finalization decoded packed states")

    monkeypatch.setattr(basis_module._OccupationCodec, "unpack", reject_unpack)

    sector = (
        edb.NParticleSector(modes, N=4)
        .project_particles("species", a=2, b=2)
        .build()
    )

    assert sector.dimension == 100


def test_projection_accepts_integer_indices_without_dof_labels():
    site = edb.DoF(3, name="site")
    modes = edb.BosonModes(site)

    sector = (
        edb.NParticleSector(modes, N=3)
        .project_particles("site", {0: 2, 2: 1})
        .build()
    )

    assert sector.dimension == 1
    assert sector.basis.states == ((2, 0, 1),)


def test_projection_mapping_accepts_labels_and_indices_together():
    species = edb.DoF(3, name="species", labels=("a", "b", "c"))
    modes = edb.BosonModes(species)

    mapped = (
        edb.NParticleSector(modes, N=4)
        .project_particles("species", {"a": 2, 1: 1, "c": 1})
        .build()
    )
    keyword = (
        edb.NParticleSector(modes, N=4)
        .project_particles("species", a=2, b=1, c=1)
        .build()
    )

    assert mapped.basis.packed_states == keyword.basis.packed_states


def test_projection_mapping_can_be_combined_with_keyword_shorthand():
    species = edb.DoF(2, name="species", labels=("a", "b"))
    modes = edb.BosonModes(species)

    sector = (
        edb.NParticleSector(modes, N=3)
        .project_particles("species", {0: 2}, b=1)
        .build()
    )

    assert sector.basis.states == ((2, 1),)


def test_projection_alias_conflict_is_rejected():
    species = edb.DoF(2, name="species", labels=("a", "b"))
    modes = edb.BosonModes(species)
    sector = edb.NParticleSector(modes, N=3)

    try:
        sector.project_particles("species", {0: 2}, a=1)
    except ValueError as exc:
        assert "already set" in str(exc)
    else:
        raise AssertionError("conflicting projection aliases were accepted")


def test_unlabeled_projection_rejects_string_keys_with_guidance():
    site = edb.DoF(2, name="site")
    modes = edb.BosonModes(site)
    sector = edb.NParticleSector(modes, N=2)

    try:
        sector.project_particles("site", {"left": 1})
    except ValueError as exc:
        assert "use integer value indices" in str(exc)
    else:
        raise AssertionError("string key was accepted for an unlabeled DoF")


def test_sector_from_vector_requires_build():
    import numpy as np
    import pytest

    modes = edb.BosonModes(edb.DoF(3))
    sector = edb.NParticleSector(modes, 2)

    with pytest.raises(RuntimeError):
        sector.from_vector(np.ones(6))


def test_independent_sectors_do_not_share_state():
    modes = edb.BosonModes(edb.DoF(4))
    first = edb.NParticleSector(modes, 2).build()
    second = edb.NParticleSector(modes, 2).build()

    assert first is not second
    assert first.basis is not second.basis
    assert first.basis.packed_states == second.basis.packed_states


def test_large_vacuum_sector_preserves_mode_count():
    modes = edb.BosonModes(edb.DoF(130))
    sector = edb.NParticleSector(modes, 0).build()

    assert sector.dimension == 1
    assert sector.basis.n_modes == 130
    assert sector.basis.states == ((0,) * 130,)


def test_one_dof_projection_can_be_accumulated_before_build():
    site = edb.DoF(3, name="site")
    species = edb.DoF(3, name="species", labels=("a", "b", "c"))
    modes = edb.BosonModes(site, species)
    sector = edb.NParticleSector(modes, 4)

    sector.project_particles("species", a=2)
    sector.project_particles("species", b=1)
    sector.build()

    assert all(
        _count_dof_value(state, modes, 1, 0) == 2
        and _count_dof_value(state, modes, 1, 1) == 1
        and _count_dof_value(state, modes, 1, 2) == 1
        for state in sector.basis.states
    )


def test_project_particles_rejects_unknown_dof_and_label():
    import pytest

    species = edb.DoF(2, name="species", labels=("a", "b"))
    modes = edb.BosonModes(species)
    sector = edb.NParticleSector(modes, 2)

    with pytest.raises(ValueError, match="No degree of freedom"):
        sector.project_particles("spin", a=1)
    with pytest.raises(ValueError, match="Unknown label"):
        sector.project_particles("species", c=1)


def test_project_particles_cannot_modify_built_sector():
    import pytest

    species = edb.DoF(2, name="species", labels=("a", "b"))
    modes = edb.BosonModes(species)
    sector = edb.NParticleSector(modes, 2).build()

    with pytest.raises(RuntimeError):
        sector.project_particles("species", a=1)


def test_two_dof_projection_supports_partial_constraints():
    site = edb.DoF(2, name="site")
    species = edb.DoF(3, name="species", labels=("a", "b", "c"))
    layer = edb.DoF(3, name="layer", labels=("top", "middle", "bottom"))
    modes = edb.BosonModes(site, species, layer)

    sector = (
        edb.NParticleSector(modes, N=4)
        .project_particles("species", a=2)
        .project_particles("layer", top=1)
        .build()
    )

    assert sector.dimension > 0
    assert all(
        _count_dof_value(state, modes, 1, 0) == 2
        and _count_dof_value(state, modes, 2, 0) == 1
        for state in sector.basis.states
    )


def test_four_dof_projection_is_supported():
    site = edb.DoF(2, name="site")
    species = edb.DoF(2, name="species", labels=("a", "b"))
    layer = edb.DoF(2, name="layer", labels=("top", "bottom"))
    orbital = edb.DoF(2, name="orbital", labels=("x", "y"))
    valley = edb.DoF(2, name="valley", labels=("K", "Kp"))
    modes = edb.BosonModes(site, species, layer, orbital, valley)

    sector = (
        edb.NParticleSector(modes, N=3)
        .project_particles("species", a=2, b=1)
        .project_particles("layer", top=1, bottom=2)
        .project_particles("orbital", x=2, y=1)
        .project_particles("valley", K=1, Kp=2)
        .build()
    )

    assert sector.dimension > 0
    for state in sector.basis.states:
        assert _count_dof_value(state, modes, 1, 0) == 2
        assert _count_dof_value(state, modes, 2, 0) == 1
        assert _count_dof_value(state, modes, 3, 0) == 2
        assert _count_dof_value(state, modes, 4, 0) == 1


def test_projected_sector_above_128_packed_bits_matches_reference_constraints():
    site = edb.DoF(33, name="site")
    species = edb.DoF(2, name="species", labels=("a", "b"))
    modes = edb.BosonModes(site, species)

    sector = (
        edb.NParticleSector(modes, N=3)
        .project_particles("species", a=2, b=1)
        .build()
    )

    assert sector.basis.packed_width == 132
    assert sector.dimension == 561 * 33
    assert all(
        _count_dof_value(state, modes, 1, 0) == 2
        and _count_dof_value(state, modes, 1, 1) == 1
        for state in sector.basis.states
    )
