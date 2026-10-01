import pytest

from edinpy import boson as edb


def test_dof_labels_are_validated():
    with pytest.raises(ValueError, match="exactly 'size'"):
        edb.DoF(2, labels=("a",))

    with pytest.raises(ValueError, match="unique"):
        edb.DoF(2, labels=("a", "a"))


def test_boson_modes_resolve_and_unravel_mixed_radix_indices():
    site = edb.DoF(3, name="site")
    species = edb.DoF(2, name="species", labels=("a", "b"))
    layer = edb.DoF(2, name="layer", labels=("top", "bottom"))
    modes = edb.BosonModes(site, species, layer)

    assert modes.n_modes == 12
    assert modes.strides == (1, 3, 6)
    assert modes.resolve((2, 1, 1)) == 11
    assert modes.unravel(11) == (2, 1, 1)


def test_dof_is_pure_descriptor():
    dof = edb.DoF(3, name="site")

    assert dof.size == 3
    assert dof.name == "site"
    assert dof.labels is None


def test_dof_accepts_ordered_labels():
    dof = edb.DoF(3, name="species", labels=("a", "b", "c"))

    assert dof.labels == ("a", "b", "c")


def test_dof_rejects_nonpositive_size():
    import pytest

    with pytest.raises(ValueError):
        edb.DoF(0)
    with pytest.raises(ValueError):
        edb.DoF(-1)


def test_boson_modes_mapping_and_strides():
    site = edb.DoF(3, name="site")
    species = edb.DoF(2, name="species")
    orbital = edb.DoF(2, name="orbital")
    modes = edb.BosonModes(site, species, orbital)

    assert modes.strides == (1, 3, 6)
    assert modes.n_modes == 12
    assert len(modes) == 12
    assert modes.resolve((2, 1, 1)) == 11


def test_boson_modes_round_trip_all_modes():
    modes = edb.BosonModes(edb.DoF(4), edb.DoF(3), edb.DoF(2))

    for mode in range(modes.n_modes):
        assert modes.resolve(modes.unravel(mode)) == mode


def test_boson_modes_rejects_wrong_number_of_indices():
    import pytest

    modes = edb.BosonModes(edb.DoF(2), edb.DoF(3))

    with pytest.raises(ValueError):
        modes.resolve(1)
    with pytest.raises(ValueError):
        modes.resolve((1,))
    with pytest.raises(ValueError):
        modes.resolve((1, 2, 0))


def test_single_dof_accepts_bare_integer_index():
    modes = edb.BosonModes(edb.DoF(5))

    assert modes.resolve(3) == 3
    assert modes.resolve((3,)) == 3
