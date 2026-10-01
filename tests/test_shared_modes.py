import pytest

from edinpy import boson as edb
from edinpy import fermion as edf


def test_dof_is_one_shared_public_type():
    assert edf.DoF is edb.DoF


def test_shared_dof_can_be_used_by_either_mode_space():
    site = edf.DoF(3, name="site")
    component = edb.DoF(2, name="component", labels=("a", "b"))

    fermions = edf.FermionModes(site, component)
    bosons = edb.BosonModes(site, component)

    assert fermions.dofs == bosons.dofs == (site, component)
    assert fermions.strides == bosons.strides == (1, 3)
    assert fermions.n_modes == bosons.n_modes == 6

    for mode in range(6):
        assert fermions.unravel(mode) == bosons.unravel(mode)
        indices = fermions.unravel(mode)
        assert fermions.resolve(indices) == mode
        assert bosons.resolve(indices) == mode


def test_statistics_specific_mode_types_remain_distinct():
    dof = edf.DoF(4)
    fermions = edf.FermionModes(dof)
    bosons = edb.BosonModes(dof)

    assert type(fermions) is not type(bosons)
    assert repr(fermions).startswith("FermionModes(")
    assert repr(bosons).startswith("BosonModes(")


def test_shared_mode_validation_matches_both_public_modules():
    site = edf.DoF(2, name="site")
    spin = edf.DoF(2, name="spin")

    for mode_type in (edf.FermionModes, edb.BosonModes):
        modes = mode_type(site, spin)
        with pytest.raises(ValueError, match="Expected 2 indices"):
            modes.resolve(0)
        with pytest.raises(IndexError, match="site"):
            modes.resolve((2, 0))
        with pytest.raises(TypeError, match="integers"):
            modes.resolve((0.5, 0))
        with pytest.raises(IndexError, match="outside"):
            modes.unravel(4)
