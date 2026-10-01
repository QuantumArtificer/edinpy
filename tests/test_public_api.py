"""Checks for the documented top-level and statistics-specific public APIs."""

import inspect

import edinpy
from edinpy import boson as edb
from edinpy import fermion as edf


def test_public_exports_are_unique_and_documented():
    """Package and statistics-specific exports are deliberate and documented."""
    assert edinpy.__all__ == ["fermion", "boson", "__version__"]
    assert isinstance(edinpy.__version__, str)
    assert edinpy.__version__

    for api in (edf, edb):
        names = api.__all__
        assert len(names) == len(set(names))
        assert all(not name.startswith("_") for name in names)

        missing = [name for name in names if inspect.getdoc(getattr(api, name)) is None]
        assert missing == []


def test_public_class_members_are_documented():
    """Public methods and properties on exported classes have docstrings."""
    missing = []
    for api in (edf, edb):
        for class_name in api.__all__:
            cls = getattr(api, class_name)
            if not inspect.isclass(cls):
                continue
            for member_name, member in cls.__dict__.items():
                if member_name.startswith("_"):
                    continue
                if inspect.isfunction(member) or isinstance(member, property):
                    if inspect.getdoc(member) is None:
                        missing.append(f"{api.__name__}.{class_name}.{member_name}")

    assert missing == []


def test_null_state_is_part_of_symbolic_state_algebra():
    """The shared null ket remains available through both public APIs."""
    for api, state in (
        (edf, edf.FockState(1, n_modes=2)),
        (edb, edb.FockState((1, 0))),
    ):
        zero = api.NullState()
        assert not zero
        assert zero.norm() == 0.0
        assert zero + state == state
