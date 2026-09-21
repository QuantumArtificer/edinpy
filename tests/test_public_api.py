"""Checks for the documented fermionic public API."""

import inspect

from edinpy import fermion as edf


def test_public_exports_are_unique_and_documented():
    names = edf.__all__

    assert len(names) == len(set(names))
    assert all(not name.startswith("_") for name in names)

    missing = [name for name in names if inspect.getdoc(getattr(edf, name)) is None]
    assert missing == []


def test_public_class_members_are_documented():
    missing = []
    for class_name in edf.__all__:
        cls = getattr(edf, class_name)
        if not inspect.isclass(cls):
            continue
        for member_name, member in cls.__dict__.items():
            if member_name.startswith("_"):
                continue
            if inspect.isfunction(member) or isinstance(member, property):
                if inspect.getdoc(member) is None:
                    missing.append(f"{class_name}.{member_name}")

    assert missing == []


def test_null_state_is_part_of_symbolic_state_algebra():
    zero = edf.NullState()

    assert not zero
    assert zero.norm() == 0.0
    assert zero + edf.FockState(1, n_modes=2) == edf.FockState(1, n_modes=2)
