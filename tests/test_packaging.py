# -*- coding: utf-8 -*-
"""Tests that fedclypse follows its packaging and documentation conventions."""
from __future__ import annotations

import importlib
import pkgutil
from importlib.resources import files

import pytest

import fedclypse


def _public_modules():
    """Discover every public fedclypse module/subpackage (skipping ``_`` names)."""
    found = []
    # onerror keeps discovery robust if a subpackage ever fails to import.
    for info in pkgutil.walk_packages(
        fedclypse.__path__, prefix="fedclypse.", onerror=lambda name: None
    ):
        short = info.name.rsplit(".", 1)[-1]
        if short.startswith("_"):
            continue
        found.append(info.name)
    return sorted(found)


def test_package_has_docstring_and_version():
    """The top-level package exposes a docstring and a version."""
    assert fedclypse.__doc__, "fedclypse package missing a module docstring"
    assert getattr(fedclypse, "__version__", None), "fedclypse missing __version__"


def test_py_typed_marker_is_packaged():
    """The PEP 561 ``py.typed`` marker ships with the package."""
    assert (files("fedclypse") / "py.typed").is_file()


@pytest.mark.parametrize("modname", _public_modules())
def test_module_documented(modname):
    """Every public module/subpackage has a docstring, an ``__all__``, and resolvable exports."""
    module = importlib.import_module(modname)
    assert module.__doc__, f"{modname} missing a module docstring"
    assert hasattr(module, "__all__"), f"{modname} missing __all__"
    for name in module.__all__:
        assert hasattr(module, name), f"{modname}.__all__ names undefined {name!r}"
