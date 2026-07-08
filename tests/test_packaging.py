# -*- coding: utf-8 -*-
"""Tests that fedclypse follows its packaging and documentation conventions."""
from __future__ import annotations

import importlib
import pathlib
from importlib.resources import files

import pytest

import fedclypse

# The deliberate public API surface (concept-named modules; no generic buckets).
CONCEPT_MODULES = [
    "fedclypse.parameters",
    "fedclypse.contribution",
    "fedclypse.aggregation",
    "fedclypse.selection",
    "fedclypse.synchronization",
    "fedclypse.compression",
    "fedclypse.data",
    "fedclypse.partition",
    "fedclypse.model",
    "fedclypse.entity",
    "fedclypse.runtime",
    "fedclypse.fedavg",
    "fedclypse.metrics",
]
TORCH_MODULES = ["fedclypse.torch", "fedclypse.torch.model"]


def test_package_has_docstring_and_version():
    """The top-level package exposes a docstring and a version."""
    assert fedclypse.__doc__, "fedclypse package missing a module docstring"
    assert getattr(fedclypse, "__version__", None), "fedclypse missing __version__"


def test_py_typed_marker_is_packaged():
    """The PEP 561 ``py.typed`` marker ships with the package."""
    assert (files("fedclypse") / "py.typed").is_file()


def test_concept_module_list_matches_disk():
    """Every top-level ``fedclypse/*.py`` module is listed in CONCEPT_MODULES."""
    pkg_dir = pathlib.Path(fedclypse.__file__).parent
    on_disk = {p.stem for p in pkg_dir.glob("*.py") if not p.stem.startswith("_")}
    listed = {m.rsplit(".", 1)[-1] for m in CONCEPT_MODULES}
    assert on_disk == listed, f"module list out of sync: {on_disk ^ listed}"


@pytest.mark.parametrize("modname", CONCEPT_MODULES)
def test_concept_module_documented(modname):
    """Each concept module has a docstring, an ``__all__``, and resolvable exports."""
    module = importlib.import_module(modname)
    assert module.__doc__, f"{modname} missing a module docstring"
    assert hasattr(module, "__all__"), f"{modname} missing __all__"
    for name in module.__all__:
        assert hasattr(module, name), f"{modname}.__all__ names undefined {name!r}"


@pytest.mark.parametrize("modname", TORCH_MODULES)
def test_torch_module_documented(modname):
    """Each torch-subpackage module (torch extra) is documented the same way."""
    pytest.importorskip("torch")
    module = importlib.import_module(modname)
    assert module.__doc__, f"{modname} missing a module docstring"
    assert hasattr(module, "__all__"), f"{modname} missing __all__"
    for name in module.__all__:
        assert hasattr(module, name), f"{modname}.__all__ names undefined {name!r}"
