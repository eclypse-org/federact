<p align="center">
<b><span style="font-size: 2.5em;">fedclypse</span></b>
</p>

![PyPI - Version](https://img.shields.io/pypi/v/fedclypse)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fedclypse)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&)](https://github.com/pre-commit/pre-commit)
[![codecov](https://codecov.io/github/eclypse-org/fedclypse/graph/badge.svg)](https://codecov.io/github/eclypse-org/fedclypse)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Import sorted with isort](https://img.shields.io/badge/isort-checked-brightgreen)](https://pycqa.github.io/isort/)
[![Doc style: docformatter](https://img.shields.io/badge/doc%20style-docformatter-blue)](https://github.com/PyCQA/docformatter)

**fedclypse** is Federated Learning as a framework-agnostic verticalization of
[ECLYPSE](https://github.com/eclypse-org/eclypse). It turns an eclypse `Simulation` into a
federation: FL participants are eclypse `Service` entities, the FL round is composable
mechanics (selection × synchronization × aggregation × optimization) layered on top of
eclypse's message passing, and a run is driven and observed the same way any eclypse
experiment is. The exchange currency is plain numpy, so the base install trains real
models with nothing but numpy — bring your own framework by subclassing
`fedclypse.core.Model` and overriding `Learner.local_update`.

## Installation

To install fedclypse and all its dependencies, you can run the following command:

```bash

pip install fedclypse

```

**N.B.** We **strongly** suggest the installation of fedclypse in a virtual environment.

## Documentation

The documentation for fedclypse can be found [here](https://fedclypse.readthedocs.io/en/latest/).

## Citation

fedclypse is built on top of ECLYPSE. If you use it in your work or research, please cite
this repository (see [`CITATION.cff`](CITATION.cff)) and the ECLYPSE framework:

```bibtex
@article{massa2026eclypse,
  title     = {{ECLYPSE: A Python Framework for Simulation and Emulation of the Cloud-Edge Continuum}},
  author    = {Massa, Jacopo and De Caro, Valerio and Forti, Stefano and Dazzi, Patrizio and Bacciu, Davide and Brogi, Antonio},
  journal   = {Journal of Software: Evolution and Process},
  volume    = {38},
  number    = {1},
  pages     = {e70081},
  year      = {2026},
  month     = {1},
  doi       = {10.1002/smr.70081},
}
```

## Maintainers

fedclypse is maintained by [Valerio De Caro](https://github.com/vdecaro).

## Contact Us

If you want to get in touch with us, [drop us an e-mail](mailto:valerio.decaro@di.unipi.it?subject=[fedclypse]%20Request%20for%20information)!
