.. toctree::
   :maxdepth: 2
   :hidden:

   Getting Started <self>

=========
fedclypse
=========

Welcome to the `fedclypse <https://github.com/eclypse-org/fedclypse>`_ documentation!

**fedclypse** is Federated Learning as a framework-agnostic verticalization of
`eclypse <https://github.com/eclypse-org/eclypse>`_. It turns an eclypse ``Simulation``
into a federation: FL participants are eclypse ``Service`` entities, the FL round is
composable mechanics layered on top of eclypse's message passing, and a run is driven
and observed the same way any eclypse experiment is.

fedclypse is a research project and is still under active development.

Why fedclypse?
==============

* **Framework-agnostic core.** The exchange currency (``fedclypse.core.Parameters``) is
  plain numpy arrays, so the base install trains real models with nothing but numpy.
  Bring your own deep-learning framework by subclassing ``fedclypse.core.Model`` and
  overriding ``Learner.local_update``.
* **Composable, orthogonal mechanics.** Selection, synchronization, aggregation,
  compression, and optimization are independent axes you mix and match rather than
  picking a monolithic "algorithm".
* **Two roles, composed freely.** ``Aggregator`` and ``Learner`` are mixins over a
  situated ``Entity``: a star server is an ``Aggregator``, a client a ``Learner``, and a
  hierarchical mid-node is both at once.
* **An experiment layer.** A ``Situation`` and a ``Behaviour`` are independent values
  joined by a single pure function, ``run(situation, behaviour, rounds=...)``.

.. _getting-started:

Getting Started
===============

fedclypse requires Python >= 3.11.

.. code-block:: console

   $ pip install -e .                     # core (numpy-only)
   $ pip install -e ".[emulation]"         # + Ray, needed to actually run a federation
   $ pip install -e ".[dev]"               # + pytest/black/ruff, for development

The canonical one-liner assembles a client-server star ``Situation`` and the reference
``FedAvg`` ``Behaviour``, then runs it:

.. code-block:: python

   import numpy as np

   from fedclypse.core import ArrayModel, Parameters
   from fedclypse.experiment import fedavg_behaviour, run, star_situation

   model_factory = lambda: ArrayModel(Parameters([np.zeros(3)]))
   situation = star_situation(
       "server", ["client_0", "client_1"], model_factory=model_factory
   )
   history = run(situation, fedavg_behaviour(), rounds=5)

For a complete, runnable demo that actually trains something — a numpy
logistic-regression classifier federated across four clients with FedAvg — see
``examples/fedavg_numpy.py`` in the repository.
