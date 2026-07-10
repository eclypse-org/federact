fedclypse
=========

**fedclypse** is Federated Learning as a framework-agnostic verticalization of
`eclypse <https://github.com/eclypse-org/eclypse>`_. It turns an eclypse
``Simulation`` into a federation: FL participants are eclypse ``Service``
entities, the FL round is composable mechanics layered on top of eclypse's
message passing, and a run is driven and observed the same way any eclypse
experiment is.

fedclypse is a research project and is still under active development.

What it is
----------

* **Framework-agnostic core.** The exchange currency (``fedclypse.core.Parameters``)
  is plain numpy arrays, so the base install trains real models with nothing but
  numpy. Bring your own deep-learning framework by subclassing
  ``fedclypse.core.Model`` (``get_parameters``/``set_parameters``) and overriding
  ``Learner.local_update`` — the rest of the stack never touches a framework object.
* **Composable, orthogonal mechanics.** Selection (who participates),
  synchronization (when a round fires and how staleness is weighted), aggregation
  (how contributions combine), compression (what crosses the wire), and
  optimization (how the aggregated update is applied) are independent axes — mix
  and match rather than picking a monolithic "algorithm".
* **Two roles, composed freely.** ``Aggregator`` and ``Learner`` are mixins over a
  situated ``Entity``: a star server is an ``Aggregator``, a client a ``Learner``,
  and a hierarchical mid-node is both at once.
* **An experiment layer.** A ``Situation`` (who the nodes are, their data, models,
  and physical layout) and a ``Behaviour`` (how each role acts) are independent
  values joined by a single pure function, ``run(situation, behaviour, rounds=...)``
  — no experiment-manager object, no hidden state.
* **Runs on eclypse.** Simulation (placement/timing only) or emulation (Ray-backed,
  real execution) are the same code path, selected by one flag.

Installation
------------

fedclypse requires Python >= 3.11.

.. code-block:: console

    $ pip install -e .                    # core (numpy-only)
    $ pip install -e ".[emulation]"        # + Ray, needed to actually run a federation
    $ pip install -e ".[dev]"              # + pytest/black/ruff, for development

Quickstart
----------

The canonical one-liner assembles a client-server star ``Situation`` and the
reference ``FedAvg`` ``Behaviour``, then runs it:

.. code-block:: python

    import numpy as np

    from fedclypse.core import ArrayModel, Parameters
    from fedclypse.experiment import fedavg_behaviour, run, star_situation

    model_factory = lambda: ArrayModel(Parameters([np.zeros(3)]))
    situation = star_situation(
        "server", ["client_0", "client_1"], model_factory=model_factory
    )
    history = run(situation, fedavg_behaviour(), rounds=5)

``history`` is a ``History`` over the collected metric samples
(``history.series(...)`` / ``history.final(...)``). For a complete, runnable
demo that actually trains something — a numpy logistic-regression classifier
federated across four clients with FedAvg, test accuracy printed each round —
see `examples/fedavg_numpy.py <examples/fedavg_numpy.py>`_:

.. code-block:: console

    $ python examples/fedavg_numpy.py

Package map
-----------

* ``fedclypse.core`` — ``Entity``, ``Parameters``, ``Contribution``,
  ``Model``/``ArrayModel``: the foundational types every other layer builds on.
* ``fedclypse.data`` — ``DataSource``/``ClientData``/``InMemorySource``/``split``
  and the ``Partitioner`` family (``IID``, ``Dirichlet``, ``Pathological``,
  ``QuantitySkew``, ``NaturalId``).
* Mechanics, one module each at the package root — ``aggregation``
  (``fedavg``, ``mean``, ``weighted_sum``), ``selection`` (``select_all``,
  ``uniform``, ``at_most``), ``synchronization`` (``Synchronous``,
  ``BufferedAsync``, ``Asynchronous``), ``compression`` (``identity``,
  ``topk``), ``optimization`` (``ServerSGD``, ``ServerAdagrad``,
  ``ServerAdam``, ``ServerYogi``): five orthogonal axes you mix and match.
* ``fedclypse.schemes`` — ``Aggregator``/``Learner`` roles and the
  off-the-shelf ``FedAvgServer``/``FedAvgClient``.
* ``fedclypse.deployment`` — logical topology (``star``, ``ring``,
  ``complete``, ``from_graph``) and physical placement (``mirror``,
  ``collapse``).
* ``fedclypse.runtime`` — ``build_simulation``/``run_federation`` to assemble
  and drive an eclypse ``Simulation``, plus ``History``/``round_metric`` to
  observe it.
* ``fedclypse.experiment`` — ``Situation`` x ``Behaviour`` x ``run``: the
  composition layer that joins a logical/physical layout with role behaviour
  into a runnable study.

Status
------

fedclypse is early-stage research software (v0.1.0); the API may still change.

License
-------

fedclypse is released under the MIT license.
