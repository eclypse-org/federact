===============
Getting started
===============

The experiment layer joins a **Situation** (who the nodes are, their data,
models, and physical layout) and a **Behaviour** (how each role acts) into a
runnable study via a single pure function, ``run``. The canonical one-liner
assembles a client-server star and the reference FedAvg behaviour, then runs it:

.. code-block:: python

   import numpy as np

   from fedclypse.core import ArrayModel, Parameters
   from fedclypse.experiment import fedavg_behaviour, run, star_situation

   model_factory = lambda: ArrayModel(Parameters([np.zeros(3)]))
   situation = star_situation(
       "server", ["client_0", "client_1"], model_factory=model_factory
   )
   history = run(situation, fedavg_behaviour(), rounds=5)

``history`` is a :class:`~fedclypse.runtime.History` over the collected metric
samples (``history.series(...)`` / ``history.final(...)``).

A complete, runnable demo -- a numpy logistic-regression classifier federated
across four clients with FedAvg, no deep-learning framework and no dataset
download -- lives in ``examples/fedavg_numpy.py`` in the repository:

.. code-block:: shell

   python examples/fedavg_numpy.py

Once you are comfortable with the workflow, continue with the
:doc:`concepts <../concepts>` to customise selection, synchronization,
aggregation, and server optimization.
