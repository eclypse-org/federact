========
Concepts
========

fedclypse models a federation as a set of situated **entities** that exchange
**contributions** and combine them with composable, orthogonal **mechanics**.

Entity
------

An :class:`~fedclypse.core.Entity` is an eclypse ``Service`` whose situatedness
(node, neighbourhood, data) and state (model) are first-class. Behaviour is
written in ``run()`` using the ``mpi`` helpers the class wraps around eclypse's
message passing.

Roles
-----

``Aggregator`` and ``Learner`` are behaviour mixins over the entity:

- a star **server** is an ``Aggregator`` (gathers contributions and aggregates);
- a **client** / leaf is a ``Learner`` (trains locally and contributes);
- a decentralized peer / hierarchical mid-node is **both** at once -- the two
  loops run concurrently on the entity's single event loop.

The orthogonal mechanics
------------------------

The FL round factorizes into independent axes, each a top-level module you mix
and match rather than picking a monolithic algorithm:

- :mod:`~fedclypse.selection` -- which neighbours participate in a round
  (``select_all``, ``uniform``, ``at_most``);
- :mod:`~fedclypse.synchronization` -- when a round fires and how staleness is
  weighted (``Synchronous``, ``BufferedAsync``, ``Asynchronous``);
- :mod:`~fedclypse.aggregation` -- how contributions combine (``fedavg``,
  ``mean``, ``weighted_sum``);
- :mod:`~fedclypse.compression` -- what crosses the wire (``identity``,
  ``topk``);
- :mod:`~fedclypse.optimization` -- how the aggregated update is applied as a
  pseudo-gradient (``ServerSGD``, ``ServerAdagrad``, ``ServerAdam``,
  ``ServerYogi``).

FedAvg is simply ``Aggregator(select_all, Synchronous, fedavg)`` with
``ServerSGD(1.0)`` and a ``Learner`` whose local update is the identity.

The experiment layer
--------------------

An experiment is a **Situation** (logical topology + physical placement + data +
models) crossed with a **Behaviour** (the role each node plays), joined by the
pure function :func:`~fedclypse.experiment.run`. A sweep is a plain ``map`` over
situations, behaviours, and seeds -- there is no experiment-manager object.
