.. toctree::
   :maxdepth: 6
   :hidden:

   Overview <source/overview/index.rst>
   Reference <source/api/index.rst>
   Changelog <https://github.com/eclypse-org/fedclypse/blob/main/CHANGELOG.md>

=======================
fedclypse documentation
=======================

**fedclypse** is Federated Learning as a framework-agnostic verticalization of
`ECLYPSE <https://github.com/eclypse-org/eclypse>`_. It turns an eclypse
``Simulation`` into a federation: FL participants are eclypse ``Service`` entities,
the FL round is composable mechanics layered on top of eclypse's message passing,
and a run is driven and observed the same way any eclypse experiment is.

The exchange currency is plain numpy, so the base install trains real models with
nothing but numpy. The FL round factorizes into orthogonal axes you mix and match:

- **selection** -- which neighbours participate in a round;
- **synchronization** -- when a round fires and how staleness is weighted;
- **aggregation** -- how contributions combine;
- **compression** -- what crosses the wire;
- **optimization** -- how the aggregated update is applied.

.. button-ref:: source/overview/index
   :ref-type: myst
   :outline:
   :color: secondary
   :expand:
   :align: center
   :shadow:

   :octicon:`play;1em;info` Start using fedclypse
