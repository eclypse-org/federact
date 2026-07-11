"""Deployment: the logical communication topology and physical entity placement."""

from fedclypse.deployment.placement import collapse, mirror
from fedclypse.deployment.topology import complete, from_graph, ring, star

__all__ = ["collapse", "complete", "from_graph", "mirror", "ring", "star"]
