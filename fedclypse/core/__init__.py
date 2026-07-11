"""Foundational abstractions: the situated entity, the exchange currency, and the model.

These are the base types the rest of fedclypse builds on — ``Entity`` (the situated FL
participant), ``Parameters``/``Contribution`` (the currency of exchange), and ``Model``
(the entity's state face).
"""

from fedclypse.core.contribution import Contribution, parameters, pseudogradient
from fedclypse.core.entity import Entity
from fedclypse.core.model import ArrayModel, Model
from fedclypse.core.parameters import Parameters

__all__ = [
    "ArrayModel",
    "Contribution",
    "Entity",
    "Model",
    "Parameters",
    "parameters",
    "pseudogradient",
]
