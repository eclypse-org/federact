# -*- coding: utf-8 -*-
"""Off-the-shelf collaboration schemes: ready FL behaviours composed from the mechanics."""
from fedclypse.schemes.fedavg import FedAvgClient, FedAvgServer
from fedclypse.schemes.roles import Aggregator, Learner, Roles

__all__ = ["FedAvgServer", "FedAvgClient", "Aggregator", "Learner", "Roles"]
