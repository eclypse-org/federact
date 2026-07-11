"""Off-the-shelf collaboration schemes: ready FL behaviours composed from mechanics."""

from fedclypse.schemes.fedavg import FedAvgClient, FedAvgServer
from fedclypse.schemes.roles import Aggregator, Learner, Roles

__all__ = ["Aggregator", "FedAvgClient", "FedAvgServer", "Learner", "Roles"]
