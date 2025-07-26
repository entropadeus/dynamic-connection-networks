"""
Dynamic Connection Networks (DCN)

A neural network architecture that can dynamically adapt its topology
during training and inference, not just its weights.

Co-created by Ben Lona and Claude (Anthropic)
"""

from .dynamic_connection_network import DynamicConnectionNetwork, DynamicConnectionLayer
from .shared_weight_banks import SharedWeightBank
from .topology_controllers import AdaptiveTopologyController
from .training_strategies import MetaLearningTrainer, MultiTaskTrainer
from .memory_optimization import MemoryOptimizer
from .incremental_learning import IncrementalLearner

__version__ = "0.1.0"
__author__ = "Ben Lona and Claude (Anthropic)"

__all__ = [
    "DynamicConnectionNetwork",
    "DynamicConnectionLayer", 
    "SharedWeightBank",
    "AdaptiveTopologyController",
    "MetaLearningTrainer",
    "MultiTaskTrainer",
    "MemoryOptimizer",
    "IncrementalLearner"
]