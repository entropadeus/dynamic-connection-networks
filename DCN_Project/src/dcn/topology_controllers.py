"""
Task-Specific Topology Controllers for Dynamic Connection Networks

This module implements sophisticated topology controllers that can dynamically
rewire connections while preserving learned weight knowledge. The controllers
manage how shared weight banks are accessed and connected for different tasks.

Key Features:
- Dynamic connection pattern generation
- Task-specific topology optimization
- Connection strength modulation
- Topology evolution strategies
- Efficient routing algorithms
- Gradient-based topology search
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set, Any, Callable
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import threading
from enum import Enum
import math
from shared_weight_banks import SharedWeightBankManager, WeightBank, WeightBankType


class TopologyType(Enum):
    """Types of topology control strategies."""
    FIXED = "fixed"
    LEARNABLE = "learnable"
    EVOLUTIONARY = "evolutionary"
    ATTENTION_BASED = "attention_based"
    GRAPH_NEURAL = "graph_neural"
    REINFORCEMENT = "reinforcement"


@dataclass
class ConnectionPattern:
    """Represents a connection pattern between layers/banks."""
    source_bank_id: str
    target_bank_id: str
    connection_matrix: torch.Tensor  # Binary or continuous connection weights
    strength_modulation: torch.Tensor  # Connection strength scaling
    routing_indices: torch.Tensor  # Efficient routing for sparse connections
    connection_type: str
    task_id: str
    is_active: bool = True
    creation_step: int = 0
    last_update_step: int = 0


class TopologyController:
    """
    Base class for topology controllers that manage dynamic connections
    between weight banks for specific tasks.
    """
    
    def __init__(self, task_id: str, controller_type: TopologyType,
                 weight_manager: SharedWeightBankManager):
        self.task_id = task_id
        self.controller_type = controller_type
        self.weight_manager = weight_manager
        
        # Connection patterns managed by this controller
        self.connection_patterns: Dict[str, ConnectionPattern] = {}
        
        # Topology evolution history
        self.topology_history = []
        self.performance_history = []
        
        # Learning parameters
        self.learning_rate = 0.01
        self.exploration_rate = 0.1
        self.temperature = 1.0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance tracking
        self.step_count = 0
        self.last_performance = 0.0
        
    def create_connection(self, source_bank_id: str, target_bank_id: str,
                         connection_type: str = "dense", 
                         sparsity: float = 0.0) -> str:
        """Create a new connection pattern between two banks."""
        with self._lock:
            connection_id = f"{source_bank_id}_to_{target_bank_id}_{connection_type}"
            
            source_bank = self.weight_manager.get_bank(source_bank_id)
            target_bank = self.weight_manager.get_bank(target_bank_id)
            
            if source_bank is None or target_bank is None:
                raise ValueError(f"One or both banks not found: {source_bank_id}, {target_bank_id}")
            
            # Create connection matrix based on bank shapes
            connection_matrix = self._generate_connection_matrix(
                source_bank.metadata.shape, target_bank.metadata.shape,
                connection_type, sparsity
            )
            
            # Initialize strength modulation
            strength_modulation = torch.ones_like(connection_matrix)
            
            # Generate routing indices for efficient sparse operations
            routing_indices = self._generate_routing_indices(connection_matrix)
            
            pattern = ConnectionPattern(
                source_bank_id=source_bank_id,
                target_bank_id=target_bank_id,
                connection_matrix=connection_matrix,
                strength_modulation=strength_modulation,
                routing_indices=routing_indices,
                connection_type=connection_type,
                task_id=self.task_id,
                creation_step=self.step_count
            )
            
            self.connection_patterns[connection_id] = pattern
            return connection_id
    
    def _generate_connection_matrix(self, source_shape: Tuple[int, ...],
                                  target_shape: Tuple[int, ...],
                                  connection_type: str, sparsity: float) -> torch.Tensor:
        """Generate connection matrix based on bank shapes and connection type."""
        if connection_type == "dense":
            if len(source_shape) == 2 and len(target_shape) == 2:
                # Linear to Linear connection
                matrix = torch.ones(target_shape[0], source_shape[1])
            elif len(source_shape) == 4 and len(target_shape) == 2:
                # Conv to Linear connection
                source_features = source_shape[0] * source_shape[2] * source_shape[3]
                matrix = torch.ones(target_shape[0], source_features)
            else:
                # Default: connect all outputs to all inputs
                source_size = np.prod(source_shape)
                target_size = np.prod(target_shape)
                matrix = torch.ones(target_size, source_size)
        
        elif connection_type == "sparse":
            matrix = self._generate_connection_matrix(source_shape, target_shape, "dense", 0.0)
            # Apply sparsity
            if sparsity > 0:
                mask = torch.rand_like(matrix) > sparsity
                matrix = matrix * mask.float()
        
        elif connection_type == "block_diagonal":
            matrix = self._generate_connection_matrix(source_shape, target_shape, "dense", 0.0)
            # Create block diagonal structure
            min_dim = min(matrix.shape)
            block_size = min_dim // 4  # 4 blocks
            matrix = torch.zeros_like(matrix)
            for i in range(0, min_dim, block_size):
                end_i = min(i + block_size, matrix.shape[0])
                end_j = min(i + block_size, matrix.shape[1])
                matrix[i:end_i, i:end_j] = 1.0
        
        elif connection_type == "attention":
            matrix = self._generate_connection_matrix(source_shape, target_shape, "dense", 0.0)
            # Initialize with attention-like pattern
            matrix = torch.softmax(torch.randn_like(matrix), dim=-1)
        
        elif connection_type == "convolutional":
            # For conv-like connections
            if len(source_shape) >= 2 and len(target_shape) >= 2:
                kernel_size = 3
                stride = 1
                padding = 1
                matrix = torch.ones(target_shape[0], source_shape[0], kernel_size, kernel_size)
            else:
                matrix = self._generate_connection_matrix(source_shape, target_shape, "dense", 0.0)
        
        else:
            # Default to dense
            matrix = self._generate_connection_matrix(source_shape, target_shape, "dense", 0.0)
        
        return matrix
    
    def _generate_routing_indices(self, connection_matrix: torch.Tensor) -> torch.Tensor:
        """Generate efficient routing indices for sparse connections."""
        # For sparse matrices, store only non-zero indices
        nonzero_indices = torch.nonzero(connection_matrix, as_tuple=False)
        return nonzero_indices
    
    def forward_connection(self, connection_id: str, source_tensor: torch.Tensor) -> torch.Tensor:
        """Apply a connection pattern to transform source tensor."""
        with self._lock:
            if connection_id not in self.connection_patterns:
                raise ValueError(f"Connection pattern {connection_id} not found")
            
            pattern = self.connection_patterns[connection_id]
            
            if not pattern.is_active:
                return source_tensor
            
            # Update last access time
            pattern.last_update_step = self.step_count
            
            # Apply connection transformation
            return self._apply_connection_transform(source_tensor, pattern)
    
    def _apply_connection_transform(self, source_tensor: torch.Tensor, 
                                  pattern: ConnectionPattern) -> torch.Tensor:
        """Apply the actual connection transformation."""
        connection_matrix = pattern.connection_matrix * pattern.strength_modulation
        
        if pattern.connection_type == "dense":
            # Standard matrix multiplication
            if source_tensor.dim() == 2:  # Batch x Features
                return F.linear(source_tensor, connection_matrix.T)
            else:
                # Reshape for matrix multiplication
                batch_size = source_tensor.shape[0]
                flattened = source_tensor.view(batch_size, -1)
                output = F.linear(flattened, connection_matrix.T)
                return output
        
        elif pattern.connection_type == "sparse":
            # Efficient sparse matrix multiplication
            indices = pattern.routing_indices
            if indices.numel() > 0:
                sparse_matrix = torch.sparse_coo_tensor(
                    indices.T, connection_matrix[indices[:, 0], indices[:, 1]],
                    connection_matrix.shape
                ).coalesce()
                
                if source_tensor.dim() == 2:
                    return torch.sparse.mm(sparse_matrix, source_tensor.T).T
                else:
                    batch_size = source_tensor.shape[0]
                    flattened = source_tensor.view(batch_size, -1)
                    return torch.sparse.mm(sparse_matrix, flattened.T).T
            else:
                return torch.zeros(source_tensor.shape[0], connection_matrix.shape[0],
                                 device=source_tensor.device, dtype=source_tensor.dtype)
        
        elif pattern.connection_type == "attention":
            # Attention-based connection
            if source_tensor.dim() == 2:
                attention_weights = F.softmax(connection_matrix, dim=-1)
                return torch.mm(attention_weights, source_tensor)
            else:
                batch_size = source_tensor.shape[0]
                flattened = source_tensor.view(batch_size, -1)
                attention_weights = F.softmax(connection_matrix, dim=-1)
                return torch.mm(attention_weights, flattened.T).T
        
        elif pattern.connection_type == "convolutional":
            # Convolutional connection
            if source_tensor.dim() == 4:  # Batch x Channels x Height x Width
                return F.conv2d(source_tensor, connection_matrix, stride=1, padding=1)
            else:
                # Fall back to dense connection
                return self._apply_connection_transform(
                    source_tensor, 
                    ConnectionPattern(
                        pattern.source_bank_id, pattern.target_bank_id,
                        connection_matrix, pattern.strength_modulation,
                        pattern.routing_indices, "dense", pattern.task_id
                    )
                )
        
        else:
            # Default to dense
            return self._apply_connection_transform(
                source_tensor,
                ConnectionPattern(
                    pattern.source_bank_id, pattern.target_bank_id,
                    connection_matrix, pattern.strength_modulation,
                    pattern.routing_indices, "dense", pattern.task_id
                )
            )
    
    def update_topology(self, performance_feedback: float, gradients: Dict[str, torch.Tensor] = None):
        """Update topology based on performance feedback."""
        with self._lock:
            self.step_count += 1
            self.performance_history.append(performance_feedback)
            self.last_performance = performance_feedback
            
            # Store current topology state
            self.topology_history.append(self._get_topology_state())
            
            # Implement topology update logic (to be overridden by subclasses)
            self._update_topology_impl(performance_feedback, gradients)
    
    def _update_topology_impl(self, performance_feedback: float, gradients: Dict[str, torch.Tensor]):
        """Implementation-specific topology update logic."""
        # Base implementation: no-op
        pass
    
    def _get_topology_state(self) -> Dict[str, Any]:
        """Get current topology state for history tracking."""
        state = {}
        for conn_id, pattern in self.connection_patterns.items():
            state[conn_id] = {
                'connection_matrix': pattern.connection_matrix.clone(),
                'strength_modulation': pattern.strength_modulation.clone(),
                'is_active': pattern.is_active,
                'step': self.step_count
            }
        return state
    
    def prune_connections(self, threshold: float = 0.1):
        """Prune weak connections based on strength threshold."""
        with self._lock:
            for pattern in self.connection_patterns.values():
                # Identify weak connections
                weak_mask = pattern.strength_modulation < threshold
                pattern.connection_matrix[weak_mask] = 0.0
                
                # Update routing indices
                pattern.routing_indices = self._generate_routing_indices(pattern.connection_matrix)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about current connections."""
        with self._lock:
            stats = {
                'total_connections': len(self.connection_patterns),
                'active_connections': sum(1 for p in self.connection_patterns.values() if p.is_active),
                'connection_types': defaultdict(int),
                'sparsity_levels': {},
                'strength_stats': {},
                'performance_trend': self._calculate_performance_trend()
            }
            
            for conn_id, pattern in self.connection_patterns.items():
                stats['connection_types'][pattern.connection_type] += 1
                
                # Calculate sparsity
                total_connections = pattern.connection_matrix.numel()
                active_connections = (pattern.connection_matrix != 0).sum().item()
                sparsity = 1.0 - (active_connections / total_connections)
                stats['sparsity_levels'][conn_id] = sparsity
                
                # Strength statistics
                strength = pattern.strength_modulation
                stats['strength_stats'][conn_id] = {
                    'mean': strength.mean().item(),
                    'std': strength.std().item(),
                    'min': strength.min().item(),
                    'max': strength.max().item()
                }
            
            return stats
    
    def _calculate_performance_trend(self) -> float:
        """Calculate performance trend over recent history."""
        if len(self.performance_history) < 10:
            return 0.0
        
        recent_performance = self.performance_history[-10:]
        # Simple linear trend calculation
        x = torch.arange(len(recent_performance), dtype=torch.float)
        y = torch.tensor(recent_performance, dtype=torch.float)
        
        # Calculate slope
        x_mean = x.mean()
        y_mean = y.mean()
        
        numerator = ((x - x_mean) * (y - y_mean)).sum()
        denominator = ((x - x_mean) ** 2).sum()
        
        if denominator > 0:
            slope = numerator / denominator
            return slope.item()
        else:
            return 0.0


class LearnableTopologyController(TopologyController):
    """
    Topology controller that learns connection patterns through gradient descent.
    """
    
    def __init__(self, task_id: str, weight_manager: SharedWeightBankManager,
                 learning_rate: float = 0.01, temperature: float = 1.0):
        super().__init__(task_id, TopologyType.LEARNABLE, weight_manager)
        self.learning_rate = learning_rate
        self.temperature = temperature
        
        # Learnable parameters for topology
        self.topology_parameters = nn.ParameterDict()
        
    def create_learnable_connection(self, source_bank_id: str, target_bank_id: str,
                                  connection_type: str = "dense", 
                                  initial_sparsity: float = 0.0) -> str:
        """Create a learnable connection with gradient-optimizable parameters."""
        connection_id = self.create_connection(source_bank_id, target_bank_id, 
                                             connection_type, initial_sparsity)
        
        pattern = self.connection_patterns[connection_id]
        
        # Make connection matrix learnable
        connection_param = nn.Parameter(pattern.connection_matrix.clone())
        strength_param = nn.Parameter(pattern.strength_modulation.clone())
        
        self.topology_parameters[f"{connection_id}_connection"] = connection_param
        self.topology_parameters[f"{connection_id}_strength"] = strength_param
        
        return connection_id
    
    def _update_topology_impl(self, performance_feedback: float, gradients: Dict[str, torch.Tensor]):
        """Update learnable topology parameters using gradients."""
        if gradients is None:
            return
        
        # Update topology parameters based on gradients
        for param_name, param in self.topology_parameters.items():
            if param_name in gradients:
                grad = gradients[param_name]
                # Apply gradient update
                param.data -= self.learning_rate * grad
                
                # Apply constraints (e.g., keep connections positive)
                if "connection" in param_name:
                    param.data = torch.clamp(param.data, 0.0, 1.0)
                elif "strength" in param_name:
                    param.data = torch.clamp(param.data, 0.0, 2.0)
        
        # Update connection patterns with new parameters
        self._sync_parameters_to_patterns()
    
    def _sync_parameters_to_patterns(self):
        """Synchronize learnable parameters back to connection patterns."""
        for conn_id, pattern in self.connection_patterns.items():
            connection_param_name = f"{conn_id}_connection"
            strength_param_name = f"{conn_id}_strength"
            
            if connection_param_name in self.topology_parameters:
                pattern.connection_matrix = self.topology_parameters[connection_param_name]
            
            if strength_param_name in self.topology_parameters:
                pattern.strength_modulation = self.topology_parameters[strength_param_name]
            
            # Update routing indices if connection matrix changed
            pattern.routing_indices = self._generate_routing_indices(pattern.connection_matrix)
    
    def apply_gumbel_softmax(self, connection_id: str, hard: bool = False):
        """Apply Gumbel-Softmax to make discrete connection decisions."""
        if connection_id not in self.connection_patterns:
            return
        
        pattern = self.connection_patterns[connection_id]
        connection_param_name = f"{connection_id}_connection"
        
        if connection_param_name in self.topology_parameters:
            logits = self.topology_parameters[connection_param_name]
            
            # Apply Gumbel-Softmax
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
            softmax_input = (logits + gumbel_noise) / self.temperature
            soft_connections = F.softmax(softmax_input, dim=-1)
            
            if hard:
                # Hard discretization
                hard_connections = torch.zeros_like(soft_connections)
                max_indices = soft_connections.argmax(dim=-1, keepdim=True)
                hard_connections.scatter_(-1, max_indices, 1.0)
                
                # Straight-through estimator
                pattern.connection_matrix = hard_connections - soft_connections.detach() + soft_connections
            else:
                pattern.connection_matrix = soft_connections


class EvolutionaryTopologyController(TopologyController):
    """
    Topology controller that evolves connection patterns using evolutionary algorithms.
    """
    
    def __init__(self, task_id: str, weight_manager: SharedWeightBankManager,
                 population_size: int = 20, mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7):
        super().__init__(task_id, TopologyType.EVOLUTIONARY, weight_manager)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Population of topology configurations
        self.population = []
        self.fitness_scores = []
        self.generation = 0
    
    def initialize_population(self):
        """Initialize population with random topology configurations."""
        self.population = []
        self.fitness_scores = []
        
        for _ in range(self.population_size):
            individual = {}
            for conn_id, pattern in self.connection_patterns.items():
                # Create random variations of connection patterns
                connection_matrix = pattern.connection_matrix.clone()
                strength_modulation = pattern.strength_modulation.clone()
                
                # Add random mutations
                if torch.rand(1) < self.mutation_rate:
                    connection_matrix += torch.randn_like(connection_matrix) * 0.1
                    connection_matrix = torch.clamp(connection_matrix, 0.0, 1.0)
                
                if torch.rand(1) < self.mutation_rate:
                    strength_modulation += torch.randn_like(strength_modulation) * 0.1
                    strength_modulation = torch.clamp(strength_modulation, 0.0, 2.0)
                
                individual[conn_id] = {
                    'connection_matrix': connection_matrix,
                    'strength_modulation': strength_modulation
                }
            
            self.population.append(individual)
            self.fitness_scores.append(0.0)
    
    def _update_topology_impl(self, performance_feedback: float, gradients: Dict[str, torch.Tensor]):
        """Evolve topology using evolutionary algorithm."""
        if not self.population:
            self.initialize_population()
            return
        
        # Update fitness score for current individual
        current_individual_idx = self.generation % self.population_size
        self.fitness_scores[current_individual_idx] = performance_feedback
        
        # Every generation, evolve the population
        if (self.step_count + 1) % self.population_size == 0:
            self._evolve_population()
            self.generation += 1
        
        # Apply next individual from population
        next_individual_idx = (self.generation + 1) % self.population_size
        self._apply_individual(self.population[next_individual_idx])
    
    def _evolve_population(self):
        """Evolve the population using selection, crossover, and mutation."""
        # Selection: tournament selection
        new_population = []
        new_fitness_scores = []
        
        for _ in range(self.population_size):
            # Tournament selection
            tournament_size = 3
            tournament_indices = torch.randint(0, self.population_size, (tournament_size,))
            tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            
            parent1 = self.population[winner_idx]
            
            # Crossover
            if torch.rand(1) < self.crossover_rate:
                # Select second parent
                tournament_indices = torch.randint(0, self.population_size, (tournament_size,))
                tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                parent2 = self.population[winner_idx]
                
                offspring = self._crossover(parent1, parent2)
            else:
                offspring = self._copy_individual(parent1)
            
            # Mutation
            if torch.rand(1) < self.mutation_rate:
                offspring = self._mutate(offspring)
            
            new_population.append(offspring)
            new_fitness_scores.append(0.0)
        
        self.population = new_population
        self.fitness_scores = new_fitness_scores
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Create offspring through crossover of two parents."""
        offspring = {}
        
        for conn_id in parent1.keys():
            if torch.rand(1) < 0.5:
                # Take from parent1
                offspring[conn_id] = {
                    'connection_matrix': parent1[conn_id]['connection_matrix'].clone(),
                    'strength_modulation': parent1[conn_id]['strength_modulation'].clone()
                }
            else:
                # Take from parent2
                offspring[conn_id] = {
                    'connection_matrix': parent2[conn_id]['connection_matrix'].clone(),
                    'strength_modulation': parent2[conn_id]['strength_modulation'].clone()
                }
        
        return offspring
    
    def _mutate(self, individual: Dict) -> Dict:
        """Apply mutations to an individual."""
        for conn_id in individual.keys():
            # Mutate connection matrix
            if torch.rand(1) < 0.3:
                noise = torch.randn_like(individual[conn_id]['connection_matrix']) * 0.05
                individual[conn_id]['connection_matrix'] += noise
                individual[conn_id]['connection_matrix'] = torch.clamp(
                    individual[conn_id]['connection_matrix'], 0.0, 1.0
                )
            
            # Mutate strength modulation
            if torch.rand(1) < 0.3:
                noise = torch.randn_like(individual[conn_id]['strength_modulation']) * 0.05
                individual[conn_id]['strength_modulation'] += noise
                individual[conn_id]['strength_modulation'] = torch.clamp(
                    individual[conn_id]['strength_modulation'], 0.0, 2.0
                )
        
        return individual
    
    def _copy_individual(self, individual: Dict) -> Dict:
        """Create a deep copy of an individual."""
        copy = {}
        for conn_id, data in individual.items():
            copy[conn_id] = {
                'connection_matrix': data['connection_matrix'].clone(),
                'strength_modulation': data['strength_modulation'].clone()
            }
        return copy
    
    def _apply_individual(self, individual: Dict):
        """Apply an individual's configuration to current connection patterns."""
        for conn_id, data in individual.items():
            if conn_id in self.connection_patterns:
                pattern = self.connection_patterns[conn_id]
                pattern.connection_matrix = data['connection_matrix']
                pattern.strength_modulation = data['strength_modulation']
                pattern.routing_indices = self._generate_routing_indices(pattern.connection_matrix)


class AttentionTopologyController(TopologyController):
    """
    Topology controller that uses attention mechanisms to determine connections.
    """
    
    def __init__(self, task_id: str, weight_manager: SharedWeightBankManager,
                 attention_dim: int = 64, num_heads: int = 8):
        super().__init__(task_id, TopologyType.ATTENTION_BASED, weight_manager)
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        
        # Attention networks for different bank types
        self.attention_networks = nn.ModuleDict()
        
    def create_attention_connection(self, source_bank_id: str, target_bank_id: str) -> str:
        """Create an attention-based connection between banks."""
        connection_id = self.create_connection(source_bank_id, target_bank_id, "attention")
        
        # Create attention network for this connection
        source_bank = self.weight_manager.get_bank(source_bank_id)
        target_bank = self.weight_manager.get_bank(target_bank_id)
        
        source_features = np.prod(source_bank.metadata.shape)
        target_features = np.prod(target_bank.metadata.shape)
        
        attention_net = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=self.num_heads,
            batch_first=True
        )
        
        # Projection layers
        source_proj = nn.Linear(source_features, self.attention_dim)
        target_proj = nn.Linear(target_features, self.attention_dim)
        output_proj = nn.Linear(self.attention_dim, target_features)
        
        self.attention_networks[connection_id] = nn.ModuleDict({
            'attention': attention_net,
            'source_proj': source_proj,
            'target_proj': target_proj,
            'output_proj': output_proj
        })
        
        return connection_id
    
    def _apply_connection_transform(self, source_tensor: torch.Tensor,
                                  pattern: ConnectionPattern) -> torch.Tensor:
        """Apply attention-based connection transformation."""
        connection_id = f"{pattern.source_bank_id}_to_{pattern.target_bank_id}_{pattern.connection_type}"
        
        if connection_id in self.attention_networks and pattern.connection_type == "attention":
            networks = self.attention_networks[connection_id]
            
            # Flatten source tensor for attention
            batch_size = source_tensor.shape[0]
            source_flat = source_tensor.view(batch_size, -1)
            
            # Project to attention dimension
            source_proj = networks['source_proj'](source_flat)
            
            # Self-attention (using source as both query and key)
            attended, attention_weights = networks['attention'](
                source_proj.unsqueeze(1),  # Add sequence dimension
                source_proj.unsqueeze(1),
                source_proj.unsqueeze(1)
            )
            
            # Remove sequence dimension and project to output
            attended = attended.squeeze(1)
            output = networks['output_proj'](attended)
            
            return output
        else:
            # Fall back to parent implementation
            return super()._apply_connection_transform(source_tensor, pattern)
    
    def _update_topology_impl(self, performance_feedback: float, gradients: Dict[str, torch.Tensor]):
        """Update attention weights based on performance feedback."""
        # Attention networks are updated through standard backpropagation
        # This method can implement additional attention-specific updates
        
        # Adjust attention temperature based on performance
        if hasattr(self, 'last_performance') and self.last_performance > 0:
            if performance_feedback > self.last_performance:
                # Good performance: reduce temperature (more focused attention)
                self.temperature = max(0.1, self.temperature * 0.99)
            else:
                # Poor performance: increase temperature (more exploration)
                self.temperature = min(2.0, self.temperature * 1.01)


# Example usage and testing
if __name__ == "__main__":
    from shared_weight_banks import SharedWeightBankManager, WeightBankType
    
    # Create weight manager
    manager = SharedWeightBankManager(memory_limit_mb=512)
    
    # Create some weight banks
    bank1 = manager.create_bank("encoder", WeightBankType.LINEAR, (256, 128))
    bank2 = manager.create_bank("decoder", WeightBankType.LINEAR, (128, 64))
    bank3 = manager.create_bank("classifier", WeightBankType.LINEAR, (64, 10))
    
    # Create different types of topology controllers
    learnable_controller = LearnableTopologyController("task_classification", manager)
    evolutionary_controller = EvolutionaryTopologyController("task_generation", manager)
    attention_controller = AttentionTopologyController("task_attention", manager)
    
    # Create connections
    conn1 = learnable_controller.create_learnable_connection("encoder", "decoder", "dense")
    conn2 = learnable_controller.create_learnable_connection("decoder", "classifier", "sparse", 0.3)
    
    conn3 = evolutionary_controller.create_connection("encoder", "decoder", "block_diagonal")
    evolutionary_controller.initialize_population()
    
    conn4 = attention_controller.create_attention_connection("encoder", "classifier")
    
    # Test forward passes
    batch_size = 32
    input_features = 128
    test_input = torch.randn(batch_size, input_features)
    
    # Test learnable controller
    encoder_weight = bank1.get_weight("task_classification")
    encoder_output = F.linear(test_input, encoder_weight)
    decoder_output = learnable_controller.forward_connection(conn1, encoder_output)
    
    print("Topology controllers initialized successfully!")
    print(f"Learnable controller stats: {learnable_controller.get_connection_stats()}")
    print(f"Evolutionary controller stats: {evolutionary_controller.get_connection_stats()}")
    print(f"Attention controller stats: {attention_controller.get_connection_stats()}")