"""
Incremental Learning Architecture for Dynamic Connection Networks

This module implements sophisticated incremental learning capabilities that enable
the network to learn new tasks without catastrophic forgetting. It combines
topology adaptation with weight preservation strategies to maintain knowledge
across multiple tasks.

Key Features:
- Elastic Weight Consolidation (EWC) for important weight preservation
- Progressive Neural Networks architecture
- Task-specific topology isolation
- Knowledge distillation between tasks
- Memory replay mechanisms
- Adaptive capacity expansion
- Gradient orthogonalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set, Any, Callable
import numpy as np
from dataclasses import dataclass
from collections import defaultdict, deque
import threading
from enum import Enum
import math
import copy
from shared_weight_banks import SharedWeightBankManager, WeightBank, WeightBankType
from topology_controllers import TopologyController, LearnableTopologyController


class ForgettingPreventionStrategy(Enum):
    """Strategies for preventing catastrophic forgetting."""
    EWC = "elastic_weight_consolidation"
    PROGRESSIVE = "progressive_networks"
    PACKNET = "packnet"
    HAT = "hard_attention_tasks"
    REPLAY = "memory_replay"
    DISTILLATION = "knowledge_distillation"
    ORTHOGONAL = "gradient_orthogonalization"
    MIXED = "mixed_strategies"


@dataclass
class TaskMemory:
    """Memory storage for a specific task."""
    task_id: str
    data_samples: List[torch.Tensor]
    labels: List[torch.Tensor]
    task_features: torch.Tensor
    importance_weights: Dict[str, torch.Tensor]
    performance_history: List[float]
    topology_snapshot: Dict[str, Any]
    creation_time: float
    last_access_time: float
    sample_count: int
    max_samples: int = 1000


class ImportanceCalculator:
    """
    Calculate importance weights for parameters using various strategies.
    """
    
    def __init__(self, strategy: str = "fisher"):
        self.strategy = strategy
        self.importance_cache = {}
    
    def calculate_fisher_information(self, model_params: Dict[str, torch.Tensor],
                                   data_loader, loss_fn, num_samples: int = 1000) -> Dict[str, torch.Tensor]:
        """Calculate Fisher Information Matrix diagonal for EWC."""
        fisher_dict = {}
        
        # Initialize Fisher information
        for name, param in model_params.items():
            fisher_dict[name] = torch.zeros_like(param)
        
        model_in_eval = True  # Assume model is in eval mode
        sample_count = 0
        
        for batch_idx, (data, target) in enumerate(data_loader):
            if sample_count >= num_samples:
                break
            
            # Forward pass
            output = self._forward_with_params(data, model_params)
            loss = loss_fn(output, target)
            
            # Backward pass to get gradients
            gradients = torch.autograd.grad(loss, model_params.values(), create_graph=False)
            
            # Accumulate Fisher information (squared gradients)
            for (name, param), grad in zip(model_params.items(), gradients):
                if grad is not None:
                    fisher_dict[name] += grad.pow(2)
            
            sample_count += data.shape[0]
        
        # Normalize by number of samples
        for name in fisher_dict:
            fisher_dict[name] /= sample_count
        
        return fisher_dict
    
    def calculate_path_integral(self, model_params: Dict[str, torch.Tensor],
                              data_loader, loss_fn) -> Dict[str, torch.Tensor]:
        """Calculate path integral importance for Synaptic Intelligence."""
        importance_dict = {}
        
        # Initialize importance weights
        for name, param in model_params.items():
            importance_dict[name] = torch.zeros_like(param)
        
        # This is a simplified version - full implementation would track
        # parameter changes over the entire learning trajectory
        for name, param in model_params.items():
            # Use gradient magnitude as proxy for importance
            if param.grad is not None:
                importance_dict[name] = param.grad.abs()
        
        return importance_dict
    
    def calculate_gradient_magnitude(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Calculate importance based on gradient magnitudes."""
        importance_dict = {}
        
        for name, grad in gradients.items():
            if grad is not None:
                importance_dict[name] = grad.abs()
            else:
                # If no gradient, assume zero importance
                importance_dict[name] = torch.zeros_like(grad) if grad is not None else torch.tensor(0.0)
        
        return importance_dict
    
    def _forward_with_params(self, data: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with specific parameters (simplified)."""
        # This is a placeholder - in practice, you'd need to implement
        # functional forward pass with the given parameters
        return torch.randn(data.shape[0], 10)  # Dummy output


class EWCRegularizer:
    """
    Elastic Weight Consolidation regularizer to prevent catastrophic forgetting.
    """
    
    def __init__(self, lambda_ewc: float = 1000.0):
        self.lambda_ewc = lambda_ewc
        self.task_fisher_info = {}
        self.task_optimal_params = {}
        self.importance_calculator = ImportanceCalculator()
    
    def compute_importance(self, task_id: str, model_params: Dict[str, torch.Tensor],
                          data_loader, loss_fn):
        """Compute and store importance weights for a task."""
        fisher_info = self.importance_calculator.calculate_fisher_information(
            model_params, data_loader, loss_fn
        )
        
        self.task_fisher_info[task_id] = fisher_info
        self.task_optimal_params[task_id] = {
            name: param.clone().detach() for name, param in model_params.items()
        }
    
    def compute_penalty(self, current_params: Dict[str, torch.Tensor],
                       exclude_tasks: Set[str] = None) -> torch.Tensor:
        """Compute EWC penalty for current parameters."""
        penalty = torch.tensor(0.0, device=next(iter(current_params.values())).device)
        
        exclude_tasks = exclude_tasks or set()
        
        for task_id in self.task_fisher_info:
            if task_id in exclude_tasks:
                continue
            
            fisher_info = self.task_fisher_info[task_id]
            optimal_params = self.task_optimal_params[task_id]
            
            for name in current_params:
                if name in fisher_info and name in optimal_params:
                    # EWC penalty: F * (θ - θ*)^2
                    param_penalty = fisher_info[name] * (
                        current_params[name] - optimal_params[name]
                    ).pow(2)
                    penalty += param_penalty.sum()
        
        return self.lambda_ewc * penalty / 2.0


class ProgressiveNetwork:
    """
    Progressive Neural Network architecture for incremental learning.
    """
    
    def __init__(self, weight_manager: SharedWeightBankManager):
        self.weight_manager = weight_manager
        self.task_columns = {}  # Task-specific network columns
        self.lateral_connections = {}  # Connections between columns
        self.column_count = 0
    
    def add_task_column(self, task_id: str, layer_configs: List[Dict[str, Any]]) -> List[str]:
        """Add a new column for a specific task."""
        column_banks = []
        
        for i, config in enumerate(layer_configs):
            bank_id = f"{task_id}_column_{self.column_count}_layer_{i}"
            
            bank = self.weight_manager.create_bank(
                bank_id=bank_id,
                bank_type=WeightBankType(config['type']),
                shape=tuple(config['shape']),
                initialization=config.get('initialization', 'xavier_uniform')
            )
            
            column_banks.append(bank_id)
        
        self.task_columns[task_id] = {
            'banks': column_banks,
            'column_id': self.column_count,
            'frozen': False
        }
        
        # Create lateral connections to previous columns
        if self.column_count > 0:
            self._create_lateral_connections(task_id)
        
        self.column_count += 1
        return column_banks
    
    def _create_lateral_connections(self, new_task_id: str):
        """Create lateral connections from previous columns to new column."""
        new_column = self.task_columns[new_task_id]
        lateral_key = f"lateral_to_{new_task_id}"
        self.lateral_connections[lateral_key] = {}
        
        # Connect each layer of previous columns to corresponding layer of new column
        for prev_task_id, prev_column in self.task_columns.items():
            if prev_task_id == new_task_id:
                continue
            
            prev_banks = prev_column['banks']
            new_banks = new_column['banks']
            
            # Create connections between corresponding layers
            min_layers = min(len(prev_banks), len(new_banks))
            
            for layer_idx in range(min_layers):
                prev_bank_id = prev_banks[layer_idx]
                new_bank_id = new_banks[layer_idx]
                
                # Create lateral connection matrix
                prev_bank = self.weight_manager.get_bank(prev_bank_id)
                new_bank = self.weight_manager.get_bank(new_bank_id)
                
                if prev_bank and new_bank:
                    # Lateral connection: typically smaller than main connections
                    lateral_shape = (new_bank.metadata.shape[0], prev_bank.metadata.shape[0])
                    lateral_bank_id = f"lateral_{prev_task_id}_to_{new_task_id}_layer_{layer_idx}"
                    
                    lateral_bank = self.weight_manager.create_bank(
                        bank_id=lateral_bank_id,
                        bank_type=WeightBankType.LINEAR,
                        shape=lateral_shape,
                        initialization='xavier_uniform'
                    )
                    
                    self.lateral_connections[lateral_key][f"{prev_task_id}_layer_{layer_idx}"] = lateral_bank_id
    
    def freeze_column(self, task_id: str):
        """Freeze a task column to prevent further updates."""
        if task_id in self.task_columns:
            self.task_columns[task_id]['frozen'] = True
            
            # Freeze all banks in the column
            for bank_id in self.task_columns[task_id]['banks']:
                bank = self.weight_manager.get_bank(bank_id)
                if bank:
                    bank.freeze()
    
    def forward_progressive(self, task_id: str, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass through progressive network for a specific task."""
        if task_id not in self.task_columns:
            raise ValueError(f"Task {task_id} not found in progressive network")
        
        column = self.task_columns[task_id]
        column_banks = column['banks']
        
        # Forward through the main column
        activations = []
        current_input = inputs[0] if inputs else torch.randn(1, 100)  # Default input
        
        for layer_idx, bank_id in enumerate(column_banks):
            bank = self.weight_manager.get_bank(bank_id)
            if bank:
                # Main column computation
                weight = bank.get_weight(task_id)
                layer_output = F.linear(current_input, weight)
                
                # Add lateral connections from previous columns
                lateral_input = self._compute_lateral_input(task_id, layer_idx, activations)
                if lateral_input is not None:
                    layer_output = layer_output + lateral_input
                
                # Apply activation function
                layer_output = F.relu(layer_output)
                activations.append(layer_output)
                current_input = layer_output
        
        return activations
    
    def _compute_lateral_input(self, task_id: str, layer_idx: int, 
                              activations: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """Compute input from lateral connections."""
        lateral_key = f"lateral_to_{task_id}"
        
        if lateral_key not in self.lateral_connections:
            return None
        
        lateral_sum = None
        
        # Collect inputs from all previous columns at this layer
        for prev_task_id, prev_column in self.task_columns.items():
            if prev_task_id == task_id:
                continue
            
            connection_key = f"{prev_task_id}_layer_{layer_idx}"
            if connection_key in self.lateral_connections[lateral_key]:
                lateral_bank_id = self.lateral_connections[lateral_key][connection_key]
                lateral_bank = self.weight_manager.get_bank(lateral_bank_id)
                
                if lateral_bank and layer_idx < len(activations):
                    # Get activation from previous column at same layer
                    prev_activation = activations[layer_idx]  # This would need proper tracking
                    lateral_weight = lateral_bank.get_weight(task_id)
                    lateral_output = F.linear(prev_activation, lateral_weight)
                    
                    if lateral_sum is None:
                        lateral_sum = lateral_output
                    else:
                        lateral_sum = lateral_sum + lateral_output
        
        return lateral_sum


class MemoryReplayBuffer:
    """
    Memory buffer for storing and replaying samples from previous tasks.
    """
    
    def __init__(self, max_size_per_task: int = 1000, selection_strategy: str = "random"):
        self.max_size_per_task = max_size_per_task
        self.selection_strategy = selection_strategy
        self.task_memories: Dict[str, TaskMemory] = {}
        self._lock = threading.RLock()
    
    def add_task_samples(self, task_id: str, data_samples: List[torch.Tensor],
                        labels: List[torch.Tensor]):
        """Add samples for a specific task to the replay buffer."""
        with self._lock:
            if task_id not in self.task_memories:
                self.task_memories[task_id] = TaskMemory(
                    task_id=task_id,
                    data_samples=[],
                    labels=[],
                    task_features=torch.zeros(1),  # Placeholder
                    importance_weights={},
                    performance_history=[],
                    topology_snapshot={},
                    creation_time=torch.cuda.Event().record() if torch.cuda.is_available() else 0,
                    last_access_time=0,
                    sample_count=0,
                    max_samples=self.max_size_per_task
                )
            
            memory = self.task_memories[task_id]
            
            # Add samples using selection strategy
            for data, label in zip(data_samples, labels):
                if len(memory.data_samples) < self.max_size_per_task:
                    memory.data_samples.append(data.clone())
                    memory.labels.append(label.clone())
                    memory.sample_count += 1
                else:
                    # Replace existing samples based on strategy
                    if self.selection_strategy == "random":
                        replace_idx = torch.randint(0, len(memory.data_samples), (1,)).item()
                        memory.data_samples[replace_idx] = data.clone()
                        memory.labels[replace_idx] = label.clone()
                    elif self.selection_strategy == "importance":
                        # Replace least important sample (simplified)
                        replace_idx = torch.randint(0, len(memory.data_samples), (1,)).item()
                        memory.data_samples[replace_idx] = data.clone()
                        memory.labels[replace_idx] = label.clone()
    
    def sample_replay_batch(self, task_ids: List[str], batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Sample a batch of replay data from specified tasks."""
        with self._lock:
            batch_data = []
            batch_labels = []
            batch_task_ids = []
            
            samples_per_task = batch_size // len(task_ids)
            
            for task_id in task_ids:
                if task_id in self.task_memories:
                    memory = self.task_memories[task_id]
                    
                    if len(memory.data_samples) > 0:
                        # Sample indices
                        num_samples = min(samples_per_task, len(memory.data_samples))
                        indices = torch.randperm(len(memory.data_samples))[:num_samples]
                        
                        for idx in indices:
                            batch_data.append(memory.data_samples[idx])
                            batch_labels.append(memory.labels[idx])
                            batch_task_ids.append(task_id)
            
            if batch_data:
                return torch.stack(batch_data), torch.stack(batch_labels), batch_task_ids
            else:
                return torch.empty(0), torch.empty(0), []
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the replay buffer."""
        with self._lock:
            stats = {
                'total_tasks': len(self.task_memories),
                'total_samples': sum(len(memory.data_samples) for memory in self.task_memories.values()),
                'task_details': {}
            }
            
            for task_id, memory in self.task_memories.items():
                stats['task_details'][task_id] = {
                    'sample_count': len(memory.data_samples),
                    'max_samples': memory.max_samples,
                    'utilization': len(memory.data_samples) / memory.max_samples
                }
            
            return stats


class IncrementalLearningNetwork:
    """
    Main incremental learning network that combines all forgetting prevention strategies.
    """
    
    def __init__(self, weight_manager: SharedWeightBankManager,
                 strategy: ForgettingPreventionStrategy = ForgettingPreventionStrategy.MIXED):
        self.weight_manager = weight_manager
        self.strategy = strategy
        
        # Initialize components based on strategy
        self.ewc_regularizer = EWCRegularizer() if strategy in [ForgettingPreventionStrategy.EWC, ForgettingPreventionStrategy.MIXED] else None
        self.progressive_network = ProgressiveNetwork(weight_manager) if strategy in [ForgettingPreventionStrategy.PROGRESSIVE, ForgettingPreventionStrategy.MIXED] else None
        self.replay_buffer = MemoryReplayBuffer() if strategy in [ForgettingPreventionStrategy.REPLAY, ForgettingPreventionStrategy.MIXED] else None
        
        # Task management
        self.current_task = None
        self.learned_tasks = set()
        self.task_controllers: Dict[str, TopologyController] = {}
        
        # Learning parameters
        self.replay_weight = 0.5
        self.distillation_temperature = 3.0
        self.orthogonal_gradient_threshold = 0.1
        
        # Thread safety
        self._lock = threading.RLock()
    
    def start_task(self, task_id: str, layer_configs: List[Dict[str, Any]] = None):
        """Start learning a new task."""
        with self._lock:
            self.current_task = task_id
            
            if task_id not in self.learned_tasks:
                # Create task-specific topology controller
                controller = LearnableTopologyController(task_id, self.weight_manager)
                self.task_controllers[task_id] = controller
                
                # Initialize based on strategy
                if self.progressive_network and layer_configs:
                    self.progressive_network.add_task_column(task_id, layer_configs)
                
                self.learned_tasks.add(task_id)
    
    def compute_incremental_loss(self, current_loss: torch.Tensor, 
                                current_params: Dict[str, torch.Tensor],
                                batch_data: torch.Tensor = None,
                                batch_labels: torch.Tensor = None) -> torch.Tensor:
        """Compute total loss including regularization terms."""
        total_loss = current_loss
        
        # EWC regularization
        if self.ewc_regularizer and len(self.learned_tasks) > 1:
            ewc_penalty = self.ewc_regularizer.compute_penalty(
                current_params, exclude_tasks={self.current_task}
            )
            total_loss = total_loss + ewc_penalty
        
        # Replay loss
        if self.replay_buffer and len(self.learned_tasks) > 1:
            replay_loss = self._compute_replay_loss(current_params, batch_data, batch_labels)
            total_loss = total_loss + self.replay_weight * replay_loss
        
        # Knowledge distillation loss
        if len(self.learned_tasks) > 1:
            distillation_loss = self._compute_distillation_loss(current_params, batch_data)
            total_loss = total_loss + distillation_loss
        
        return total_loss
    
    def _compute_replay_loss(self, current_params: Dict[str, torch.Tensor],
                           batch_data: torch.Tensor, batch_labels: torch.Tensor) -> torch.Tensor:
        """Compute loss on replayed samples from previous tasks."""
        if not self.replay_buffer:
            return torch.tensor(0.0)
        
        # Get previous tasks (excluding current task)
        previous_tasks = [t for t in self.learned_tasks if t != self.current_task]
        
        if not previous_tasks:
            return torch.tensor(0.0)
        
        # Sample replay batch
        replay_data, replay_labels, replay_task_ids = self.replay_buffer.sample_replay_batch(
            previous_tasks, batch_size=min(32, batch_data.shape[0] if batch_data is not None else 32)
        )
        
        if replay_data.numel() == 0:
            return torch.tensor(0.0)
        
        # Forward pass on replay data (simplified)
        # In practice, you'd use the actual network forward pass
        replay_output = self._forward_with_params(replay_data, current_params)
        replay_loss = F.cross_entropy(replay_output, replay_labels.long())
        
        return replay_loss
    
    def _compute_distillation_loss(self, current_params: Dict[str, torch.Tensor],
                                 batch_data: torch.Tensor) -> torch.Tensor:
        """Compute knowledge distillation loss."""
        if batch_data is None or len(self.learned_tasks) <= 1:
            return torch.tensor(0.0)
        
        # Get outputs from previous task models (teacher)
        teacher_outputs = []
        for prev_task in self.learned_tasks:
            if prev_task != self.current_task:
                # Get teacher output (simplified)
                teacher_output = self._forward_with_task_params(batch_data, prev_task)
                teacher_outputs.append(teacher_output)
        
        if not teacher_outputs:
            return torch.tensor(0.0)
        
        # Get current model output (student)
        student_output = self._forward_with_params(batch_data, current_params)
        
        # Compute distillation loss
        distillation_loss = torch.tensor(0.0)
        for teacher_output in teacher_outputs:
            # Soft targets from teacher
            teacher_soft = F.softmax(teacher_output / self.distillation_temperature, dim=1)
            student_soft = F.log_softmax(student_output / self.distillation_temperature, dim=1)
            
            # KL divergence loss
            kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
            distillation_loss += kl_loss
        
        # Scale by temperature squared (standard in distillation)
        distillation_loss *= (self.distillation_temperature ** 2)
        
        return distillation_loss / len(teacher_outputs)
    
    def _forward_with_params(self, data: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with specific parameters (simplified placeholder)."""
        # This is a placeholder implementation
        # In practice, you'd implement functional forward pass
        return torch.randn(data.shape[0], 10)
    
    def _forward_with_task_params(self, data: torch.Tensor, task_id: str) -> torch.Tensor:
        """Forward pass using parameters specific to a task."""
        # This is a placeholder implementation
        # In practice, you'd use the stored parameters for the specific task
        return torch.randn(data.shape[0], 10)
    
    def orthogonalize_gradients(self, current_gradients: Dict[str, torch.Tensor],
                              task_id: str) -> Dict[str, torch.Tensor]:
        """Orthogonalize gradients to prevent interference with previous tasks."""
        if len(self.learned_tasks) <= 1:
            return current_gradients
        
        orthogonal_gradients = {}
        
        for param_name, grad in current_gradients.items():
            if grad is None:
                orthogonal_gradients[param_name] = grad
                continue
            
            # Start with current gradient
            orthogonal_grad = grad.clone()
            
            # Project out components that interfere with previous tasks
            for prev_task in self.learned_tasks:
                if prev_task == task_id:
                    continue
                
                # Get importance weights for previous task
                if self.ewc_regularizer and prev_task in self.ewc_regularizer.task_fisher_info:
                    importance = self.ewc_regularizer.task_fisher_info[prev_task].get(param_name)
                    
                    if importance is not None:
                        # Project out component in direction of important parameters
                        importance_normalized = importance / (importance.norm() + 1e-8)
                        projection = torch.sum(orthogonal_grad * importance_normalized) * importance_normalized
                        
                        # Remove projection if it's significant
                        if projection.norm() > self.orthogonal_gradient_threshold * orthogonal_grad.norm():
                            orthogonal_grad = orthogonal_grad - projection
            
            orthogonal_gradients[param_name] = orthogonal_grad
        
        return orthogonal_gradients
    
    def finish_task(self, task_id: str, final_params: Dict[str, torch.Tensor],
                   data_loader=None, loss_fn=None):
        """Finish learning a task and consolidate knowledge."""
        with self._lock:
            # Compute importance weights for EWC
            if self.ewc_regularizer and data_loader and loss_fn:
                self.ewc_regularizer.compute_importance(task_id, final_params, data_loader, loss_fn)
            
            # Freeze progressive network column
            if self.progressive_network:
                self.progressive_network.freeze_column(task_id)
            
            # Store samples in replay buffer
            if self.replay_buffer and data_loader:
                samples_data = []
                samples_labels = []
                
                # Collect some samples for replay
                for batch_idx, (data, labels) in enumerate(data_loader):
                    if batch_idx >= 10:  # Limit samples
                        break
                    samples_data.extend([d for d in data])
                    samples_labels.extend([l for l in labels])
                
                self.replay_buffer.add_task_samples(task_id, samples_data, samples_labels)
            
            # Update current task
            if self.current_task == task_id:
                self.current_task = None
    
    def evaluate_forgetting(self, task_evaluators: Dict[str, Callable]) -> Dict[str, float]:
        """Evaluate catastrophic forgetting across all learned tasks."""
        forgetting_scores = {}
        
        for task_id, evaluator in task_evaluators.items():
            if task_id in self.learned_tasks:
                # Evaluate current performance on this task
                current_performance = evaluator()
                forgetting_scores[task_id] = current_performance
        
        return forgetting_scores
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about incremental learning."""
        stats = {
            'strategy': self.strategy.value,
            'learned_tasks': list(self.learned_tasks),
            'current_task': self.current_task,
            'total_tasks': len(self.learned_tasks)
        }
        
        if self.ewc_regularizer:
            stats['ewc_tasks'] = list(self.ewc_regularizer.task_fisher_info.keys())
        
        if self.progressive_network:
            stats['progressive_columns'] = len(self.progressive_network.task_columns)
        
        if self.replay_buffer:
            stats['replay_stats'] = self.replay_buffer.get_memory_stats()
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    from shared_weight_banks import SharedWeightBankManager, WeightBankType
    
    # Create weight manager
    manager = SharedWeightBankManager(memory_limit_mb=512)
    
    # Create incremental learning network
    incremental_net = IncrementalLearningNetwork(
        manager, 
        strategy=ForgettingPreventionStrategy.MIXED
    )
    
    # Define layer configurations for progressive networks
    layer_configs = [
        {'type': 'linear', 'shape': (128, 784), 'initialization': 'xavier_uniform'},
        {'type': 'linear', 'shape': (64, 128), 'initialization': 'xavier_uniform'},
        {'type': 'linear', 'shape': (10, 64), 'initialization': 'xavier_uniform'}
    ]
    
    # Start first task
    incremental_net.start_task("mnist_classification", layer_configs)
    
    # Simulate training on first task
    dummy_params = {
        'layer1_weight': torch.randn(128, 784, requires_grad=True),
        'layer2_weight': torch.randn(64, 128, requires_grad=True),
        'layer3_weight': torch.randn(10, 64, requires_grad=True)
    }
    
    dummy_loss = torch.tensor(0.5, requires_grad=True)
    incremental_loss = incremental_net.compute_incremental_loss(dummy_loss, dummy_params)
    
    # Finish first task
    incremental_net.finish_task("mnist_classification", dummy_params)
    
    # Start second task
    incremental_net.start_task("cifar_classification", layer_configs)
    
    print("Incremental learning network initialized successfully!")
    print(f"Learning stats: {incremental_net.get_learning_stats()}")