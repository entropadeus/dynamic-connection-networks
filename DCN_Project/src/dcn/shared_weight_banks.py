"""
Advanced Shared Weight Banks System for Dynamic Connection Networks

This module implements a sophisticated weight sharing mechanism that allows
multiple tasks to share weight matrices while maintaining task-specific
connection patterns and enabling dynamic topology changes.

Key Features:
- Efficient weight bank storage with versioning
- Task-specific access patterns
- Memory-optimized sharing mechanisms
- Dynamic weight allocation and deallocation
- Gradient aggregation across shared weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set, Any
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import weakref
import threading
from enum import Enum


class WeightBankType(Enum):
    """Types of weight banks for different layer operations."""
    LINEAR = "linear"
    CONV2D = "conv2d"
    CONV1D = "conv1d"
    ATTENTION = "attention"
    EMBEDDING = "embedding"


@dataclass
class WeightBankMetadata:
    """Metadata for tracking weight bank usage and properties."""
    bank_id: str
    bank_type: WeightBankType
    shape: Tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    creation_time: float
    last_access_time: float
    reference_count: int
    memory_size: int
    is_frozen: bool = False
    task_ids: Set[str] = None
    
    def __post_init__(self):
        if self.task_ids is None:
            self.task_ids = set()


class WeightBank:
    """
    A single weight bank that can be shared across multiple tasks.
    
    Supports efficient access, gradient accumulation, and memory management.
    """
    
    def __init__(self, bank_id: str, bank_type: WeightBankType, 
                 shape: Tuple[int, ...], dtype: torch.dtype = torch.float32,
                 device: torch.device = None, initialization: str = "xavier_uniform"):
        self.metadata = WeightBankMetadata(
            bank_id=bank_id,
            bank_type=bank_type,
            shape=shape,
            dtype=dtype,
            device=device or torch.device('cpu'),
            creation_time=torch.cuda.Event().record() if torch.cuda.is_available() else 0,
            last_access_time=0,
            reference_count=0,
            memory_size=np.prod(shape) * torch.tensor([], dtype=dtype).element_size()
        )
        
        # Initialize weight tensor
        self.weight = self._initialize_weights(initialization)
        self.weight.requires_grad_(True)
        
        # Gradient accumulation for shared weights
        self.gradient_accumulator = torch.zeros_like(self.weight)
        self.gradient_count = 0
        
        # Task-specific scaling factors
        self.task_scales = {}
        
        # Lock for thread-safe operations
        self._lock = threading.RLock()
        
        # Version tracking for incremental learning
        self.version = 0
        self.checkpoint_weights = {}
    
    def _initialize_weights(self, initialization: str) -> torch.Tensor:
        """Initialize weights based on the specified initialization scheme."""
        weight = torch.empty(self.metadata.shape, dtype=self.metadata.dtype, 
                           device=self.metadata.device)
        
        if initialization == "xavier_uniform":
            nn.init.xavier_uniform_(weight)
        elif initialization == "xavier_normal":
            nn.init.xavier_normal_(weight)
        elif initialization == "kaiming_uniform":
            nn.init.kaiming_uniform_(weight, nonlinearity='relu')
        elif initialization == "kaiming_normal":
            nn.init.kaiming_normal_(weight, nonlinearity='relu')
        elif initialization == "orthogonal":
            nn.init.orthogonal_(weight)
        elif initialization == "zeros":
            nn.init.zeros_(weight)
        elif initialization == "ones":
            nn.init.ones_(weight)
        else:
            nn.init.normal_(weight, 0, 0.02)
        
        return weight
    
    def get_weight(self, task_id: str, apply_scaling: bool = True) -> torch.Tensor:
        """Get weight tensor for a specific task with optional scaling."""
        with self._lock:
            self.metadata.last_access_time = torch.cuda.Event().record() if torch.cuda.is_available() else 0
            self.metadata.task_ids.add(task_id)
            
            if apply_scaling and task_id in self.task_scales:
                return self.weight * self.task_scales[task_id]
            return self.weight
    
    def set_task_scaling(self, task_id: str, scale: torch.Tensor):
        """Set task-specific scaling factors."""
        with self._lock:
            self.task_scales[task_id] = scale.to(self.metadata.device)
    
    def accumulate_gradient(self, gradient: torch.Tensor, task_id: str):
        """Accumulate gradients from different tasks."""
        with self._lock:
            if gradient.shape != self.weight.shape:
                raise ValueError(f"Gradient shape {gradient.shape} doesn't match weight shape {self.weight.shape}")
            
            # Apply task-specific scaling to gradient if available
            if task_id in self.task_scales:
                gradient = gradient * self.task_scales[task_id]
            
            self.gradient_accumulator += gradient
            self.gradient_count += 1
    
    def apply_accumulated_gradients(self, learning_rate: float):
        """Apply accumulated gradients and reset accumulator."""
        with self._lock:
            if self.gradient_count > 0:
                # Average the accumulated gradients
                avg_gradient = self.gradient_accumulator / self.gradient_count
                
                # Apply gradient update
                self.weight.data -= learning_rate * avg_gradient
                
                # Reset accumulator
                self.gradient_accumulator.zero_()
                self.gradient_count = 0
    
    def create_checkpoint(self, checkpoint_name: str):
        """Create a checkpoint of current weights."""
        with self._lock:
            self.checkpoint_weights[checkpoint_name] = self.weight.clone().detach()
            self.version += 1
    
    def restore_checkpoint(self, checkpoint_name: str):
        """Restore weights from a checkpoint."""
        with self._lock:
            if checkpoint_name in self.checkpoint_weights:
                self.weight.data.copy_(self.checkpoint_weights[checkpoint_name])
                return True
            return False
    
    def freeze(self):
        """Freeze the weight bank to prevent updates."""
        with self._lock:
            self.metadata.is_frozen = True
            self.weight.requires_grad_(False)
    
    def unfreeze(self):
        """Unfreeze the weight bank to allow updates."""
        with self._lock:
            self.metadata.is_frozen = False
            self.weight.requires_grad_(True)
    
    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        base_memory = self.metadata.memory_size
        accumulator_memory = self.gradient_accumulator.numel() * self.gradient_accumulator.element_size()
        checkpoint_memory = sum(w.numel() * w.element_size() for w in self.checkpoint_weights.values())
        scaling_memory = sum(s.numel() * s.element_size() for s in self.task_scales.values())
        
        return base_memory + accumulator_memory + checkpoint_memory + scaling_memory


class SharedWeightBankManager:
    """
    Central manager for all shared weight banks in the system.
    
    Provides efficient allocation, deallocation, and access to weight banks
    with memory optimization and usage tracking.
    """
    
    def __init__(self, memory_limit_mb: float = 1024.0):
        self.banks: Dict[str, WeightBank] = {}
        self.memory_limit_bytes = int(memory_limit_mb * 1024 * 1024)
        self.current_memory_usage = 0
        
        # Usage tracking
        self.access_patterns = defaultdict(list)
        self.task_to_banks = defaultdict(set)
        self.bank_to_tasks = defaultdict(set)
        
        # Thread safety
        self._global_lock = threading.RLock()
        
        # Garbage collection tracking
        self._weak_refs = {}
    
    def create_bank(self, bank_id: str, bank_type: WeightBankType,
                   shape: Tuple[int, ...], dtype: torch.dtype = torch.float32,
                   device: torch.device = None, initialization: str = "xavier_uniform") -> WeightBank:
        """Create a new weight bank."""
        with self._global_lock:
            if bank_id in self.banks:
                raise ValueError(f"Bank with ID '{bank_id}' already exists")
            
            # Check memory constraints
            estimated_memory = np.prod(shape) * torch.tensor([], dtype=dtype).element_size()
            if self.current_memory_usage + estimated_memory > self.memory_limit_bytes:
                self._cleanup_unused_banks()
                if self.current_memory_usage + estimated_memory > self.memory_limit_bytes:
                    raise RuntimeError(f"Insufficient memory for new bank. Required: {estimated_memory}, Available: {self.memory_limit_bytes - self.current_memory_usage}")
            
            # Create the bank
            bank = WeightBank(bank_id, bank_type, shape, dtype, device, initialization)
            self.banks[bank_id] = bank
            self.current_memory_usage += bank.get_memory_usage()
            
            # Set up weak reference for garbage collection
            self._weak_refs[bank_id] = weakref.ref(bank, lambda ref: self._bank_garbage_collected(bank_id))
            
            return bank
    
    def get_bank(self, bank_id: str) -> Optional[WeightBank]:
        """Get an existing weight bank."""
        with self._global_lock:
            return self.banks.get(bank_id)
    
    def register_task_bank_usage(self, task_id: str, bank_id: str):
        """Register that a task is using a specific bank."""
        with self._global_lock:
            if bank_id in self.banks:
                self.task_to_banks[task_id].add(bank_id)
                self.bank_to_tasks[bank_id].add(task_id)
                self.banks[bank_id].metadata.reference_count += 1
                self.access_patterns[bank_id].append((task_id, torch.cuda.Event().record() if torch.cuda.is_available() else 0))
    
    def unregister_task_bank_usage(self, task_id: str, bank_id: str):
        """Unregister task usage of a bank."""
        with self._global_lock:
            if bank_id in self.banks:
                self.task_to_banks[task_id].discard(bank_id)
                self.bank_to_tasks[bank_id].discard(task_id)
                self.banks[bank_id].metadata.reference_count = max(0, self.banks[bank_id].metadata.reference_count - 1)
                self.banks[bank_id].metadata.task_ids.discard(task_id)
    
    def get_task_banks(self, task_id: str) -> List[WeightBank]:
        """Get all banks used by a specific task."""
        with self._global_lock:
            bank_ids = self.task_to_banks.get(task_id, set())
            return [self.banks[bank_id] for bank_id in bank_ids if bank_id in self.banks]
    
    def get_shared_banks(self) -> List[Tuple[str, WeightBank, Set[str]]]:
        """Get all banks that are shared between multiple tasks."""
        with self._global_lock:
            shared = []
            for bank_id, bank in self.banks.items():
                task_ids = self.bank_to_tasks[bank_id]
                if len(task_ids) > 1:
                    shared.append((bank_id, bank, task_ids.copy()))
            return shared
    
    def optimize_memory_layout(self):
        """Optimize memory layout by consolidating frequently accessed banks."""
        with self._global_lock:
            # Sort banks by access frequency
            access_counts = {}
            for bank_id, accesses in self.access_patterns.items():
                access_counts[bank_id] = len(accesses)
            
            # Move frequently accessed banks to contiguous memory
            sorted_banks = sorted(access_counts.items(), key=lambda x: x[1], reverse=True)
            
            # This is a simplified version - in practice, you'd implement
            # more sophisticated memory layout optimization
            for bank_id, _ in sorted_banks[:10]:  # Top 10 most accessed
                if bank_id in self.banks:
                    bank = self.banks[bank_id]
                    # Ensure bank is on the correct device and contiguous
                    bank.weight.data = bank.weight.data.contiguous()
    
    def _cleanup_unused_banks(self):
        """Remove banks with zero references."""
        with self._global_lock:
            to_remove = []
            for bank_id, bank in self.banks.items():
                if bank.metadata.reference_count == 0 and len(self.bank_to_tasks[bank_id]) == 0:
                    to_remove.append(bank_id)
            
            for bank_id in to_remove:
                self._remove_bank(bank_id)
    
    def _remove_bank(self, bank_id: str):
        """Remove a bank and clean up associated data."""
        if bank_id in self.banks:
            bank = self.banks[bank_id]
            self.current_memory_usage -= bank.get_memory_usage()
            
            # Clean up tracking data
            del self.banks[bank_id]
            if bank_id in self.access_patterns:
                del self.access_patterns[bank_id]
            if bank_id in self.bank_to_tasks:
                del self.bank_to_tasks[bank_id]
            
            # Clean up task references
            for task_id in list(self.task_to_banks.keys()):
                self.task_to_banks[task_id].discard(bank_id)
    
    def _bank_garbage_collected(self, bank_id: str):
        """Callback for when a bank is garbage collected."""
        with self._global_lock:
            if bank_id in self._weak_refs:
                del self._weak_refs[bank_id]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory usage statistics."""
        with self._global_lock:
            stats = {
                'total_banks': len(self.banks),
                'total_memory_bytes': self.current_memory_usage,
                'total_memory_mb': self.current_memory_bytes / (1024 * 1024),
                'memory_limit_mb': self.memory_limit_bytes / (1024 * 1024),
                'memory_utilization': self.current_memory_usage / self.memory_limit_bytes,
                'banks_by_type': defaultdict(int),
                'shared_banks_count': len([b for b in self.bank_to_tasks.values() if len(b) > 1]),
                'average_sharing_factor': 0,
                'bank_details': []
            }
            
            total_sharing = 0
            for bank_id, bank in self.banks.items():
                stats['banks_by_type'][bank.metadata.bank_type.value] += 1
                task_count = len(self.bank_to_tasks[bank_id])
                total_sharing += task_count
                
                stats['bank_details'].append({
                    'bank_id': bank_id,
                    'type': bank.metadata.bank_type.value,
                    'shape': bank.metadata.shape,
                    'memory_mb': bank.get_memory_usage() / (1024 * 1024),
                    'task_count': task_count,
                    'reference_count': bank.metadata.reference_count,
                    'is_frozen': bank.metadata.is_frozen,
                    'version': bank.version
                })
            
            if len(self.banks) > 0:
                stats['average_sharing_factor'] = total_sharing / len(self.banks)
            
            return stats
    
    def save_state(self, filepath: str):
        """Save the entire weight bank manager state."""
        with self._global_lock:
            state = {
                'banks': {},
                'task_to_banks': dict(self.task_to_banks),
                'bank_to_tasks': dict(self.bank_to_tasks),
                'memory_limit_bytes': self.memory_limit_bytes
            }
            
            for bank_id, bank in self.banks.items():
                state['banks'][bank_id] = {
                    'metadata': {
                        'bank_type': bank.metadata.bank_type.value,
                        'shape': bank.metadata.shape,
                        'dtype': str(bank.metadata.dtype),
                        'device': str(bank.metadata.device),
                        'is_frozen': bank.metadata.is_frozen,
                        'task_ids': list(bank.metadata.task_ids),
                        'version': bank.version
                    },
                    'weight': bank.weight.cpu().numpy(),
                    'task_scales': {tid: scale.cpu().numpy() for tid, scale in bank.task_scales.items()},
                    'checkpoint_weights': {name: w.cpu().numpy() for name, w in bank.checkpoint_weights.items()}
                }
            
            torch.save(state, filepath)
    
    def load_state(self, filepath: str):
        """Load weight bank manager state from file."""
        with self._global_lock:
            state = torch.load(filepath, map_location='cpu')
            
            # Clear current state
            self.banks.clear()
            self.task_to_banks.clear()
            self.bank_to_tasks.clear()
            self.current_memory_usage = 0
            self.memory_limit_bytes = state['memory_limit_bytes']
            
            # Restore banks
            for bank_id, bank_data in state['banks'].items():
                metadata = bank_data['metadata']
                
                # Create bank
                bank = WeightBank(
                    bank_id=bank_id,
                    bank_type=WeightBankType(metadata['bank_type']),
                    shape=tuple(metadata['shape']),
                    dtype=getattr(torch, metadata['dtype'].split('.')[-1]),
                    device=torch.device(metadata['device']),
                    initialization="zeros"  # Will be overridden below
                )
                
                # Restore weight data
                bank.weight.data = torch.from_numpy(bank_data['weight']).to(bank.metadata.device)
                bank.metadata.is_frozen = metadata['is_frozen']
                bank.metadata.task_ids = set(metadata['task_ids'])
                bank.version = metadata['version']
                
                # Restore task scales
                for tid, scale_data in bank_data['task_scales'].items():
                    bank.task_scales[tid] = torch.from_numpy(scale_data).to(bank.metadata.device)
                
                # Restore checkpoints
                for name, weight_data in bank_data['checkpoint_weights'].items():
                    bank.checkpoint_weights[name] = torch.from_numpy(weight_data).to(bank.metadata.device)
                
                if bank.metadata.is_frozen:
                    bank.freeze()
                
                self.banks[bank_id] = bank
                self.current_memory_usage += bank.get_memory_usage()
            
            # Restore task mappings
            for task_id, bank_ids in state['task_to_banks'].items():
                self.task_to_banks[task_id] = set(bank_ids)
            
            for bank_id, task_ids in state['bank_to_tasks'].items():
                self.bank_to_tasks[bank_id] = set(task_ids)


# Example usage and testing
if __name__ == "__main__":
    # Create manager
    manager = SharedWeightBankManager(memory_limit_mb=512)
    
    # Create some weight banks
    bank1 = manager.create_bank("linear_1", WeightBankType.LINEAR, (256, 128))
    bank2 = manager.create_bank("conv_1", WeightBankType.CONV2D, (64, 32, 3, 3))
    
    # Register task usage
    manager.register_task_bank_usage("task_a", "linear_1")
    manager.register_task_bank_usage("task_b", "linear_1")  # Shared usage
    manager.register_task_bank_usage("task_b", "conv_1")
    
    # Set task-specific scaling
    bank1.set_task_scaling("task_a", torch.ones(256, 128) * 1.2)
    bank1.set_task_scaling("task_b", torch.ones(256, 128) * 0.8)
    
    # Get weights for different tasks
    weight_a = bank1.get_weight("task_a")
    weight_b = bank1.get_weight("task_b")
    
    print("Shared weight bank system initialized successfully!")
    print(f"Memory stats: {manager.get_memory_stats()}")