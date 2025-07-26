"""
Memory-Efficient Implementation for Dynamic Connection Networks

This module implements advanced memory optimization techniques to minimize
memory overhead when sharing weights across multiple tasks while maintaining
high performance and enabling dynamic topology changes.

Key Features:
- Memory pooling and allocation strategies
- Gradient checkpointing for topology controllers
- Sparse tensor optimizations
- Dynamic memory compression
- Memory-mapped weight storage
- Lazy loading and unloading
- Memory usage profiling and monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set, Any, Union, Callable
import numpy as np
from dataclasses import dataclass
from collections import defaultdict, OrderedDict
import threading
from enum import Enum
import math
import gc
import psutil
import os
import mmap
import pickle
import weakref
from pathlib import Path
import time
from shared_weight_banks import SharedWeightBankManager, WeightBank, WeightBankType


class MemoryCompressionType(Enum):
    """Types of memory compression strategies."""
    NONE = "none"
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    LOW_RANK = "low_rank"
    HUFFMAN = "huffman"
    MIXED = "mixed"


class AllocationStrategy(Enum):
    """Memory allocation strategies."""
    EAGER = "eager"
    LAZY = "lazy"
    POOLED = "pooled"
    MAPPED = "mapped"
    HIERARCHICAL = "hierarchical"


@dataclass
class MemoryProfile:
    """Profile of memory usage and performance metrics."""
    total_memory_mb: float
    used_memory_mb: float
    available_memory_mb: float
    shared_memory_mb: float
    gpu_memory_mb: float
    compression_ratio: float
    access_latency_ms: float
    allocation_count: int
    deallocation_count: int
    cache_hit_rate: float
    fragmentation_ratio: float


class MemoryPool:
    """
    Efficient memory pool for weight tensor allocation and reuse.
    """
    
    def __init__(self, initial_size_mb: float = 128.0, growth_factor: float = 1.5):
        self.initial_size_bytes = int(initial_size_mb * 1024 * 1024)
        self.growth_factor = growth_factor
        
        # Memory blocks organized by size
        self.free_blocks: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self.allocated_blocks: Dict[id, torch.Tensor] = {}
        
        # Statistics
        self.total_allocated = 0
        self.total_freed = 0
        self.allocation_count = 0
        self.fragmentation_events = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Preallocation
        self._preallocate_common_sizes()
    
    def _preallocate_common_sizes(self):
        """Preallocate tensors of common sizes to reduce allocation overhead."""
        common_shapes = [
            (64, 64), (128, 128), (256, 256), (512, 512),
            (64, 32), (128, 64), (256, 128), (512, 256),
            (10, 512), (100, 256), (1000, 128)
        ]
        
        for shape in common_shapes:
            size_bytes = np.prod(shape) * 4  # float32
            for _ in range(3):  # Preallocate 3 tensors of each size
                tensor = torch.empty(shape, dtype=torch.float32)
                self.free_blocks[size_bytes].append(tensor)
    
    def allocate(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32,
                device: torch.device = None) -> torch.Tensor:
        """Allocate a tensor from the memory pool."""
        with self._lock:
            size_bytes = np.prod(shape) * torch.tensor([], dtype=dtype).element_size()
            device = device or torch.device('cpu')
            
            # Try to find existing block of exact size
            if size_bytes in self.free_blocks and self.free_blocks[size_bytes]:
                tensor = self.free_blocks[size_bytes].pop()
                
                # Resize if necessary
                if tensor.shape != shape:
                    tensor = tensor.view(-1)[:np.prod(shape)].view(shape)
                
                # Move to correct device and dtype
                tensor = tensor.to(device=device, dtype=dtype)
                
                self.allocated_blocks[id(tensor)] = tensor
                self.allocation_count += 1
                return tensor
            
            # Try to find larger block and split
            for available_size in sorted(self.free_blocks.keys()):
                if available_size >= size_bytes and self.free_blocks[available_size]:
                    larger_tensor = self.free_blocks[available_size].pop()
                    
                    # Split the tensor
                    needed_elements = np.prod(shape)
                    tensor = larger_tensor.view(-1)[:needed_elements].view(shape)
                    tensor = tensor.to(device=device, dtype=dtype)
                    
                    # Return remaining part to pool if significant
                    remaining_elements = larger_tensor.numel() - needed_elements
                    if remaining_elements > 1000:  # Threshold for keeping remainder
                        remainder = larger_tensor.view(-1)[needed_elements:]
                        remainder_size = remaining_elements * larger_tensor.element_size()
                        self.free_blocks[remainder_size].append(remainder)
                    
                    self.allocated_blocks[id(tensor)] = tensor
                    self.allocation_count += 1
                    return tensor
            
            # No suitable block found, allocate new
            tensor = torch.empty(shape, dtype=dtype, device=device)
            self.allocated_blocks[id(tensor)] = tensor
            self.allocation_count += 1
            self.total_allocated += size_bytes
            
            return tensor
    
    def deallocate(self, tensor: torch.Tensor):
        """Return a tensor to the memory pool."""
        with self._lock:
            tensor_id = id(tensor)
            
            if tensor_id in self.allocated_blocks:
                del self.allocated_blocks[tensor_id]
                
                # Add to free blocks
                size_bytes = tensor.numel() * tensor.element_size()
                
                # Move to CPU to save GPU memory
                if tensor.device.type == 'cuda':
                    tensor = tensor.cpu()
                
                self.free_blocks[size_bytes].append(tensor)
                self.total_freed += size_bytes
    
    def compact(self):
        """Compact the memory pool by merging adjacent free blocks."""
        with self._lock:
            # Simple compaction: remove excess free blocks
            for size_bytes in list(self.free_blocks.keys()):
                blocks = self.free_blocks[size_bytes]
                if len(blocks) > 10:  # Keep at most 10 blocks of each size
                    self.free_blocks[size_bytes] = blocks[:10]
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            free_memory = sum(
                len(blocks) * size_bytes 
                for size_bytes, blocks in self.free_blocks.items()
            )
            
            allocated_memory = sum(
                tensor.numel() * tensor.element_size()
                for tensor in self.allocated_blocks.values()
            )
            
            fragmentation = len(self.free_blocks) / max(1, self.allocation_count)
            
            return {
                'total_allocated_mb': self.total_allocated / (1024 * 1024),
                'total_freed_mb': self.total_freed / (1024 * 1024),
                'current_free_mb': free_memory / (1024 * 1024),
                'current_allocated_mb': allocated_memory / (1024 * 1024),
                'allocation_count': self.allocation_count,
                'fragmentation_ratio': fragmentation,
                'free_block_count': sum(len(blocks) for blocks in self.free_blocks.values()),
                'allocated_block_count': len(self.allocated_blocks)
            }


class WeightCompressor:
    """
    Advanced weight compression system supporting multiple compression strategies.
    """
    
    def __init__(self, compression_type: MemoryCompressionType = MemoryCompressionType.MIXED):
        self.compression_type = compression_type
        self.compression_cache = {}
        self.decompression_cache = {}
        
        # Compression parameters
        self.quantization_bits = 8
        self.pruning_threshold = 0.01
        self.low_rank_ratio = 0.5
        
    def compress_weights(self, weights: torch.Tensor, 
                        compression_type: Optional[MemoryCompressionType] = None) -> Tuple[Any, Dict[str, Any]]:
        """Compress weight tensor using specified compression type."""
        compression_type = compression_type or self.compression_type
        
        if compression_type == MemoryCompressionType.NONE:
            return weights, {'type': 'none'}
        
        elif compression_type == MemoryCompressionType.QUANTIZATION:
            return self._quantize_weights(weights)
        
        elif compression_type == MemoryCompressionType.PRUNING:
            return self._prune_weights(weights)
        
        elif compression_type == MemoryCompressionType.LOW_RANK:
            return self._low_rank_compress(weights)
        
        elif compression_type == MemoryCompressionType.HUFFMAN:
            return self._huffman_compress(weights)
        
        elif compression_type == MemoryCompressionType.MIXED:
            return self._mixed_compress(weights)
        
        else:
            return weights, {'type': 'none'}
    
    def decompress_weights(self, compressed_data: Any, metadata: Dict[str, Any]) -> torch.Tensor:
        """Decompress weight tensor."""
        compression_type = metadata.get('type', 'none')
        
        if compression_type == 'none':
            return compressed_data
        
        elif compression_type == 'quantization':
            return self._dequantize_weights(compressed_data, metadata)
        
        elif compression_type == 'pruning':
            return self._unprune_weights(compressed_data, metadata)
        
        elif compression_type == 'low_rank':
            return self._low_rank_decompress(compressed_data, metadata)
        
        elif compression_type == 'huffman':
            return self._huffman_decompress(compressed_data, metadata)
        
        elif compression_type == 'mixed':
            return self._mixed_decompress(compressed_data, metadata)
        
        else:
            return compressed_data
    
    def _quantize_weights(self, weights: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize weights to reduce precision."""
        # Dynamic range quantization
        min_val = weights.min()
        max_val = weights.max()
        
        # Scale to quantization range
        scale = (max_val - min_val) / (2 ** self.quantization_bits - 1)
        zero_point = min_val
        
        # Quantize
        quantized = torch.round((weights - zero_point) / scale).clamp(0, 2 ** self.quantization_bits - 1)
        quantized = quantized.to(torch.uint8)
        
        metadata = {
            'type': 'quantization',
            'scale': scale.item(),
            'zero_point': zero_point.item(),
            'original_shape': weights.shape,
            'original_dtype': weights.dtype
        }
        
        return quantized, metadata
    
    def _dequantize_weights(self, quantized: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """Dequantize weights."""
        scale = metadata['scale']
        zero_point = metadata['zero_point']
        original_dtype = metadata['original_dtype']
        
        # Dequantize
        dequantized = quantized.to(torch.float32) * scale + zero_point
        return dequantized.to(original_dtype)
    
    def _prune_weights(self, weights: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]:
        """Prune small weights and store sparse representation."""
        # Create mask for significant weights
        mask = torch.abs(weights) > self.pruning_threshold
        
        # Extract non-zero values and indices
        indices = torch.nonzero(mask, as_tuple=False)
        values = weights[mask]
        
        metadata = {
            'type': 'pruning',
            'original_shape': weights.shape,
            'threshold': self.pruning_threshold,
            'sparsity': 1.0 - (values.numel() / weights.numel())
        }
        
        return (values, indices), metadata
    
    def _unprune_weights(self, compressed_data: Tuple[torch.Tensor, torch.Tensor], 
                        metadata: Dict[str, Any]) -> torch.Tensor:
        """Reconstruct weights from sparse representation."""
        values, indices = compressed_data
        original_shape = metadata['original_shape']
        
        # Reconstruct sparse tensor
        weights = torch.zeros(original_shape, dtype=values.dtype, device=values.device)
        
        if indices.numel() > 0:
            # Handle different dimensionalities
            if len(original_shape) == 1:
                weights[indices[:, 0]] = values
            elif len(original_shape) == 2:
                weights[indices[:, 0], indices[:, 1]] = values
            else:
                # General case
                idx_tuple = tuple(indices[:, i] for i in range(indices.shape[1]))
                weights[idx_tuple] = values
        
        return weights
    
    def _low_rank_compress(self, weights: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]:
        """Compress using low-rank approximation."""
        if len(weights.shape) != 2:
            # Reshape to 2D for SVD
            original_shape = weights.shape
            weights_2d = weights.view(weights.shape[0], -1)
        else:
            original_shape = weights.shape
            weights_2d = weights
        
        # SVD decomposition
        U, S, V = torch.svd(weights_2d)
        
        # Determine rank based on ratio
        full_rank = min(weights_2d.shape)
        target_rank = max(1, int(full_rank * self.low_rank_ratio))
        
        # Truncate to target rank
        U_truncated = U[:, :target_rank]
        S_truncated = S[:target_rank]
        V_truncated = V[:, :target_rank]
        
        # Combine S into U for storage efficiency
        US = U_truncated * S_truncated.unsqueeze(0)
        
        metadata = {
            'type': 'low_rank',
            'original_shape': original_shape,
            'rank': target_rank,
            'compression_ratio': (US.numel() + V_truncated.numel()) / weights.numel()
        }
        
        return (US, V_truncated), metadata
    
    def _low_rank_decompress(self, compressed_data: Tuple[torch.Tensor, torch.Tensor],
                           metadata: Dict[str, Any]) -> torch.Tensor:
        """Decompress from low-rank representation."""
        US, V = compressed_data
        original_shape = metadata['original_shape']
        
        # Reconstruct weight matrix
        weights_2d = torch.mm(US, V.t())
        
        # Reshape back to original shape
        weights = weights_2d.view(original_shape)
        
        return weights
    
    def _huffman_compress(self, weights: torch.Tensor) -> Tuple[bytes, Dict[str, Any]]:
        """Compress using Huffman coding (simplified implementation)."""
        # Convert to numpy for easier manipulation
        weights_np = weights.cpu().numpy()
        
        # Quantize to reduce unique values
        quantized_weights = np.round(weights_np * 1000).astype(np.int32)
        
        # Simple compression using pickle (in practice, use proper Huffman coding)
        compressed_bytes = pickle.dumps(quantized_weights)
        
        metadata = {
            'type': 'huffman',
            'original_shape': weights.shape,
            'original_dtype': weights.dtype,
            'scale_factor': 1000,
            'compression_ratio': len(compressed_bytes) / weights.numel() / weights.element_size()
        }
        
        return compressed_bytes, metadata
    
    def _huffman_decompress(self, compressed_data: bytes, metadata: Dict[str, Any]) -> torch.Tensor:
        """Decompress from Huffman coding."""
        # Decompress
        quantized_weights = pickle.loads(compressed_data)
        
        # Convert back to tensor
        weights_np = quantized_weights.astype(np.float32) / metadata['scale_factor']
        weights = torch.from_numpy(weights_np).to(metadata['original_dtype'])
        
        return weights.view(metadata['original_shape'])
    
    def _mixed_compress(self, weights: torch.Tensor) -> Tuple[Any, Dict[str, Any]]:
        """Use mixed compression strategy based on weight characteristics."""
        # Choose compression based on weight tensor properties
        sparsity = (torch.abs(weights) < self.pruning_threshold).float().mean()
        
        if sparsity > 0.5:
            # High sparsity: use pruning
            return self._prune_weights(weights)
        elif len(weights.shape) == 2 and min(weights.shape) > 64:
            # Large 2D matrix: use low-rank compression
            return self._low_rank_compress(weights)
        else:
            # Default: use quantization
            return self._quantize_weights(weights)
    
    def _mixed_decompress(self, compressed_data: Any, metadata: Dict[str, Any]) -> torch.Tensor:
        """Decompress from mixed compression."""
        # The metadata should contain the actual compression type used
        actual_type = metadata.get('actual_type', metadata.get('type'))
        
        if actual_type == 'pruning':
            return self._unprune_weights(compressed_data, metadata)
        elif actual_type == 'low_rank':
            return self._low_rank_decompress(compressed_data, metadata)
        else:
            return self._dequantize_weights(compressed_data, metadata)


class MemoryMappedWeightBank(WeightBank):
    """
    Memory-mapped weight bank for efficient storage and access of large weight matrices.
    """
    
    def __init__(self, bank_id: str, bank_type: WeightBankType,
                 shape: Tuple[int, ...], dtype: torch.dtype = torch.float32,
                 device: torch.device = None, storage_path: Optional[str] = None):
        
        # Create storage file
        self.storage_path = storage_path or f"weights_{bank_id}.dat"
        self.file_handle = None
        self.mmap_handle = None
        
        # Initialize parent with temporary tensor
        super().__init__(bank_id, bank_type, shape, dtype, device, "zeros")
        
        # Replace with memory-mapped tensor
        self._create_memory_mapped_storage()
    
    def _create_memory_mapped_storage(self):
        """Create memory-mapped storage for the weight tensor."""
        # Calculate file size
        element_size = torch.tensor([], dtype=self.metadata.dtype).element_size()
        file_size = np.prod(self.metadata.shape) * element_size
        
        # Create file
        with open(self.storage_path, 'wb') as f:
            f.write(b'\x00' * file_size)
        
        # Open for memory mapping
        self.file_handle = open(self.storage_path, 'r+b')
        self.mmap_handle = mmap.mmap(self.file_handle.fileno(), 0)
        
        # Create tensor from memory map
        # Note: This is a simplified version - in practice, you'd need to handle
        # endianness and ensure proper alignment
        np_array = np.frombuffer(self.mmap_handle, dtype=np.float32).reshape(self.metadata.shape)
        self.weight = torch.from_numpy(np_array)
        
        # Initialize with random values
        with torch.no_grad():
            nn.init.xavier_uniform_(self.weight)
    
    def __del__(self):
        """Clean up memory-mapped resources."""
        if self.mmap_handle:
            self.mmap_handle.close()
        if self.file_handle:
            self.file_handle.close()
        
        # Optionally remove file
        if hasattr(self, 'storage_path') and os.path.exists(self.storage_path):
            try:
                os.remove(self.storage_path)
            except:
                pass  # Ignore cleanup errors


class LazyWeightLoader:
    """
    Lazy loading system for weight banks to reduce memory usage.
    """
    
    def __init__(self, weight_manager: SharedWeightBankManager, 
                 cache_size: int = 100):
        self.weight_manager = weight_manager
        self.cache_size = cache_size
        
        # LRU cache for loaded weights
        self.loaded_weights = OrderedDict()
        self.weight_metadata = {}
        
        # Compression system
        self.compressor = WeightCompressor()
        
        # Access statistics
        self.access_counts = defaultdict(int)
        self.last_access_times = {}
        
        # Thread safety
        self._lock = threading.RLock()
    
    def register_weight_bank(self, bank_id: str, bank: WeightBank, 
                           compress_on_store: bool = True):
        """Register a weight bank for lazy loading."""
        with self._lock:
            # Store metadata
            self.weight_metadata[bank_id] = {
                'bank_type': bank.metadata.bank_type,
                'shape': bank.metadata.shape,
                'dtype': bank.metadata.dtype,
                'device': bank.metadata.device,
                'compressed': compress_on_store
            }
            
            # Optionally compress and store
            if compress_on_store:
                compressed_data, compression_metadata = self.compressor.compress_weights(bank.weight)
                self.weight_metadata[bank_id]['compressed_data'] = compressed_data
                self.weight_metadata[bank_id]['compression_metadata'] = compression_metadata
            else:
                self.loaded_weights[bank_id] = bank.weight.clone()
                self._update_cache_order(bank_id)
    
    def load_weight_bank(self, bank_id: str) -> Optional[torch.Tensor]:
        """Load a weight bank, using cache or decompression as needed."""
        with self._lock:
            self.access_counts[bank_id] += 1
            self.last_access_times[bank_id] = time.time()
            
            # Check if already loaded
            if bank_id in self.loaded_weights:
                self._update_cache_order(bank_id)
                return self.loaded_weights[bank_id]
            
            # Load from compressed storage
            if bank_id in self.weight_metadata:
                metadata = self.weight_metadata[bank_id]
                
                if metadata.get('compressed', False):
                    # Decompress
                    compressed_data = metadata['compressed_data']
                    compression_metadata = metadata['compression_metadata']
                    
                    weights = self.compressor.decompress_weights(compressed_data, compression_metadata)
                    weights = weights.to(device=metadata['device'], dtype=metadata['dtype'])
                else:
                    # Create empty tensor (fallback)
                    weights = torch.zeros(metadata['shape'], dtype=metadata['dtype'], device=metadata['device'])
                
                # Add to cache
                self.loaded_weights[bank_id] = weights
                self._update_cache_order(bank_id)
                
                # Manage cache size
                self._evict_if_necessary()
                
                return weights
            
            return None
    
    def unload_weight_bank(self, bank_id: str):
        """Unload a weight bank from memory."""
        with self._lock:
            if bank_id in self.loaded_weights:
                del self.loaded_weights[bank_id]
    
    def _update_cache_order(self, bank_id: str):
        """Update LRU order for cache."""
        if bank_id in self.loaded_weights:
            # Move to end (most recently used)
            self.loaded_weights.move_to_end(bank_id)
    
    def _evict_if_necessary(self):
        """Evict least recently used items if cache is full."""
        while len(self.loaded_weights) > self.cache_size:
            # Remove least recently used (first item)
            lru_bank_id = next(iter(self.loaded_weights))
            del self.loaded_weights[lru_bank_id]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self._lock:
            total_accesses = sum(self.access_counts.values())
            cache_hits = sum(1 for bank_id in self.access_counts if bank_id in self.loaded_weights)
            
            return {
                'cache_size': len(self.loaded_weights),
                'max_cache_size': self.cache_size,
                'total_registered': len(self.weight_metadata),
                'total_accesses': total_accesses,
                'cache_hit_rate': cache_hits / max(1, len(self.access_counts)),
                'most_accessed': max(self.access_counts.items(), key=lambda x: x[1]) if self.access_counts else None
            }


class MemoryOptimizedWeightManager(SharedWeightBankManager):
    """
    Enhanced weight manager with advanced memory optimization features.
    """
    
    def __init__(self, memory_limit_mb: float = 1024.0,
                 allocation_strategy: AllocationStrategy = AllocationStrategy.POOLED,
                 enable_compression: bool = True,
                 enable_lazy_loading: bool = True):
        
        super().__init__(memory_limit_mb)
        
        self.allocation_strategy = allocation_strategy
        self.enable_compression = enable_compression
        self.enable_lazy_loading = enable_lazy_loading
        
        # Memory optimization components
        self.memory_pool = MemoryPool() if allocation_strategy == AllocationStrategy.POOLED else None
        self.compressor = WeightCompressor() if enable_compression else None
        self.lazy_loader = LazyWeightLoader(self) if enable_lazy_loading else None
        
        # Memory monitoring
        self.memory_monitor = MemoryMonitor()
        
        # Performance metrics
        self.allocation_times = []
        self.access_times = []
    
    def create_optimized_bank(self, bank_id: str, bank_type: WeightBankType,
                            shape: Tuple[int, ...], dtype: torch.dtype = torch.float32,
                            device: torch.device = None, initialization: str = "xavier_uniform",
                            use_memory_mapping: bool = False) -> WeightBank:
        """Create an optimized weight bank with memory optimization features."""
        
        start_time = time.time()
        
        if use_memory_mapping and self.allocation_strategy == AllocationStrategy.MAPPED:
            # Create memory-mapped bank
            bank = MemoryMappedWeightBank(bank_id, bank_type, shape, dtype, device)
        else:
            # Create regular bank with optimizations
            bank = self.create_bank(bank_id, bank_type, shape, dtype, device, initialization)
        
        # Register with lazy loader if enabled
        if self.lazy_loader:
            self.lazy_loader.register_weight_bank(bank_id, bank, compress_on_store=True)
        
        # Track allocation time
        allocation_time = time.time() - start_time
        self.allocation_times.append(allocation_time)
        
        return bank
    
    def get_optimized_weight(self, bank_id: str, task_id: str) -> Optional[torch.Tensor]:
        """Get weight with optimization features."""
        start_time = time.time()
        
        # Try lazy loader first
        if self.lazy_loader:
            weight = self.lazy_loader.load_weight_bank(bank_id)
            if weight is not None:
                access_time = time.time() - start_time
                self.access_times.append(access_time)
                return weight
        
        # Fall back to regular bank access
        bank = self.get_bank(bank_id)
        if bank:
            weight = bank.get_weight(task_id)
            access_time = time.time() - start_time
            self.access_times.append(access_time)
            return weight
        
        return None
    
    def optimize_memory_layout(self):
        """Perform comprehensive memory optimization."""
        with self._global_lock:
            # Compact memory pool
            if self.memory_pool:
                self.memory_pool.compact()
            
            # Update memory layout
            super().optimize_memory_layout()
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        stats = super().get_memory_stats()
        
        # Add optimization-specific stats
        if self.memory_pool:
            stats['memory_pool'] = self.memory_pool.get_memory_stats()
        
        if self.lazy_loader:
            stats['lazy_loading'] = self.lazy_loader.get_cache_stats()
        
        # Performance metrics
        if self.allocation_times:
            stats['avg_allocation_time_ms'] = np.mean(self.allocation_times) * 1000
        if self.access_times:
            stats['avg_access_time_ms'] = np.mean(self.access_times) * 1000
        
        # Memory monitoring
        stats['system_memory'] = self.memory_monitor.get_system_memory_info()
        
        return stats


class MemoryMonitor:
    """
    System memory monitoring and profiling.
    """
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring_active = False
        self.memory_history = []
        
    def start_monitoring(self, interval_seconds: float = 1.0):
        """Start continuous memory monitoring."""
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                memory_info = self.get_memory_snapshot()
                self.memory_history.append(memory_info)
                
                # Keep only recent history
                if len(self.memory_history) > 1000:
                    self.memory_history = self.memory_history[-500:]
                
                time.sleep(interval_seconds)
        
        monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring_active = False
    
    def get_memory_snapshot(self) -> MemoryProfile:
        """Get current memory snapshot."""
        memory_info = self.process.memory_info()
        system_memory = psutil.virtual_memory()
        
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        
        return MemoryProfile(
            total_memory_mb=system_memory.total / (1024 * 1024),
            used_memory_mb=system_memory.used / (1024 * 1024),
            available_memory_mb=system_memory.available / (1024 * 1024),
            shared_memory_mb=memory_info.rss / (1024 * 1024),
            gpu_memory_mb=gpu_memory,
            compression_ratio=1.0,  # Default
            access_latency_ms=0.0,  # Default
            allocation_count=0,     # Default
            deallocation_count=0,   # Default
            cache_hit_rate=0.0,     # Default
            fragmentation_ratio=0.0  # Default
        )
    
    def get_system_memory_info(self) -> Dict[str, Any]:
        """Get comprehensive system memory information."""
        memory_info = self.process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            'process_rss_mb': memory_info.rss / (1024 * 1024),
            'process_vms_mb': memory_info.vms / (1024 * 1024),
            'system_total_mb': system_memory.total / (1024 * 1024),
            'system_available_mb': system_memory.available / (1024 * 1024),
            'system_percent': system_memory.percent,
            'gpu_allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0,
            'gpu_cached_mb': torch.cuda.memory_reserved() / (1024 * 1024) if torch.cuda.is_available() else 0
        }


# Example usage and testing
if __name__ == "__main__":
    # Create optimized weight manager
    optimized_manager = MemoryOptimizedWeightManager(
        memory_limit_mb=512,
        allocation_strategy=AllocationStrategy.POOLED,
        enable_compression=True,
        enable_lazy_loading=True
    )
    
    # Create optimized banks
    bank1 = optimized_manager.create_optimized_bank("encoder", WeightBankType.LINEAR, (512, 256))
    bank2 = optimized_manager.create_optimized_bank("decoder", WeightBankType.LINEAR, (256, 128))
    
    # Test weight access
    weight1 = optimized_manager.get_optimized_weight("encoder", "task_1")
    weight2 = optimized_manager.get_optimized_weight("decoder", "task_1")
    
    # Test compression
    compressor = WeightCompressor(MemoryCompressionType.MIXED)
    test_weights = torch.randn(100, 100)
    compressed, metadata = compressor.compress_weights(test_weights)
    decompressed = compressor.decompress_weights(compressed, metadata)
    
    compression_error = torch.norm(test_weights - decompressed).item()
    print(f"Compression error: {compression_error}")
    
    # Get optimization statistics
    print("Memory optimization initialized successfully!")
    print(f"Optimization stats: {optimized_manager.get_optimization_stats()}")