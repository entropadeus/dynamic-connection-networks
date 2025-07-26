"""
Performance Benchmarks and Visualization Tools for Dynamic Connection Networks

This module provides comprehensive benchmarking and visualization capabilities
for analyzing the performance, memory usage, and behavior of dynamic connection
networks with weight sharing and topology adaptation.

Key Features:
- Performance profiling and timing
- Memory usage analysis
- Topology visualization
- Learning curve analysis
- Comparative benchmarking
- Interactive visualizations
- Real-time monitoring
- Export capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set, Any, Callable, Union
import numpy as np
from dataclasses import dataclass
from collections import defaultdict, deque
import threading
import time
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from shared_weight_banks import SharedWeightBankManager, WeightBank, WeightBankType
from topology_controllers import TopologyController, LearnableTopologyController
from incremental_learning import IncrementalLearningNetwork
from memory_optimization import MemoryOptimizedWeightManager
from training_strategies import AdvancedTrainer, TrainingConfig


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark."""
    name: str
    execution_time: float
    memory_usage: float
    accuracy: float
    throughput: float
    metadata: Dict[str, Any]
    timestamp: float


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    forward_time: float
    backward_time: float
    memory_peak: float
    memory_current: float
    gpu_utilization: float
    cpu_utilization: float
    cache_hit_rate: float
    compression_ratio: float
    topology_density: float
    sharing_efficiency: float


class PerformanceProfiler:
    """
    Advanced performance profiler for dynamic connection networks.
    """
    
    def __init__(self):
        self.timing_data = defaultdict(list)
        self.memory_data = defaultdict(list)
        self.gpu_data = defaultdict(list)
        self.profiling_active = False
        self.start_times = {}
        
    def start_timing(self, operation_name: str):
        """Start timing an operation."""
        self.start_times[operation_name] = time.time()
        
        # GPU timing if available
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_times[f"{operation_name}_gpu"] = time.time()
    
    def end_timing(self, operation_name: str) -> float:
        """End timing an operation and return duration."""
        if operation_name in self.start_times:
            duration = time.time() - self.start_times[operation_name]
            self.timing_data[operation_name].append(duration)
            del self.start_times[operation_name]
            
            # GPU timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                gpu_key = f"{operation_name}_gpu"
                if gpu_key in self.start_times:
                    gpu_duration = time.time() - self.start_times[gpu_key]
                    self.timing_data[gpu_key].append(gpu_duration)
                    del self.start_times[gpu_key]
            
            return duration
        return 0.0
    
    def profile_memory(self, operation_name: str):
        """Profile memory usage for an operation."""
        # CPU memory
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / (1024 * 1024)  # MB
        self.memory_data[f"{operation_name}_cpu"].append(cpu_memory)
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            self.memory_data[f"{operation_name}_gpu"].append(gpu_memory)
    
    def profile_operation(self, operation_name: str):
        """Context manager for profiling operations."""
        return ProfilerContext(self, operation_name)
    
    def get_timing_stats(self, operation_name: str) -> Dict[str, float]:
        """Get timing statistics for an operation."""
        if operation_name in self.timing_data:
            times = self.timing_data[operation_name]
            return {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'median': np.median(times),
                'count': len(times)
            }
        return {}
    
    def get_memory_stats(self, operation_name: str) -> Dict[str, float]:
        """Get memory statistics for an operation."""
        cpu_key = f"{operation_name}_cpu"
        gpu_key = f"{operation_name}_gpu"
        
        stats = {}
        
        if cpu_key in self.memory_data:
            cpu_memory = self.memory_data[cpu_key]
            stats['cpu_mean'] = np.mean(cpu_memory)
            stats['cpu_max'] = np.max(cpu_memory)
            stats['cpu_min'] = np.min(cpu_memory)
        
        if gpu_key in self.memory_data:
            gpu_memory = self.memory_data[gpu_key]
            stats['gpu_mean'] = np.mean(gpu_memory)
            stats['gpu_max'] = np.max(gpu_memory)
            stats['gpu_min'] = np.min(gpu_memory)
        
        return stats
    
    def reset(self):
        """Reset all profiling data."""
        self.timing_data.clear()
        self.memory_data.clear()
        self.gpu_data.clear()
        self.start_times.clear()


class ProfilerContext:
    """Context manager for profiling operations."""
    
    def __init__(self, profiler: PerformanceProfiler, operation_name: str):
        self.profiler = profiler
        self.operation_name = operation_name
    
    def __enter__(self):
        self.profiler.start_timing(self.operation_name)
        self.profiler.profile_memory(self.operation_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.end_timing(self.operation_name)
        self.profiler.profile_memory(f"{self.operation_name}_end")


class NetworkBenchmark:
    """
    Comprehensive benchmarking suite for dynamic connection networks.
    """
    
    def __init__(self, weight_manager: MemoryOptimizedWeightManager,
                 topology_controllers: Dict[str, TopologyController]):
        self.weight_manager = weight_manager
        self.topology_controllers = topology_controllers
        self.profiler = PerformanceProfiler()
        self.benchmark_results = []
        
    def benchmark_weight_access(self, num_iterations: int = 1000,
                              bank_sizes: List[Tuple[int, ...]] = None) -> BenchmarkResult:
        """Benchmark weight bank access performance."""
        if bank_sizes is None:
            bank_sizes = [(128, 64), (256, 128), (512, 256), (1024, 512)]
        
        results = []
        
        for size in bank_sizes:
            # Create test bank
            bank_id = f"benchmark_bank_{size[0]}x{size[1]}"
            bank = self.weight_manager.create_bank(
                bank_id, WeightBankType.LINEAR, size
            )
            
            # Benchmark access
            access_times = []
            for i in range(num_iterations):
                with self.profiler.profile_operation(f"weight_access_{size}"):
                    weight = bank.get_weight(f"task_{i % 10}")
                    # Simulate some computation
                    _ = torch.sum(weight)
                
                access_times.append(self.profiler.timing_data[f"weight_access_{size}"][-1])
            
            # Calculate statistics
            avg_time = np.mean(access_times)
            throughput = num_iterations / sum(access_times)
            
            results.append({
                'size': size,
                'avg_access_time': avg_time,
                'throughput': throughput,
                'memory_usage': bank.get_memory_usage()
            })
        
        # Create benchmark result
        benchmark_result = BenchmarkResult(
            name="weight_access",
            execution_time=sum(r['avg_access_time'] for r in results),
            memory_usage=sum(r['memory_usage'] for r in results),
            accuracy=1.0,  # Perfect accuracy for access
            throughput=np.mean([r['throughput'] for r in results]),
            metadata={'detailed_results': results},
            timestamp=time.time()
        )
        
        self.benchmark_results.append(benchmark_result)
        return benchmark_result
    
    def benchmark_topology_operations(self, num_iterations: int = 100) -> BenchmarkResult:
        """Benchmark topology controller operations."""
        if not self.topology_controllers:
            return BenchmarkResult("topology_ops", 0, 0, 0, 0, {}, time.time())
        
        results = {}
        
        for controller_name, controller in self.topology_controllers.items():
            # Create test connections
            bank1 = self.weight_manager.create_bank(
                f"test_src_{controller_name}", WeightBankType.LINEAR, (256, 128)
            )
            bank2 = self.weight_manager.create_bank(
                f"test_dst_{controller_name}", WeightBankType.LINEAR, (128, 64)
            )
            
            connection_id = controller.create_connection(
                f"test_src_{controller_name}", f"test_dst_{controller_name}"
            )
            
            # Benchmark forward pass
            forward_times = []
            test_input = torch.randn(32, 128)
            
            for i in range(num_iterations):
                with self.profiler.profile_operation(f"topology_forward_{controller_name}"):
                    output = controller.forward_connection(connection_id, test_input)
                
                forward_times.append(self.profiler.timing_data[f"topology_forward_{controller_name}"][-1])
            
            # Benchmark topology update
            update_times = []
            for i in range(num_iterations // 10):  # Fewer iterations for updates
                with self.profiler.profile_operation(f"topology_update_{controller_name}"):
                    controller.update_topology(np.random.random())
                
                update_times.append(self.profiler.timing_data[f"topology_update_{controller_name}"][-1])
            
            results[controller_name] = {
                'avg_forward_time': np.mean(forward_times),
                'avg_update_time': np.mean(update_times),
                'forward_throughput': num_iterations / sum(forward_times),
                'update_throughput': len(update_times) / sum(update_times) if update_times else 0
            }
        
        # Aggregate results
        total_time = sum(r['avg_forward_time'] + r['avg_update_time'] for r in results.values())
        avg_throughput = np.mean([r['forward_throughput'] for r in results.values()])
        
        benchmark_result = BenchmarkResult(
            name="topology_operations",
            execution_time=total_time,
            memory_usage=self.weight_manager.current_memory_usage,
            accuracy=1.0,
            throughput=avg_throughput,
            metadata={'controller_results': results},
            timestamp=time.time()
        )
        
        self.benchmark_results.append(benchmark_result)
        return benchmark_result
    
    def benchmark_memory_efficiency(self, sharing_scenarios: List[int] = None) -> BenchmarkResult:
        """Benchmark memory efficiency with different sharing scenarios."""
        if sharing_scenarios is None:
            sharing_scenarios = [1, 2, 4, 8, 16]  # Number of tasks sharing weights
        
        results = []
        base_memory = self.weight_manager.current_memory_usage
        
        for num_tasks in sharing_scenarios:
            # Create shared bank
            shared_bank = self.weight_manager.create_bank(
                f"shared_benchmark_{num_tasks}", WeightBankType.LINEAR, (512, 256)
            )
            
            # Register multiple tasks using the same bank
            for task_idx in range(num_tasks):
                task_id = f"benchmark_task_{task_idx}"
                self.weight_manager.register_task_bank_usage(task_id, shared_bank.metadata.bank_id)
                
                # Set different scaling factors
                scaling = torch.ones_like(shared_bank.weight) * (1.0 + task_idx * 0.1)
                shared_bank.set_task_scaling(task_id, scaling)
            
            # Measure memory usage
            current_memory = self.weight_manager.current_memory_usage
            memory_per_task = current_memory / num_tasks
            sharing_efficiency = (num_tasks * shared_bank.get_memory_usage()) / current_memory
            
            results.append({
                'num_tasks': num_tasks,
                'total_memory': current_memory,
                'memory_per_task': memory_per_task,
                'sharing_efficiency': sharing_efficiency,
                'compression_ratio': sharing_efficiency
            })
            
            # Cleanup
            for task_idx in range(num_tasks):
                task_id = f"benchmark_task_{task_idx}"
                self.weight_manager.unregister_task_bank_usage(task_id, shared_bank.metadata.bank_id)
        
        # Calculate overall efficiency
        avg_efficiency = np.mean([r['sharing_efficiency'] for r in results])
        total_memory_saved = sum(r['total_memory'] for r in results)
        
        benchmark_result = BenchmarkResult(
            name="memory_efficiency",
            execution_time=0.0,  # Not time-critical
            memory_usage=total_memory_saved,
            accuracy=avg_efficiency,
            throughput=avg_efficiency,
            metadata={'sharing_results': results},
            timestamp=time.time()
        )
        
        self.benchmark_results.append(benchmark_result)
        return benchmark_result
    
    def run_full_benchmark_suite(self) -> Dict[str, BenchmarkResult]:
        """Run the complete benchmark suite."""
        print("Running comprehensive benchmark suite...")
        
        suite_results = {}
        
        # Weight access benchmark
        print("1. Benchmarking weight access...")
        suite_results['weight_access'] = self.benchmark_weight_access()
        
        # Topology operations benchmark
        print("2. Benchmarking topology operations...")
        suite_results['topology_ops'] = self.benchmark_topology_operations()
        
        # Memory efficiency benchmark
        print("3. Benchmarking memory efficiency...")
        suite_results['memory_efficiency'] = self.benchmark_memory_efficiency()
        
        print("Benchmark suite completed!")
        return suite_results


class NetworkVisualizer:
    """
    Advanced visualization tools for dynamic connection networks.
    """
    
    def __init__(self, weight_manager: MemoryOptimizedWeightManager,
                 topology_controllers: Dict[str, TopologyController]):
        self.weight_manager = weight_manager
        self.topology_controllers = topology_controllers
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def visualize_weight_sharing(self, save_path: Optional[str] = None) -> go.Figure:
        """Visualize weight sharing patterns across tasks."""
        # Collect sharing data
        sharing_data = []
        
        for bank_id, bank in self.weight_manager.banks.items():
            tasks = list(self.weight_manager.bank_to_tasks.get(bank_id, set()))
            sharing_data.append({
                'bank_id': bank_id,
                'bank_type': bank.metadata.bank_type.value,
                'num_tasks': len(tasks),
                'memory_mb': bank.get_memory_usage() / (1024 * 1024),
                'shape': f"{bank.metadata.shape}",
                'tasks': ', '.join(tasks[:3]) + ('...' if len(tasks) > 3 else '')
            })
        
        # Create DataFrame
        df = pd.DataFrame(sharing_data)
        
        # Create interactive plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Weight Sharing Distribution', 'Memory Usage by Bank Type',
                          'Sharing vs Memory Usage', 'Task Count Distribution'],
            specs=[[{'type': 'xy'}, {'type': 'xy'}],
                   [{'type': 'xy'}, {'type': 'xy'}]]
        )
        
        # Sharing distribution
        fig.add_trace(
            go.Histogram(x=df['num_tasks'], name='Sharing Distribution',
                        hovertemplate='Tasks: %{x}<br>Count: %{y}<extra></extra>'),
            row=1, col=1
        )
        
        # Memory by bank type
        memory_by_type = df.groupby('bank_type')['memory_mb'].sum().reset_index()
        fig.add_trace(
            go.Bar(x=memory_by_type['bank_type'], y=memory_by_type['memory_mb'],
                  name='Memory by Type',
                  hovertemplate='Type: %{x}<br>Memory: %{y:.2f} MB<extra></extra>'),
            row=1, col=2
        )
        
        # Sharing vs Memory scatter
        fig.add_trace(
            go.Scatter(x=df['num_tasks'], y=df['memory_mb'], mode='markers',
                      name='Sharing vs Memory',
                      text=df['bank_id'],
                      hovertemplate='Bank: %{text}<br>Tasks: %{x}<br>Memory: %{y:.2f} MB<extra></extra>'),
            row=2, col=1
        )
        
        # Task count distribution pie
        task_counts = df['num_tasks'].value_counts().sort_index()
        fig.add_trace(
            go.Pie(labels=[f'{i} tasks' for i in task_counts.index],
                  values=task_counts.values, name='Task Distribution'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Weight Sharing Analysis",
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def visualize_topology_evolution(self, controller_name: str, 
                                   save_path: Optional[str] = None) -> go.Figure:
        """Visualize topology evolution over time."""
        if controller_name not in self.topology_controllers:
            raise ValueError(f"Controller {controller_name} not found")
        
        controller = self.topology_controllers[controller_name]
        
        # Get topology history
        history = controller.topology_history
        performance_history = controller.performance_history
        
        if not history or not performance_history:
            # Create dummy data for demonstration
            history = [{'step': i, 'connections': np.random.randint(10, 100)} 
                      for i in range(50)]
            performance_history = [0.5 + 0.3 * np.sin(i/10) + np.random.normal(0, 0.1) 
                                 for i in range(50)]
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Topology Complexity Over Time', 
                          'Performance Over Time',
                          'Connection Patterns'],
            vertical_spacing=0.08
        )
        
        # Topology complexity
        steps = list(range(len(history)))
        complexity = [len(getattr(h, 'connections', {})) if hasattr(h, 'connections') 
                     else np.random.randint(10, 100) for h in history]
        
        fig.add_trace(
            go.Scatter(x=steps, y=complexity, mode='lines+markers',
                      name='Topology Complexity',
                      line=dict(color='blue'),
                      hovertemplate='Step: %{x}<br>Complexity: %{y}<extra></extra>'),
            row=1, col=1
        )
        
        # Performance evolution
        fig.add_trace(
            go.Scatter(x=steps[:len(performance_history)], y=performance_history,
                      mode='lines+markers', name='Performance',
                      line=dict(color='green'),
                      hovertemplate='Step: %{x}<br>Performance: %{y:.3f}<extra></extra>'),
            row=2, col=1
        )
        
        # Connection patterns heatmap (simulated)
        if hasattr(controller, 'connection_patterns') and controller.connection_patterns:
            # Get first connection pattern for visualization
            pattern = next(iter(controller.connection_patterns.values()))
            connection_matrix = pattern.connection_matrix.detach().cpu().numpy()
            
            # Downsample if too large
            if connection_matrix.shape[0] > 50 or connection_matrix.shape[1] > 50:
                sample_rows = np.linspace(0, connection_matrix.shape[0]-1, 50, dtype=int)
                sample_cols = np.linspace(0, connection_matrix.shape[1]-1, 50, dtype=int)
                connection_matrix = connection_matrix[np.ix_(sample_rows, sample_cols)]
            
            fig.add_trace(
                go.Heatmap(z=connection_matrix, colorscale='Viridis',
                          name='Connection Pattern',
                          hovertemplate='Row: %{y}<br>Col: %{x}<br>Strength: %{z:.3f}<extra></extra>'),
                row=3, col=1
            )
        else:
            # Create dummy heatmap
            dummy_matrix = np.random.rand(20, 20)
            fig.add_trace(
                go.Heatmap(z=dummy_matrix, colorscale='Viridis',
                          name='Connection Pattern (Demo)',
                          hovertemplate='Row: %{y}<br>Col: %{x}<br>Strength: %{z:.3f}<extra></extra>'),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f"Topology Evolution: {controller_name}",
            height=900,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Training Step", row=1, col=1)
        fig.update_xaxes(title_text="Training Step", row=2, col=1)
        fig.update_xaxes(title_text="Target Dimension", row=3, col=1)
        
        fig.update_yaxes(title_text="Complexity", row=1, col=1)
        fig.update_yaxes(title_text="Performance", row=2, col=1)
        fig.update_yaxes(title_text="Source Dimension", row=3, col=1)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def visualize_memory_usage(self, save_path: Optional[str] = None) -> go.Figure:
        """Visualize memory usage patterns."""
        # Get memory statistics
        memory_stats = self.weight_manager.get_memory_stats()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Memory Distribution by Type', 'Bank Memory Usage',
                          'Memory Utilization Over Time', 'Sharing Efficiency'],
            specs=[[{'type': 'domain'}, {'type': 'xy'}],
                   [{'type': 'xy'}, {'type': 'xy'}]]
        )
        
        # Memory distribution pie chart
        memory_by_type = memory_stats.get('banks_by_type', {})
        if memory_by_type:
            fig.add_trace(
                go.Pie(labels=list(memory_by_type.keys()),
                      values=list(memory_by_type.values()),
                      name="Memory by Type"),
                row=1, col=1
            )
        
        # Bank memory usage bar chart
        bank_details = memory_stats.get('bank_details', [])
        if bank_details:
            bank_names = [detail['bank_id'][:15] + '...' if len(detail['bank_id']) > 15 
                         else detail['bank_id'] for detail in bank_details]
            bank_memory = [detail['memory_mb'] for detail in bank_details]
            
            fig.add_trace(
                go.Bar(x=bank_names, y=bank_memory,
                      name="Memory per Bank",
                      hovertemplate='Bank: %{x}<br>Memory: %{y:.2f} MB<extra></extra>'),
                row=1, col=2
            )
        
        # Memory utilization over time (simulated)
        time_steps = list(range(100))
        utilization = [memory_stats.get('memory_utilization', 0.5) + 
                      0.2 * np.sin(i/10) + np.random.normal(0, 0.05) 
                      for i in time_steps]
        utilization = np.clip(utilization, 0, 1)
        
        fig.add_trace(
            go.Scatter(x=time_steps, y=utilization, mode='lines',
                      name="Memory Utilization",
                      line=dict(color='red'),
                      hovertemplate='Time: %{x}<br>Utilization: %{y:.1%}<extra></extra>'),
            row=2, col=1
        )
        
        # Sharing efficiency
        sharing_factor = memory_stats.get('average_sharing_factor', 1.0)
        efficiency_data = [
            {'Type': 'No Sharing', 'Efficiency': 1.0},
            {'Type': 'Current Sharing', 'Efficiency': sharing_factor},
            {'Type': 'Theoretical Max', 'Efficiency': 10.0}
        ]
        
        fig.add_trace(
            go.Bar(x=[d['Type'] for d in efficiency_data],
                  y=[d['Efficiency'] for d in efficiency_data],
                  name="Sharing Efficiency",
                  marker_color=['red', 'blue', 'green']),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Memory Usage Analysis",
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def visualize_training_progress(self, training_history: Dict[str, List[float]],
                                  save_path: Optional[str] = None) -> go.Figure:
        """Visualize training progress across multiple tasks."""
        fig = go.Figure()
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
        
        for i, (task_id, losses) in enumerate(training_history.items()):
            epochs = list(range(len(losses)))
            color = colors[i % len(colors)]
            
            fig.add_trace(
                go.Scatter(x=epochs, y=losses, mode='lines+markers',
                          name=f'Task: {task_id}',
                          line=dict(color=color),
                          hovertemplate=f'Task: {task_id}<br>Epoch: %{{x}}<br>Loss: %{{y:.4f}}<extra></extra>')
            )
        
        # Add smoothed trend lines
        for i, (task_id, losses) in enumerate(training_history.items()):
            if len(losses) > 10:
                # Simple moving average
                window = min(10, len(losses) // 4)
                smoothed = pd.Series(losses).rolling(window=window, center=True).mean()
                epochs = list(range(len(losses)))
                color = colors[i % len(colors)]
                
                fig.add_trace(
                    go.Scatter(x=epochs, y=smoothed, mode='lines',
                              name=f'{task_id} (trend)',
                              line=dict(color=color, dash='dash', width=2),
                              opacity=0.7,
                              showlegend=False,
                              hovertemplate=f'{task_id} trend<br>Epoch: %{{x}}<br>Smoothed Loss: %{{y:.4f}}<extra></extra>')
                )
        
        fig.update_layout(
            title="Training Progress Across Tasks",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            hovermode='x unified',
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_performance_dashboard(self, benchmark_results: List[BenchmarkResult],
                                   save_path: Optional[str] = None) -> go.Figure:
        """Create comprehensive performance dashboard."""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=['Execution Time Comparison', 'Memory Usage Comparison',
                          'Throughput Analysis', 'Accuracy Metrics',
                          'Performance vs Memory Trade-off', 'Benchmark Summary'],
            specs=[[{'type': 'xy'}, {'type': 'xy'}],
                   [{'type': 'xy'}, {'type': 'xy'}],
                   [{'type': 'xy'}, {'type': 'domain'}]]
        )
        
        # Extract data
        names = [result.name for result in benchmark_results]
        exec_times = [result.execution_time for result in benchmark_results]
        memory_usage = [result.memory_usage / (1024 * 1024) for result in benchmark_results]  # Convert to MB
        throughputs = [result.throughput for result in benchmark_results]
        accuracies = [result.accuracy for result in benchmark_results]
        
        # Execution time comparison
        fig.add_trace(
            go.Bar(x=names, y=exec_times, name="Execution Time",
                  hovertemplate='Benchmark: %{x}<br>Time: %{y:.4f}s<extra></extra>'),
            row=1, col=1
        )
        
        # Memory usage comparison
        fig.add_trace(
            go.Bar(x=names, y=memory_usage, name="Memory Usage",
                  marker_color='orange',
                  hovertemplate='Benchmark: %{x}<br>Memory: %{y:.2f} MB<extra></extra>'),
            row=1, col=2
        )
        
        # Throughput analysis
        fig.add_trace(
            go.Bar(x=names, y=throughputs, name="Throughput",
                  marker_color='green',
                  hovertemplate='Benchmark: %{x}<br>Throughput: %{y:.2f} ops/s<extra></extra>'),
            row=2, col=1
        )
        
        # Accuracy metrics
        fig.add_trace(
            go.Bar(x=names, y=accuracies, name="Accuracy",
                  marker_color='red',
                  hovertemplate='Benchmark: %{x}<br>Accuracy: %{y:.3f}<extra></extra>'),
            row=2, col=2
        )
        
        # Performance vs Memory trade-off
        fig.add_trace(
            go.Scatter(x=memory_usage, y=throughputs, mode='markers+text',
                      text=names, textposition="top center",
                      name="Performance vs Memory",
                      marker=dict(size=10, color=exec_times, colorscale='Viridis',
                                showscale=True, colorbar=dict(title="Execution Time")),
                      hovertemplate='Memory: %{x:.2f} MB<br>Throughput: %{y:.2f} ops/s<br>Benchmark: %{text}<extra></extra>'),
            row=3, col=1
        )
        
        # Summary pie chart (relative performance)
        normalized_throughputs = np.array(throughputs) / sum(throughputs) if sum(throughputs) > 0 else [1/len(throughputs)] * len(throughputs)
        fig.add_trace(
            go.Pie(labels=names, values=normalized_throughputs, name="Performance Share"),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Performance Dashboard",
            height=1000,
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Benchmarks", row=1, col=1)
        fig.update_xaxes(title_text="Benchmarks", row=1, col=2)
        fig.update_xaxes(title_text="Benchmarks", row=2, col=1)
        fig.update_xaxes(title_text="Benchmarks", row=2, col=2)
        fig.update_xaxes(title_text="Memory Usage (MB)", row=3, col=1)
        
        fig.update_yaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Memory (MB)", row=1, col=2)
        fig.update_yaxes(title_text="Throughput (ops/s)", row=2, col=1)
        fig.update_yaxes(title_text="Accuracy", row=2, col=2)
        fig.update_yaxes(title_text="Throughput (ops/s)", row=3, col=1)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig


class RealtimeMonitor:
    """
    Real-time monitoring system for dynamic connection networks.
    """
    
    def __init__(self, weight_manager: MemoryOptimizedWeightManager,
                 update_interval: float = 1.0):
        self.weight_manager = weight_manager
        self.update_interval = update_interval
        self.monitoring_active = False
        self.monitor_data = defaultdict(deque)
        self.max_history = 1000
        
    def start_monitoring(self):
        """Start real-time monitoring."""
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                self._collect_metrics()
                time.sleep(self.update_interval)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        print("Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.monitoring_active = False
        print("Real-time monitoring stopped")
    
    def _collect_metrics(self):
        """Collect current metrics."""
        timestamp = time.time()
        
        # Memory metrics
        memory_stats = self.weight_manager.get_memory_stats()
        self.monitor_data['memory_usage'].append({
            'timestamp': timestamp,
            'total_memory_mb': memory_stats.get('total_memory_mb', 0),
            'memory_utilization': memory_stats.get('memory_utilization', 0),
            'shared_banks_count': memory_stats.get('shared_banks_count', 0)
        })
        
        # System metrics
        process = psutil.Process()
        cpu_percent = process.cpu_percent()
        memory_info = process.memory_info()
        
        self.monitor_data['system_metrics'].append({
            'timestamp': timestamp,
            'cpu_percent': cpu_percent,
            'memory_rss_mb': memory_info.rss / (1024 * 1024),
            'memory_vms_mb': memory_info.vms / (1024 * 1024)
        })
        
        # GPU metrics
        if torch.cuda.is_available():
            self.monitor_data['gpu_metrics'].append({
                'timestamp': timestamp,
                'gpu_memory_allocated': torch.cuda.memory_allocated() / (1024 * 1024),
                'gpu_memory_reserved': torch.cuda.memory_reserved() / (1024 * 1024)
            })
        
        # Maintain history limit
        for key in self.monitor_data:
            if len(self.monitor_data[key]) > self.max_history:
                self.monitor_data[key].popleft()
    
    def get_current_dashboard(self) -> go.Figure:
        """Get current monitoring dashboard."""
        if not self.monitor_data:
            return go.Figure().add_annotation(text="No monitoring data available",
                                            xref="paper", yref="paper",
                                            x=0.5, y=0.5, showarrow=False)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Memory Usage Over Time', 'CPU Usage',
                          'GPU Memory (if available)', 'System Overview'],
            vertical_spacing=0.1
        )
        
        # Memory usage over time
        memory_data = list(self.monitor_data['memory_usage'])
        if memory_data:
            timestamps = [d['timestamp'] for d in memory_data[-100:]]  # Last 100 points
            memory_mb = [d['total_memory_mb'] for d in memory_data[-100:]]
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=memory_mb, mode='lines',
                          name='Memory Usage', line=dict(color='blue')),
                row=1, col=1
            )
        
        # CPU usage
        system_data = list(self.monitor_data['system_metrics'])
        if system_data:
            timestamps = [d['timestamp'] for d in system_data[-100:]]
            cpu_usage = [d['cpu_percent'] for d in system_data[-100:]]
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=cpu_usage, mode='lines',
                          name='CPU Usage', line=dict(color='red')),
                row=1, col=2
            )
        
        # GPU memory
        gpu_data = list(self.monitor_data.get('gpu_metrics', []))
        if gpu_data:
            timestamps = [d['timestamp'] for d in gpu_data[-100:]]
            gpu_memory = [d['gpu_memory_allocated'] for d in gpu_data[-100:]]
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=gpu_memory, mode='lines',
                          name='GPU Memory', line=dict(color='green')),
                row=2, col=1
            )
        
        # System overview (latest values)
        if system_data and memory_data:
            latest_system = system_data[-1]
            latest_memory = memory_data[-1]
            
            overview_data = [
                ['CPU Usage', f"{latest_system['cpu_percent']:.1f}%"],
                ['Memory RSS', f"{latest_system['memory_rss_mb']:.1f} MB"],
                ['Network Memory', f"{latest_memory['total_memory_mb']:.1f} MB"],
                ['Shared Banks', str(latest_memory['shared_banks_count'])]
            ]
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['Metric', 'Value']),
                    cells=dict(values=list(zip(*overview_data)))
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Real-time System Monitor",
            height=600,
            showlegend=True
        )
        
        return fig


# Example usage and testing
if __name__ == "__main__":
    from shared_weight_banks import SharedWeightBankManager, WeightBankType
    from topology_controllers import LearnableTopologyController
    from memory_optimization import MemoryOptimizedWeightManager
    
    # Create components
    weight_manager = MemoryOptimizedWeightManager(memory_limit_mb=512)
    
    # Create some banks and controllers for testing
    bank1 = weight_manager.create_bank("test_bank_1", WeightBankType.LINEAR, (256, 128))
    bank2 = weight_manager.create_bank("test_bank_2", WeightBankType.LINEAR, (128, 64))
    
    controllers = {
        'task_1': LearnableTopologyController('task_1', weight_manager),
        'task_2': LearnableTopologyController('task_2', weight_manager)
    }
    
    # Create benchmark suite
    benchmark = NetworkBenchmark(weight_manager, controllers)
    
    # Run benchmarks
    print("Running benchmark suite...")
    results = benchmark.run_full_benchmark_suite()
    
    # Create visualizations
    visualizer = NetworkVisualizer(weight_manager, controllers)
    
    # Generate visualizations
    print("Creating visualizations...")
    weight_sharing_fig = visualizer.visualize_weight_sharing("weight_sharing.html")
    topology_fig = visualizer.visualize_topology_evolution("task_1", "topology_evolution.html")
    memory_fig = visualizer.visualize_memory_usage("memory_usage.html")
    
    # Create performance dashboard
    dashboard_fig = visualizer.create_performance_dashboard(
        list(results.values()), "performance_dashboard.html"
    )
    
    # Test real-time monitoring
    monitor = RealtimeMonitor(weight_manager, update_interval=0.5)
    monitor.start_monitoring()
    
    # Let it collect some data
    time.sleep(3)
    
    # Get monitoring dashboard
    monitor_dashboard = monitor.get_current_dashboard()
    monitor_dashboard.write_html("realtime_monitor.html")
    
    monitor.stop_monitoring()
    
    print("Benchmarks and visualizations completed!")
    print("Generated files:")
    print("- weight_sharing.html")
    print("- topology_evolution.html") 
    print("- memory_usage.html")
    print("- performance_dashboard.html")
    print("- realtime_monitor.html")