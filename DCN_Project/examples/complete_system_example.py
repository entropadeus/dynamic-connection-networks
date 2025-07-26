"""
Complete System Example: Dynamic Connection Network with Advanced Weight Sharing

This comprehensive example demonstrates the entire Dynamic Connection Network
system with weight sharing, topology adaptation, incremental learning, memory
optimization, and advanced training strategies working together.

The example includes:
1. Multi-task learning scenario with shared weight banks
2. Dynamic topology adaptation for different tasks
3. Incremental learning with catastrophic forgetting prevention
4. Memory-efficient weight sharing and compression
5. Advanced training strategies with gradient balancing
6. Performance monitoring and visualization
7. Real-world-like neural network implementation

Use Case: Multi-Domain Learning System
- Image Classification (CIFAR-10 style)
- Natural Language Processing (Sentiment Analysis)
- Time Series Prediction (Stock Prices)
- Knowledge Transfer between domains
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional, Any
import random
from dataclasses import dataclass

# Import our custom modules
from shared_weight_banks import SharedWeightBankManager, WeightBankType
from topology_controllers import LearnableTopologyController, EvolutionaryTopologyController, AttentionTopologyController
from incremental_learning import IncrementalLearningNetwork, ForgettingPreventionStrategy
from memory_optimization import MemoryOptimizedWeightManager, WeightCompressor, MemoryCompressionType
from training_strategies import AdvancedTrainer, TrainingConfig, TrainingStrategy
from benchmarks_visualization import NetworkBenchmark, NetworkVisualizer, RealtimeMonitor


@dataclass
class TaskConfig:
    """Configuration for each task in the multi-domain system."""
    task_id: str
    input_dim: int
    output_dim: int
    hidden_dims: List[int]
    task_type: str  # 'classification', 'regression', 'sequence'
    difficulty: float
    data_size: int


class DynamicConnectionLayer(nn.Module):
    """
    A neural network layer that uses dynamic connections through topology controllers.
    """
    
    def __init__(self, task_id: str, layer_name: str, 
                 topology_controller: LearnableTopologyController,
                 input_bank_id: str, output_bank_id: str,
                 activation: str = 'relu'):
        super().__init__()
        self.task_id = task_id
        self.layer_name = layer_name
        self.topology_controller = topology_controller
        self.input_bank_id = input_bank_id
        self.output_bank_id = output_bank_id
        self.activation = activation
        
        # Create connection through topology controller
        self.connection_id = topology_controller.create_learnable_connection(
            input_bank_id, output_bank_id, connection_type="dense"
        )
        
        # Bias parameters (task-specific)
        output_bank = topology_controller.weight_manager.get_bank(output_bank_id)
        if output_bank:
            output_dim = output_bank.metadata.shape[0]
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply dynamic connection
        output = self.topology_controller.forward_connection(self.connection_id, x)
        
        # Add bias if available
        if self.bias is not None:
            output = output + self.bias
        
        # Apply activation
        if self.activation == 'relu':
            output = F.relu(output)
        elif self.activation == 'tanh':
            output = torch.tanh(output)
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(output)
        elif self.activation == 'none':
            pass  # No activation
        
        return output


class MultiDomainDynamicNetwork(nn.Module):
    """
    Complete multi-domain neural network with dynamic connections and weight sharing.
    """
    
    def __init__(self, task_configs: List[TaskConfig], 
                 weight_manager: MemoryOptimizedWeightManager,
                 shared_hidden_dims: List[int] = [512, 256, 128]):
        super().__init__()
        
        self.task_configs = {config.task_id: config for config in task_configs}
        self.weight_manager = weight_manager
        self.shared_hidden_dims = shared_hidden_dims
        
        # Create topology controllers for each task
        self.topology_controllers = {}
        for task_config in task_configs:
            if task_config.task_type == 'classification':
                controller = LearnableTopologyController(task_config.task_id, weight_manager)
            elif task_config.task_type == 'regression':
                controller = EvolutionaryTopologyController(task_config.task_id, weight_manager)
            else:
                controller = AttentionTopologyController(task_config.task_id, weight_manager)
            
            self.topology_controllers[task_config.task_id] = controller
        
        # Create shared weight banks
        self._create_shared_banks()
        
        # Create task-specific layers
        self.task_networks = nn.ModuleDict()
        self._create_task_networks()
        
        # Task embeddings for meta-learning
        self.task_embeddings = nn.ParameterDict()
        for task_config in task_configs:
            self.task_embeddings[task_config.task_id] = nn.Parameter(
                torch.randn(64)  # Task embedding dimension
            )
    
    def _create_shared_banks(self):
        """Create shared weight banks for the network."""
        bank_configs = []
        
        # Shared hidden layers
        for i, hidden_dim in enumerate(self.shared_hidden_dims):
            if i == 0:
                # First hidden layer - will connect to task-specific input projections
                input_dim = 512  # Maximum input dimension across tasks
            else:
                input_dim = self.shared_hidden_dims[i-1]
            
            bank_id = f"shared_hidden_{i}"
            bank_configs.append((bank_id, WeightBankType.LINEAR, (hidden_dim, input_dim)))
        
        # Create the banks
        for bank_id, bank_type, shape in bank_configs:
            self.weight_manager.create_optimized_bank(
                bank_id, bank_type, shape, 
                initialization='xavier_uniform',
                use_memory_mapping=False
            )
        
        # Create task-specific input/output projection banks
        for task_config in self.task_configs.values():
            # Input projection bank
            input_bank_id = f"{task_config.task_id}_input_proj"
            self.weight_manager.create_optimized_bank(
                input_bank_id, WeightBankType.LINEAR, 
                (512, task_config.input_dim),  # Project to shared dimension
                initialization='xavier_uniform'
            )
            
            # Output projection bank
            output_bank_id = f"{task_config.task_id}_output_proj"
            self.weight_manager.create_optimized_bank(
                output_bank_id, WeightBankType.LINEAR,
                (task_config.output_dim, self.shared_hidden_dims[-1]),
                initialization='xavier_uniform'
            )
    
    def _create_task_networks(self):
        """Create task-specific network architectures."""
        for task_config in self.task_configs.values():
            task_id = task_config.task_id
            controller = self.topology_controllers[task_id]
            
            layers = nn.ModuleList()
            
            # Input projection layer
            input_layer = DynamicConnectionLayer(
                task_id, "input_proj", controller,
                f"{task_id}_input_proj", "shared_hidden_0",
                activation='relu'
            )
            layers.append(input_layer)
            
            # Shared hidden layers
            for i in range(len(self.shared_hidden_dims) - 1):
                hidden_layer = DynamicConnectionLayer(
                    task_id, f"hidden_{i}", controller,
                    f"shared_hidden_{i}", f"shared_hidden_{i+1}",
                    activation='relu'
                )
                layers.append(hidden_layer)
            
            # Output projection layer
            output_layer = DynamicConnectionLayer(
                task_id, "output_proj", controller,
                f"shared_hidden_{len(self.shared_hidden_dims)-1}",
                f"{task_id}_output_proj",
                activation='none'  # No activation for output
            )
            layers.append(output_layer)
            
            self.task_networks[task_id] = layers
    
    def forward(self, x: torch.Tensor, task_id: str) -> torch.Tensor:
        """Forward pass for a specific task."""
        if task_id not in self.task_networks:
            raise ValueError(f"Task {task_id} not found")
        
        # Get task-specific network
        layers = self.task_networks[task_id]
        
        # Forward through all layers
        current_input = x
        for layer in layers:
            current_input = layer(current_input)
        
        return current_input
    
    def get_task_embedding(self, task_id: str) -> torch.Tensor:
        """Get task embedding for meta-learning."""
        return self.task_embeddings.get(task_id, torch.zeros(64))


class MultiDomainDataGenerator:
    """
    Generates synthetic data for multiple domains to demonstrate the system.
    """
    
    def __init__(self, task_configs: List[TaskConfig], seed: int = 42):
        self.task_configs = task_configs
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def generate_classification_data(self, task_config: TaskConfig, 
                                   num_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic classification data."""
        # Create data with some structure (e.g., clustered classes)
        X = torch.randn(num_samples, task_config.input_dim)
        
        # Add class-specific patterns
        y = torch.randint(0, task_config.output_dim, (num_samples,))
        
        for class_idx in range(task_config.output_dim):
            class_mask = (y == class_idx)
            class_center = torch.randn(task_config.input_dim) * 2
            X[class_mask] += class_center
        
        # Add some noise based on difficulty
        noise_scale = task_config.difficulty * 0.5
        X += torch.randn_like(X) * noise_scale
        
        return X, y
    
    def generate_regression_data(self, task_config: TaskConfig,
                               num_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic regression data."""
        X = torch.randn(num_samples, task_config.input_dim)
        
        # Create non-linear relationship
        W = torch.randn(task_config.input_dim, task_config.output_dim)
        y = torch.mm(X, W)
        
        # Add non-linearity based on difficulty
        if task_config.difficulty > 0.5:
            y = y + 0.1 * torch.sin(y * 3.14159)
        
        # Add noise
        noise_scale = task_config.difficulty * 0.3
        y += torch.randn_like(y) * noise_scale
        
        return X, y
    
    def generate_sequence_data(self, task_config: TaskConfig,
                             num_samples: int = 1000, seq_length: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic sequence data."""
        # Generate time series data
        X = torch.randn(num_samples, seq_length, task_config.input_dim)
        
        # Create temporal patterns
        for i in range(1, seq_length):
            X[:, i] += 0.5 * X[:, i-1]  # Autoregressive component
        
        # Flatten for input to network
        X_flat = X.view(num_samples, -1)
        
        # Target is next value prediction or classification
        if task_config.output_dim == 1:
            # Regression: predict next value
            y = X[:, -1, 0].unsqueeze(1)  # Predict first feature
        else:
            # Classification: classify sequence type
            y = torch.randint(0, task_config.output_dim, (num_samples,))
        
        return X_flat, y
    
    def generate_task_data(self, task_config: TaskConfig, 
                          num_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate data for a specific task."""
        if task_config.task_type == 'classification':
            return self.generate_classification_data(task_config, num_samples)
        elif task_config.task_type == 'regression':
            return self.generate_regression_data(task_config, num_samples)
        elif task_config.task_type == 'sequence':
            return self.generate_sequence_data(task_config, num_samples)
        else:
            raise ValueError(f"Unknown task type: {task_config.task_type}")


class MultiDomainTrainingSystem:
    """
    Complete training system for multi-domain dynamic networks.
    """
    
    def __init__(self, network: MultiDomainDynamicNetwork, 
                 training_config: TrainingConfig):
        self.network = network
        self.training_config = training_config
        
        # Initialize incremental learning
        self.incremental_learner = IncrementalLearningNetwork(
            network.weight_manager,
            strategy=ForgettingPreventionStrategy.MIXED
        )
        
        # Initialize trainer
        self.trainer = AdvancedTrainer(
            network.weight_manager,
            network.topology_controllers,
            self.incremental_learner,
            training_config
        )
        
        # Data generator
        self.data_generator = MultiDomainDataGenerator(
            list(network.task_configs.values())
        )
        
        # Performance tracking
        self.training_history = defaultdict(list)
        self.performance_metrics = defaultdict(list)
        
        # Benchmark and monitoring
        self.benchmark = NetworkBenchmark(
            network.weight_manager, network.topology_controllers
        )
        self.visualizer = NetworkVisualizer(
            network.weight_manager, network.topology_controllers
        )
        self.monitor = RealtimeMonitor(network.weight_manager)
    
    def create_data_loaders(self, batch_size: int = 32, 
                          train_samples: int = 1000,
                          val_samples: int = 200) -> Dict[str, Dict[str, Any]]:
        """Create data loaders for all tasks."""
        data_loaders = {}
        
        for task_config in self.network.task_configs.values():
            # Generate training data
            X_train, y_train = self.data_generator.generate_task_data(
                task_config, train_samples
            )
            
            # Generate validation data
            X_val, y_val = self.data_generator.generate_task_data(
                task_config, val_samples
            )
            
            # Create simple data loaders (in practice, use torch.utils.data.DataLoader)
            train_loader = {
                'data': X_train,
                'labels': y_train,
                'batch_size': batch_size,
                'num_batches': len(X_train) // batch_size
            }
            
            val_loader = {
                'data': X_val,
                'labels': y_val,
                'batch_size': batch_size,
                'num_batches': len(X_val) // batch_size
            }
            
            data_loaders[task_config.task_id] = {
                'train': train_loader,
                'val': val_loader
            }
        
        return data_loaders
    
    def create_loss_functions(self) -> Dict[str, Callable]:
        """Create appropriate loss functions for each task."""
        loss_functions = {}
        
        for task_config in self.network.task_configs.values():
            if task_config.task_type == 'classification':
                loss_functions[task_config.task_id] = self._classification_loss
            elif task_config.task_type == 'regression':
                loss_functions[task_config.task_id] = self._regression_loss
            else:
                loss_functions[task_config.task_id] = self._sequence_loss
        
        return loss_functions
    
    def _classification_loss(self, data: torch.Tensor, labels: torch.Tensor, 
                           task_id: str) -> torch.Tensor:
        """Classification loss function."""
        # Get random batch
        batch_size = min(32, len(data))
        indices = torch.randperm(len(data))[:batch_size]
        batch_data = data[indices]
        batch_labels = labels[indices]
        
        # Forward pass
        outputs = self.network(batch_data, task_id)
        
        # Cross-entropy loss
        loss = F.cross_entropy(outputs, batch_labels)
        return loss
    
    def _regression_loss(self, data: torch.Tensor, labels: torch.Tensor,
                        task_id: str) -> torch.Tensor:
        """Regression loss function."""
        # Get random batch
        batch_size = min(32, len(data))
        indices = torch.randperm(len(data))[:batch_size]
        batch_data = data[indices]
        batch_labels = labels[indices]
        
        # Forward pass
        outputs = self.network(batch_data, task_id)
        
        # MSE loss
        loss = F.mse_loss(outputs, batch_labels)
        return loss
    
    def _sequence_loss(self, data: torch.Tensor, labels: torch.Tensor,
                      task_id: str) -> torch.Tensor:
        """Sequence prediction loss function."""
        task_config = self.network.task_configs[task_id]
        
        if task_config.output_dim == 1:
            return self._regression_loss(data, labels, task_id)
        else:
            return self._classification_loss(data, labels, task_id)
    
    def evaluate_task(self, task_id: str, data_loader: Dict[str, Any]) -> float:
        """Evaluate performance on a specific task."""
        self.network.eval()
        
        data = data_loader['data']
        labels = data_loader['labels']
        task_config = self.network.task_configs[task_id]
        
        with torch.no_grad():
            outputs = self.network(data, task_id)
            
            if task_config.task_type == 'classification':
                # Accuracy
                predictions = torch.argmax(outputs, dim=1)
                accuracy = (predictions == labels).float().mean().item()
                return accuracy
            else:
                # R-squared for regression
                mse = F.mse_loss(outputs, labels)
                var = torch.var(labels)
                r2 = 1 - (mse / var)
                return r2.item()
    
    def train_epoch(self, data_loaders: Dict[str, Dict[str, Any]],
                   loss_functions: Dict[str, Callable]) -> Dict[str, float]:
        """Train for one epoch."""
        self.network.train()
        epoch_losses = {}
        
        # Prepare data for trainer
        train_loaders = {task_id: loaders['train'] 
                        for task_id, loaders in data_loaders.items()}
        
        # Create wrapped loss functions
        def wrapped_loss_fn(task_id):
            def loss_fn(data, labels):
                return loss_functions[task_id](data, labels, task_id)
            return loss_fn
        
        wrapped_losses = {task_id: wrapped_loss_fn(task_id) 
                         for task_id in loss_functions.keys()}
        
        # Train using the advanced trainer
        epoch_losses = self.trainer.train_epoch(train_loaders, wrapped_losses)
        
        return epoch_losses
    
    def train(self, num_epochs: int = 50, save_checkpoints: bool = True) -> Dict[str, Any]:
        """Complete training pipeline."""
        print("Starting multi-domain training...")
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Create data loaders and loss functions
        data_loaders = self.create_data_loaders()
        loss_functions = self.create_loss_functions()
        
        # Training loop
        best_avg_performance = 0.0
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Train for one epoch
            epoch_losses = self.train_epoch(data_loaders, loss_functions)
            
            # Evaluate on validation sets
            val_performances = {}
            for task_id, loaders in data_loaders.items():
                val_performance = self.evaluate_task(task_id, loaders['val'])
                val_performances[task_id] = val_performance
            
            # Record history
            for task_id, loss in epoch_losses.items():
                self.training_history[f"{task_id}_loss"].append(loss)
            
            for task_id, perf in val_performances.items():
                self.performance_metrics[f"{task_id}_val"].append(perf)
            
            # Update topology controllers
            for task_id, controller in self.network.topology_controllers.items():
                performance = val_performances.get(task_id, 0.0)
                controller.update_topology(performance)
            
            # Calculate average performance
            avg_performance = sum(val_performances.values()) / len(val_performances)
            
            # Save best model
            if avg_performance > best_avg_performance:
                best_avg_performance = avg_performance
                if save_checkpoints:
                    self._save_checkpoint(epoch, avg_performance)
            
            # Print progress
            epoch_time = time.time() - epoch_start_time
            if epoch % 5 == 0:
                print(f"Epoch {epoch:3d} | Time: {epoch_time:.2f}s | "
                      f"Avg Loss: {np.mean(list(epoch_losses.values())):.4f} | "
                      f"Avg Val Perf: {avg_performance:.4f}")
                
                # Print task-specific performance
                for task_id, perf in val_performances.items():
                    print(f"  {task_id}: {perf:.4f}")
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        # Run final benchmarks
        print("\nRunning final benchmarks...")
        benchmark_results = self.benchmark.run_full_benchmark_suite()
        
        # Generate visualizations
        print("Generating visualizations...")
        self._generate_final_visualizations()
        
        print("Training completed!")
        
        return {
            'training_history': dict(self.training_history),
            'performance_metrics': dict(self.performance_metrics),
            'best_performance': best_avg_performance,
            'benchmark_results': benchmark_results
        }
    
    def _save_checkpoint(self, epoch: int, performance: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'performance': performance,
            'network_state': self.network.state_dict(),
            'training_history': dict(self.training_history),
            'performance_metrics': dict(self.performance_metrics)
        }
        
        # Save weight manager state
        self.network.weight_manager.save_state(f"checkpoint_epoch_{epoch}_weights.pt")
        
        # Save checkpoint
        torch.save(checkpoint, f"checkpoint_epoch_{epoch}.pt")
        print(f"Checkpoint saved at epoch {epoch} with performance {performance:.4f}")
    
    def _generate_final_visualizations(self):
        """Generate comprehensive visualizations."""
        try:
            # Training progress
            training_fig = self.visualizer.visualize_training_progress(
                self.training_history, "final_training_progress.html"
            )
            
            # Weight sharing analysis
            sharing_fig = self.visualizer.visualize_weight_sharing(
                "final_weight_sharing.html"
            )
            
            # Memory usage
            memory_fig = self.visualizer.visualize_memory_usage(
                "final_memory_usage.html"
            )
            
            # Topology evolution for each task
            for task_id in self.network.topology_controllers.keys():
                topology_fig = self.visualizer.visualize_topology_evolution(
                    task_id, f"final_topology_{task_id}.html"
                )
            
            # Real-time monitoring dashboard
            monitor_fig = self.monitor.get_current_dashboard()
            monitor_fig.write_html("final_monitoring_dashboard.html")
            
            print("Visualizations saved:")
            print("- final_training_progress.html")
            print("- final_weight_sharing.html")
            print("- final_memory_usage.html")
            print("- final_topology_*.html (for each task)")
            print("- final_monitoring_dashboard.html")
            
        except Exception as e:
            print(f"Warning: Could not generate some visualizations: {e}")


def main():
    """
    Main function demonstrating the complete Dynamic Connection Network system.
    """
    print("=" * 80)
    print("Dynamic Connection Network with Advanced Weight Sharing")
    print("Complete System Demonstration")
    print("=" * 80)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Define multi-domain tasks
    task_configs = [
        TaskConfig(
            task_id="image_classification",
            input_dim=32*32*3,  # Like CIFAR-10
            output_dim=10,
            hidden_dims=[512, 256, 128],
            task_type="classification",
            difficulty=0.6,
            data_size=5000
        ),
        TaskConfig(
            task_id="sentiment_analysis",
            input_dim=300,  # Word embedding dimension
            output_dim=3,   # Positive, Negative, Neutral
            hidden_dims=[256, 128, 64],
            task_type="classification",
            difficulty=0.7,
            data_size=3000
        ),
        TaskConfig(
            task_id="stock_prediction",
            input_dim=50,   # Technical indicators
            output_dim=1,   # Price change
            hidden_dims=[128, 64, 32],
            task_type="regression",
            difficulty=0.8,
            data_size=2000
        ),
        TaskConfig(
            task_id="time_series_classification",
            input_dim=100,  # Sequence length * features
            output_dim=5,   # 5 different pattern types
            hidden_dims=[256, 128, 64],
            task_type="sequence",
            difficulty=0.5,
            data_size=4000
        )
    ]
    
    print(f"Configured {len(task_configs)} tasks:")
    for config in task_configs:
        print(f"  - {config.task_id}: {config.task_type} "
              f"({config.input_dim}→{config.output_dim}, difficulty={config.difficulty})")
    
    # Create memory-optimized weight manager
    print("\nInitializing memory-optimized weight manager...")
    weight_manager = MemoryOptimizedWeightManager(
        memory_limit_mb=1024,
        enable_compression=True,
        enable_lazy_loading=True
    )
    
    # Create the multi-domain network
    print("Creating multi-domain dynamic network...")
    network = MultiDomainDynamicNetwork(
        task_configs, weight_manager, 
        shared_hidden_dims=[512, 256, 128]
    )
    
    print(f"Network created with:")
    print(f"  - {len(network.task_networks)} task-specific networks")
    print(f"  - {len(network.topology_controllers)} topology controllers")
    print(f"  - {len(weight_manager.banks)} shared weight banks")
    
    # Create training configuration
    training_config = TrainingConfig(
        strategy=TrainingStrategy.MULTI_TASK,
        learning_rate=0.001,
        topology_lr=0.01,
        batch_size=32,
        num_epochs=30,
        task_weights={
            "image_classification": 1.0,
            "sentiment_analysis": 1.2,
            "stock_prediction": 0.8,
            "time_series_classification": 1.0
        },
        gradient_clip_norm=1.0
    )
    
    # Create training system
    print("Initializing training system...")
    training_system = MultiDomainTrainingSystem(network, training_config)
    
    # Print initial memory statistics
    memory_stats = weight_manager.get_memory_stats()
    print(f"\nInitial memory usage:")
    print(f"  - Total banks: {memory_stats['total_banks']}")
    print(f"  - Memory usage: {memory_stats['total_memory_mb']:.2f} MB")
    print(f"  - Shared banks: {memory_stats['shared_banks_count']}")
    print(f"  - Sharing factor: {memory_stats['average_sharing_factor']:.2f}")
    
    # Run the complete training pipeline
    print("\n" + "="*50)
    print("STARTING TRAINING PIPELINE")
    print("="*50)
    
    try:
        results = training_system.train(
            num_epochs=training_config.num_epochs,
            save_checkpoints=True
        )
        
        # Print final results
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*50)
        
        print(f"Best average performance: {results['best_performance']:.4f}")
        
        print("\nFinal task performances:")
        for task_id in task_configs:
            task_id = task_id.task_id
            val_key = f"{task_id}_val"
            if val_key in results['performance_metrics']:
                final_perf = results['performance_metrics'][val_key][-1]
                print(f"  {task_id}: {final_perf:.4f}")
        
        # Print memory efficiency results
        print("\nMemory efficiency:")
        final_memory_stats = weight_manager.get_memory_stats()
        print(f"  - Final memory usage: {final_memory_stats['total_memory_mb']:.2f} MB")
        print(f"  - Memory utilization: {final_memory_stats['memory_utilization']:.1%}")
        print(f"  - Final sharing factor: {final_memory_stats['average_sharing_factor']:.2f}")
        
        # Print benchmark results
        benchmark_results = results['benchmark_results']
        print(f"\nBenchmark results:")
        for name, result in benchmark_results.items():
            print(f"  {name}:")
            print(f"    - Execution time: {result.execution_time:.4f}s")
            print(f"    - Throughput: {result.throughput:.2f} ops/s")
            print(f"    - Memory usage: {result.memory_usage/(1024*1024):.2f} MB")
        
        # Save final results
        print("\nSaving final results...")
        with open("training_results.json", 'w') as f:
            import json
            # Convert results to JSON-serializable format
            json_results = {
                'best_performance': results['best_performance'],
                'num_epochs': training_config.num_epochs,
                'num_tasks': len(task_configs),
                'final_memory_stats': final_memory_stats
            }
            json.dump(json_results, f, indent=2)
        
        print("Results saved to training_results.json")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files:")
    print("- Checkpoints: checkpoint_epoch_*.pt")
    print("- Weight states: checkpoint_epoch_*_weights.pt")
    print("- Visualizations: final_*.html")
    print("- Results: training_results.json")
    print("\nOpen the HTML files in a web browser to view interactive visualizations.")
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Complete system demonstration finished successfully!")
    else:
        print("\n❌ Demonstration encountered errors.")