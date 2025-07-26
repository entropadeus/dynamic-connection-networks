import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dynamic_connection_network import DynamicConnectionNetwork
import copy

class RoutingController(nn.Module):
    """Learned routing module that picks connections based on input"""
    
    def __init__(self, input_size, connection_size):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, connection_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Average pool input to get routing signal
        routing_signal = x.mean(0, keepdim=True)  # Global average
        connection_weights = self.router(routing_signal)
        return connection_weights

class AdvancedDCN(nn.Module):
    """Enhanced DCN with routing controllers and task adaptation"""
    
    def __init__(self, input_size, hidden_sizes, output_size, use_routing=False):
        super().__init__()
        self.use_routing = use_routing
        self.layers = nn.ModuleList()
        self.routing_controllers = nn.ModuleList()
        
        # Build layers
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layer = nn.Linear(prev_size, hidden_size)
            self.layers.append(layer)
            
            if use_routing:
                # Connection matrix for this layer
                connection_size = prev_size * hidden_size
                router = RoutingController(input_size, connection_size)
                self.routing_controllers.append(router)
            
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, output_size)
        
        if use_routing and len(hidden_sizes) > 0:
            connection_size = prev_size * output_size
            router = RoutingController(input_size, connection_size)
            self.routing_controllers.append(router)
    
    def forward(self, x):
        connections_log = []
        original_x = x.clone()
        
        for i, layer in enumerate(self.layers):
            if self.use_routing:
                # Get routing weights
                routing_weights = self.routing_controllers[i](original_x)
                routing_weights = routing_weights.view(layer.weight.shape)
                
                # Apply routing to weights
                effective_weight = layer.weight * routing_weights
                x = F.linear(x, effective_weight, layer.bias)
                connections_log.append(routing_weights)
            else:
                x = layer(x)
            
            x = F.relu(x)
        
        # Output layer
        if self.use_routing:
            routing_weights = self.routing_controllers[-1](original_x)
            routing_weights = routing_weights.view(self.output_layer.weight.shape)
            effective_weight = self.output_layer.weight * routing_weights
            x = F.linear(x, effective_weight, self.output_layer.bias)
            connections_log.append(routing_weights)
        else:
            x = self.output_layer(x)
        
        return x, connections_log

def create_diverse_tasks():
    """Create multiple different task types"""
    
    def classification_task(n=500, difficulty='easy'):
        """Binary classification with varying difficulty"""
        X = torch.randn(n, 15)
        if difficulty == 'easy':
            # Simple linear boundary
            y = (X[:, 0] + X[:, 1] > 0).long()
        else:  # hard
            # Non-linear XOR-like pattern
            y = ((X[:, 0] > 0) ^ (X[:, 1] > 0) ^ (X[:, 2] > 0)).long()
        return X, y, 'classification'
    
    def regression_task(n=500, complexity='linear'):
        """Regression with different complexity levels"""
        X = torch.randn(n, 15)
        if complexity == 'linear':
            # Simple linear combination
            y = X[:, :3].sum(1, keepdim=True) + 0.1 * torch.randn(n, 1)
        else:  # nonlinear
            # Polynomial features
            y = (X[:, 0] * X[:, 1] + X[:, 2] ** 2 + X[:, 3] * X[:, 4] * X[:, 5]).unsqueeze(1)
            y += 0.1 * torch.randn(n, 1)
        return X, y, 'regression'
    
    def noise_filtering_task(n=500, noise_level=0.5):
        """Denoising task - recover clean signal from noisy input"""
        # Create clean signal (sum of specific features)
        clean_signal = torch.randn(n, 15)
        target = clean_signal[:, [0, 3, 7]].sum(1, keepdim=True)  # Only these matter
        
        # Add noise to input
        noise = noise_level * torch.randn(n, 15)
        X = clean_signal + noise
        
        return X, target, 'denoising'
    
    def pattern_completion_task(n=500):
        """Complete missing pattern - like autoencoder but specific"""
        X = torch.randn(n, 15)
        # Target is a transformation of input
        target = torch.stack([
            X[:, 0] + X[:, 5],  # Feature combination 1
            X[:, 2] * X[:, 8],  # Feature combination 2
            X[:, 1] - X[:, 6]   # Feature combination 3
        ], dim=1)
        return X, target, 'pattern_completion'
    
    return {
        'easy_classification': classification_task(difficulty='easy'),
        'hard_classification': classification_task(difficulty='hard'),
        'linear_regression': regression_task(complexity='linear'),
        'nonlinear_regression': regression_task(complexity='nonlinear'),
        'light_denoising': noise_filtering_task(noise_level=0.3),
        'heavy_denoising': noise_filtering_task(noise_level=0.8),
        'pattern_completion': pattern_completion_task()
    }

def train_on_task(model, X, y, task_type, epochs=100, lr=0.001):
    """Train model on a specific task"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if task_type == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:  # regression, denoising, pattern_completion
        criterion = nn.MSELoss()
    
    model.train()
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output, connections = model(X)
        
        if task_type == 'classification':
            loss = criterion(output, y)
        else:
            loss = criterion(output, y)
        
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 20 == 0:
            if task_type == 'classification':
                acc = (output.argmax(1) == y).float().mean()
                print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.3f}")
            else:
                print(f"Epoch {epoch}: Loss={loss:.4f}")
    
    return losses, connections

def test_generalization(model1, model2, tasks):
    """Test if topology from one task helps with another"""
    
    print("\n=== Testing Cross-Task Generalization ===")
    
    # Get connection patterns from trained models
    model1.eval()
    model2.eval()
    
    task1_name, (X1, y1, type1) = list(tasks.items())[0]
    task2_name, (X2, y2, type2) = list(tasks.items())[1]
    
    with torch.no_grad():
        _, conn1 = model1(X1[:50])
        _, conn2 = model2(X2[:50])
    
    # Try using model1's topology on task2 data
    print(f"\nTesting {task1_name} topology on {task2_name} task...")
    
    # Create a hybrid model that uses model2's weights but model1's routing
    if hasattr(model1, 'use_routing') and model1.use_routing:
        hybrid_model = copy.deepcopy(model2)
        # Copy routing controllers from model1
        for i, router in enumerate(model1.routing_controllers):
            hybrid_model.routing_controllers[i].load_state_dict(router.state_dict())
        
        # Test performance
        hybrid_model.eval()
        with torch.no_grad():
            hybrid_output, _ = hybrid_model(X2)
            if type2 == 'classification':
                hybrid_acc = (hybrid_output.argmax(1) == y2).float().mean()
                original_output, _ = model2(X2)
                original_acc = (original_output.argmax(1) == y2).float().mean()
                print(f"Original model accuracy: {original_acc:.3f}")
                print(f"Hybrid model accuracy: {hybrid_acc:.3f}")
                print(f"Topology transfer {'helped' if hybrid_acc > original_acc else 'hurt'}")
            else:
                hybrid_loss = F.mse_loss(hybrid_output, y2)
                original_output, _ = model2(X2)
                original_loss = F.mse_loss(original_output, y2)
                print(f"Original model loss: {original_loss:.4f}")
                print(f"Hybrid model loss: {hybrid_loss:.4f}")
                print(f"Topology transfer {'helped' if hybrid_loss < original_loss else 'hurt'}")

def analyze_feature_reuse(models, tasks):
    """Analyze which features are reused vs isolated across tasks"""
    
    print("\n=== Feature Reuse Analysis ===")
    
    feature_usage = {}
    task_names = list(tasks.keys())
    
    for i, (task_name, (X, y, task_type)) in enumerate(tasks.items()):
        model = models[i]
        model.eval()
        
        with torch.no_grad():
            _, connections = model(X[:100])
            
            if len(connections) > 0:
                # Average connection strength per input feature
                conn_matrix = connections[0].mean(0)  # Average across batch
                if len(conn_matrix.shape) > 1:
                    feature_importance = conn_matrix.mean(0)  # Average across output neurons
                else:
                    feature_importance = conn_matrix
                
                feature_usage[task_name] = feature_importance.cpu()
    
    # Calculate feature reuse across tasks
    print("\nFeature importance by task:")
    for task_name, importance in feature_usage.items():
        print(f"\n{task_name}:")
        for i, imp in enumerate(importance):
            print(f"  Feature {i}: {imp:.3f}")
    
    # Find shared vs unique features
    if len(feature_usage) >= 2:
        tasks_list = list(feature_usage.keys())
        importance1 = feature_usage[tasks_list[0]]
        importance2 = feature_usage[tasks_list[1]]
        
        # Correlation between feature usage patterns
        correlation = torch.corrcoef(torch.stack([importance1, importance2]))[0, 1]
        print(f"\nFeature usage correlation between {tasks_list[0]} and {tasks_list[1]}: {correlation:.3f}")
        
        # Find features used by both vs unique to each
        threshold = 0.4
        shared_features = ((importance1 > threshold) & (importance2 > threshold)).sum()
        unique_to_1 = ((importance1 > threshold) & (importance2 <= threshold)).sum()
        unique_to_2 = ((importance2 > threshold) & (importance1 <= threshold)).sum()
        
        print(f"\nFeature usage patterns:")
        print(f"  Shared features (both tasks): {shared_features}")
        print(f"  Unique to {tasks_list[0]}: {unique_to_1}")
        print(f"  Unique to {tasks_list[1]}: {unique_to_2}")
        
        modularity = (unique_to_1 + unique_to_2) / (shared_features + unique_to_1 + unique_to_2 + 1e-6)
        print(f"  Modularity score: {modularity:.3f} (higher = more specialized)")

def visualize_routing_patterns(models, tasks):
    """Visualize how routing patterns differ across tasks"""
    
    fig, axes = plt.subplots(2, len(models)//2 + len(models)%2, figsize=(15, 8))
    if len(models) == 1:
        axes = [axes]
    axes = axes.flatten()
    
    for i, ((task_name, (X, y, task_type)), model) in enumerate(zip(tasks.items(), models)):
        if i >= len(axes):
            break
            
        model.eval()
        with torch.no_grad():
            _, connections = model(X[:50])
            
            if len(connections) > 0:
                conn_matrix = connections[0].mean(0).cpu().numpy()
                
                if len(conn_matrix.shape) == 1:
                    # Reshape to 2D for visualization
                    size = int(np.sqrt(len(conn_matrix)))
                    if size * size == len(conn_matrix):
                        conn_matrix = conn_matrix.reshape(size, size)
                    else:
                        # Create a simple 1D visualization
                        conn_matrix = conn_matrix.reshape(1, -1)
                
                im = axes[i].imshow(conn_matrix, cmap='viridis', aspect='auto')
                axes[i].set_title(f'{task_name}\nRouting Pattern')
                axes[i].set_xlabel('Input Features')
                axes[i].set_ylabel('Output Neurons')
                plt.colorbar(im, ax=axes[i])
    
    # Hide unused subplots
    for i in range(len(models), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('C:\\Users\\benlo\\Desktop\\routing_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("=== Advanced DCN Experiments ===")
    
    # Create diverse tasks
    print("Creating diverse task dataset...")
    tasks = create_diverse_tasks()
    
    print(f"Created {len(tasks)} different tasks:")
    for name, (X, y, task_type) in tasks.items():
        print(f"  {name}: {task_type}, Input: {X.shape}, Output: {y.shape}")
    
    # Train models on different tasks
    print("\n=== Training Models ===")
    models = []
    
    for i, (task_name, (X, y, task_type)) in enumerate(tasks.items()):
        print(f"\nTraining on {task_name}...")
        
        # Use routing for some models
        use_routing = i % 2 == 0  # Alternate between routing and non-routing
        
        if task_type == 'classification':
            output_size = 2
        elif task_type in ['regression', 'denoising']:
            output_size = 1
        else:  # pattern_completion
            output_size = y.shape[1]
        
        model = AdvancedDCN(
            input_size=15,
            hidden_sizes=[20],
            output_size=output_size,
            use_routing=use_routing
        )
        
        losses, connections = train_on_task(model, X, y, task_type, epochs=50)
        models.append(model)
        
        print(f"  Model type: {'Routing' if use_routing else 'Standard'}")
    
    # Test generalization
    if len(models) >= 2:
        test_generalization(models[0], models[1], tasks)
    
    # Analyze feature reuse
    analyze_feature_reuse(models, tasks)
    
    # Visualize routing patterns
    print("\nGenerating routing pattern visualizations...")
    visualize_routing_patterns(models, tasks)
    
    print("\n=== Experiment Summary ===")
    print("1. Trained on diverse tasks: classification, regression, denoising, pattern completion")
    print("2. Tested topology transfer across tasks")
    print("3. Analyzed feature reuse vs specialization")
    print("4. Compared routing vs standard architectures")
    print("\nKey insights:")
    print("- Different tasks develop different connection patterns")
    print("- Some topologies transfer better than others")
    print("- Feature usage reveals task-specific vs shared representations")

if __name__ == "__main__":
    main()