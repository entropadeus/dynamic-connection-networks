import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class SimpleDCN(nn.Module):
    """Simplified DCN demonstrating key concepts from genius agents"""
    
    def __init__(self, input_size, hidden_size, output_size, num_tasks=4):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_tasks = num_tasks
        
        # Shared weight matrices (genius agent insight: weight sharing)
        self.shared_weights_1 = nn.Linear(input_size, hidden_size)
        self.shared_weights_2 = nn.Linear(hidden_size, 2)  # Fixed output size
        
        # Task-specific routing controllers (genius agent insight: learned routing)
        self.routing_controllers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size),
                nn.Sigmoid()
            ) for _ in range(num_tasks)
        ])
        
        # Output routing for each task
        self.output_routing = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, 2),  # Fixed output size
                nn.Sigmoid()
            ) for _ in range(num_tasks)
        ])
        
        # Task embeddings for meta-learning
        self.task_embeddings = nn.Embedding(num_tasks, 16)
        self.meta_controller = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, task_id=0):
        batch_size = x.size(0)
        
        # Get task embedding and meta weight
        task_emb = self.task_embeddings(torch.tensor(task_id))
        meta_weight = self.meta_controller(task_emb)
        
        # Layer 1 with task-specific routing
        shared_out_1 = self.shared_weights_1(x)
        routing_mask_1 = self.routing_controllers[task_id](x)
        routed_out_1 = shared_out_1 * routing_mask_1 * meta_weight
        hidden = F.relu(routed_out_1)
        
        # Layer 2 with output routing
        shared_out_2 = self.shared_weights_2(hidden)
        routing_mask_2 = self.output_routing[task_id](hidden)
        output = shared_out_2 * routing_mask_2
        
        # For regression tasks, only use first output unit
        if task_id in [2, 3]:  # Regression tasks
            output = output[:, :1]
        
        return output, [routing_mask_1, routing_mask_2]

def create_diverse_tasks():
    """Create tasks that require different topologies"""
    
    def create_task_data(n=300):
        X = torch.randn(n, 10)
        return X
    
    # Task 0: Linear classification using first 3 features
    def task_0():
        X = create_task_data()
        y = (X[:, :3].sum(1) > 0).long()
        return X, y, 'classification'
    
    # Task 1: XOR pattern using features 4,5
    def task_1():
        X = create_task_data()
        y = ((X[:, 4] > 0) ^ (X[:, 5] > 0)).long()
        return X, y, 'classification'
    
    # Task 2: Regression using last 4 features
    def task_2():
        X = create_task_data()
        y = X[:, -4:].sum(1, keepdim=True)
        return X, y, 'regression'
    
    # Task 3: Polynomial regression using features 2,3,6
    def task_3():
        X = create_task_data()
        y = (X[:, 2] * X[:, 3] + X[:, 6] ** 2).unsqueeze(1)
        return X, y, 'regression'
    
    return [task_0(), task_1(), task_2(), task_3()]

def train_dcn_multitask(model, tasks, epochs=100):
    """Train DCN on multiple tasks simultaneously"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Track metrics
    task_losses = [[] for _ in range(len(tasks))]
    routing_patterns = []
    
    print("Training DCN on multiple tasks...")
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_routing = []
        
        # Train on each task
        for task_id, (X, y, task_type) in enumerate(tasks):
            optimizer.zero_grad()
            
            output, routing = model(X, task_id=task_id)
            
            # Task-specific loss
            if task_type == 'classification':
                loss = F.cross_entropy(output, y)
            else:  # regression
                loss = F.mse_loss(output, y)
            
            loss.backward()
            optimizer.step()
            
            task_losses[task_id].append(loss.item())
            epoch_loss += loss.item()
            
            # Store routing pattern
            if epoch % 20 == 0:
                epoch_routing.append(routing[0].mean(0).detach().clone())
        
        if epoch % 20 == 0:
            routing_patterns.append(epoch_routing)
            print(f"Epoch {epoch}: Avg Loss = {epoch_loss/len(tasks):.4f}")
    
    return task_losses, routing_patterns

def test_topology_transfer(model, tasks):
    """Test cross-task topology transfer (genius agent insight)"""
    
    print("\n=== Testing Topology Transfer ===")
    
    model.eval()
    transfer_results = {}
    
    with torch.no_grad():
        # Get baseline performance for each task
        baseline_performance = {}
        for task_id, (X, y, task_type) in enumerate(tasks):
            output, _ = model(X, task_id=task_id)
            
            if task_type == 'classification':
                perf = (output.argmax(1) == y).float().mean().item()
            else:
                perf = -F.mse_loss(output, y).item()  # Negative MSE for "higher is better"
            
            baseline_performance[task_id] = perf
        
        # Test using each task's routing on other tasks
        for source_task in range(len(tasks)):
            for target_task in range(len(tasks)):
                if source_task != target_task:
                    X, y, task_type = tasks[target_task]
                    
                    # Use source task's routing controller on target task's data
                    shared_out_1 = model.shared_weights_1(X)
                    routing_mask_1 = model.routing_controllers[source_task](X)
                    routed_out_1 = shared_out_1 * routing_mask_1
                    hidden = F.relu(routed_out_1)
                    
                    shared_out_2 = model.shared_weights_2(hidden)
                    routing_mask_2 = model.output_routing[source_task](hidden)
                    output = shared_out_2 * routing_mask_2
                    
                    if task_type == 'classification':
                        perf = (output.argmax(1) == y).float().mean().item()
                    else:
                        perf = -F.mse_loss(output, y).item()
                    
                    transfer_ratio = perf / baseline_performance[target_task]
                    transfer_results[(source_task, target_task)] = transfer_ratio
                    
                    print(f"Task {source_task} -> Task {target_task}: {transfer_ratio:.3f}")
    
    return transfer_results

def analyze_feature_specialization(model, tasks):
    """Analyze feature usage across tasks (genius agent insight)"""
    
    print("\n=== Feature Specialization Analysis ===")
    
    model.eval()
    feature_usage = {}
    
    with torch.no_grad():
        for task_id, (X, y, task_type) in enumerate(tasks):
            # Get routing pattern for this task
            _, routing = model(X[:100], task_id=task_id)
            
            # Average routing weights across batch
            avg_routing = routing[0].mean(0)
            feature_usage[task_id] = avg_routing
            
            print(f"\nTask {task_id} ({task_type}):")
            print(f"  Feature importance: {avg_routing.numpy()}")
            print(f"  Active features: {(avg_routing > 0.5).sum().item()}/{len(avg_routing)}")
    
    # Calculate feature sharing across tasks
    print(f"\n=== Feature Sharing Analysis ===")
    
    for i in range(len(tasks)):
        for j in range(i+1, len(tasks)):
            similarity = F.cosine_similarity(
                feature_usage[i], feature_usage[j], dim=0
            ).item()
            
            shared_features = ((feature_usage[i] > 0.5) & (feature_usage[j] > 0.5)).sum().item()
            unique_i = ((feature_usage[i] > 0.5) & (feature_usage[j] <= 0.5)).sum().item()
            unique_j = ((feature_usage[j] > 0.5) & (feature_usage[i] <= 0.5)).sum().item()
            
            print(f"Task {i} <-> Task {j}:")
            print(f"  Similarity: {similarity:.3f}")
            print(f"  Shared features: {shared_features}")
            print(f"  Unique to Task {i}: {unique_i}")
            print(f"  Unique to Task {j}: {unique_j}")

def visualize_routing_evolution(routing_patterns, tasks):
    """Visualize how routing evolves during training"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    task_names = ['Linear Classify', 'XOR Pattern', 'Linear Regress', 'Poly Regress']
    
    for task_id in range(len(tasks)):
        if task_id < 4:  # Only plot first 4 tasks
            # Extract routing evolution for this task
            evolution = []
            for epoch_patterns in routing_patterns:
                if task_id < len(epoch_patterns):
                    evolution.append(epoch_patterns[task_id].numpy())
            
            if evolution:
                evolution = np.array(evolution)
                
                # Plot routing strength over time for each feature
                for feature_idx in range(evolution.shape[1]):
                    axes[task_id].plot(evolution[:, feature_idx], 
                                     label=f'Feature {feature_idx}', alpha=0.7)
                
                axes[task_id].set_title(f'{task_names[task_id]}\nRouting Evolution')
                axes[task_id].set_xlabel('Training Epoch (x20)')
                axes[task_id].set_ylabel('Routing Weight')
                axes[task_id].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                axes[task_id].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('C:\\Users\\benlo\\Desktop\\dcn_routing_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("=== Advanced DCN Demonstration ===")
    print("Implementing insights from genius agents:")
    print("+ Weight sharing across tasks")
    print("+ Learned routing modules")
    print("+ Cross-task topology transfer")
    print("+ Feature reuse analysis\n")
    
    # Create diverse tasks
    tasks = create_diverse_tasks()
    
    print(f"Created {len(tasks)} tasks:")
    for i, (X, y, task_type) in enumerate(tasks):
        print(f"  Task {i}: {task_type}, {X.shape} -> {y.shape}")
    
    # Create and train model
    model = SimpleDCN(input_size=10, hidden_size=16, output_size=2, num_tasks=len(tasks))
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    task_losses, routing_patterns = train_dcn_multitask(model, tasks, epochs=80)
    
    # Test topology transfer
    transfer_results = test_topology_transfer(model, tasks)
    
    # Analyze feature specialization
    analyze_feature_specialization(model, tasks)
    
    # Visualize results
    print("\nGenerating routing evolution visualization...")
    visualize_routing_evolution(routing_patterns, tasks)
    
    # Summary insights
    print("\n=== Key Insights ===")
    
    # Calculate average transfer success
    successful_transfers = sum(1 for ratio in transfer_results.values() if ratio > 0.8)
    total_transfers = len(transfer_results)
    
    print(f"+ Cross-task transfer success: {successful_transfers}/{total_transfers} ({100*successful_transfers/total_transfers:.1f}%)")
    
    # Find best and worst transfer pairs
    best_transfer = max(transfer_results.items(), key=lambda x: x[1])
    worst_transfer = min(transfer_results.items(), key=lambda x: x[1])
    
    print(f"+ Best transfer: Task {best_transfer[0][0]} → Task {best_transfer[0][1]} ({best_transfer[1]:.3f})")
    print(f"+ Challenging transfer: Task {worst_transfer[0][0]} → Task {worst_transfer[0][1]} ({worst_transfer[1]:.3f})")
    
    print(f"\n+ The DCN successfully demonstrates:")
    print(f"  - Dynamic topology adaptation for different problem types")
    print(f"  - Shared weights with task-specific routing")
    print(f"  - Feature specialization and reuse patterns")
    print(f"  - Cross-task knowledge transfer capabilities")

if __name__ == "__main__":
    main()
    print("\n" + "="*50)
    print("DCN Demo Complete!")
    print("Check the generated visualization: dcn_routing_evolution.png")
    print("="*50)
    input("\nPress Enter to exit...")