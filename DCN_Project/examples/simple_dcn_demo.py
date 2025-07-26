import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dynamic_connection_network import DynamicConnectionNetwork

def create_simple_tasks():
    """Create simple test tasks to demonstrate topology adaptation"""
    
    # Task 1: First 3 features matter
    def task1_data(n=200):
        X = torch.randn(n, 10)
        y = (X[:, :3].sum(1) > 0).long()
        return X, y
    
    # Task 2: Last 3 features matter  
    def task2_data(n=200):
        X = torch.randn(n, 10)
        y = (X[:, -3:].sum(1) > 0).long()
        return X, y
    
    return task1_data(), task2_data()

def quick_train(model, X, y, epochs=50):
    """Quick training function"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output, connections = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            acc = (output.argmax(1) == y).float().mean()
            print(f"Epoch {epoch}: Loss={loss:.3f}, Acc={acc:.3f}")
    
    return connections

def main():
    print("=== Dynamic Connection Network Demo ===")
    
    # Create tasks
    (X1, y1), (X2, y2) = create_simple_tasks()
    
    print(f"\nTask 1: Uses features 0-2 (first 3)")
    print(f"Task 2: Uses features 7-9 (last 3)")
    print(f"Data shape: {X1.shape}")
    
    # Train model on Task 1
    print("\n--- Training on Task 1 ---")
    model1 = DynamicConnectionNetwork(input_size=10, hidden_sizes=[16], output_size=2, sparsity=0.3)
    connections1 = quick_train(model1, X1, y1)
    
    # Train model on Task 2  
    print("\n--- Training on Task 2 ---")
    model2 = DynamicConnectionNetwork(input_size=10, hidden_sizes=[16], output_size=2, sparsity=0.3)
    connections2 = quick_train(model2, X2, y2)
    
    # Compare topologies
    print("\n=== Topology Comparison ===")
    
    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        # Get final connection patterns
        _, final_conn1 = model1(X1[:10])
        _, final_conn2 = model2(X2[:10])
        
        conn1_avg = final_conn1[0].mean(0)  # Average across batch
        conn2_avg = final_conn2[0].mean(0)  # Average across batch
        
        print(f"Task 1 connection matrix shape: {conn1_avg.shape}")
        print(f"Task 2 connection matrix shape: {conn2_avg.shape}")
        
        print(f"Task 1 connection strength by input feature:")
        for i in range(10):
            strength = conn1_avg[i].item()
            print(f"  Feature {i}: {strength:.3f}")
        
        print(f"\nTask 2 connection strength by input feature:")
        for i in range(10):
            strength = conn2_avg[i].item()
            print(f"  Feature {i}: {strength:.3f}")
        
        # Highlight the key differences
        print(f"\n=== Key Insights ===")
        task1_early_features = conn1_avg[:3].mean().item()
        task1_late_features = conn1_avg[-3:].mean().item()
        
        task2_early_features = conn2_avg[:3].mean().item()
        task2_late_features = conn2_avg[-3:].mean().item()
        
        print(f"Task 1 (should focus on early features):")
        print(f"  Early features (0-2): {task1_early_features:.3f}")
        print(f"  Late features (7-9): {task1_late_features:.3f}")
        
        print(f"\nTask 2 (should focus on late features):")
        print(f"  Early features (0-2): {task2_early_features:.3f}")
        print(f"  Late features (7-9): {task2_late_features:.3f}")
        
        # Success metric
        task1_correct_focus = task1_early_features > task1_late_features
        task2_correct_focus = task2_late_features > task2_early_features
        
        print(f"\n=== Results ===")
        print(f"Task 1 correctly focuses on early features: {'YES' if task1_correct_focus else 'NO'}")
        print(f"Task 2 correctly focuses on late features: {'YES' if task2_correct_focus else 'NO'}")
        
        if task1_correct_focus and task2_correct_focus:
            print("\nSUCCESS: Network learned different topologies for different tasks!")
        else:
            print("\nPartial success - Network shows some adaptation but may need more training")

if __name__ == "__main__":
    main()