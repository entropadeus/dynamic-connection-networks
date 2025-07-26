import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DynamicConnectionLayer(nn.Module):
    def __init__(self, input_size, output_size, sparsity=0.1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.sparsity = sparsity
        
        # Weight matrix for connection strengths
        self.weights = nn.Parameter(torch.randn(output_size, input_size) * 0.01)
        
        # Connection matrix - learnable binary connections
        # Use Gumbel softmax for differentiable discrete sampling
        self.connection_logits = nn.Parameter(torch.randn(output_size, input_size))
        
        # Bias terms
        self.bias = nn.Parameter(torch.zeros(output_size))
        
    def forward(self, x, temperature=1.0, hard=False):
        # Generate dynamic connections using Gumbel softmax
        # This makes discrete connections differentiable
        connection_probs = torch.sigmoid(self.connection_logits / temperature)
        
        if hard and self.training:
            # During training, use straight-through estimator
            connections = (connection_probs > 0.5).float()
            connections = connections - connection_probs.detach() + connection_probs
        else:
            connections = connection_probs
        
        # Apply sparsity constraint
        if self.training:
            # Encourage sparsity by only keeping top connections
            flat_connections = connections.view(-1)
            k = int(len(flat_connections) * self.sparsity)
            _, top_indices = torch.topk(flat_connections, k)
            sparse_mask = torch.zeros_like(flat_connections)
            sparse_mask[top_indices] = 1
            connections = sparse_mask.view(connections.shape)
        
        # Combine weights and connections
        effective_weights = self.weights * connections
        
        # Standard linear transformation
        output = F.linear(x, effective_weights, self.bias)
        
        return output, connections

class DynamicConnectionNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, sparsity=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Build layers
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(DynamicConnectionLayer(prev_size, hidden_size, sparsity))
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = DynamicConnectionLayer(prev_size, output_size, sparsity)
        
        # Temperature for Gumbel softmax (annealed during training)
        self.register_buffer('temperature', torch.tensor(5.0))
        
    def forward(self, x):
        connections_log = []
        
        # Forward through dynamic layers
        for layer in self.layers:
            x, connections = layer(x, self.temperature, hard=True)
            x = F.relu(x)
            connections_log.append(connections)
        
        # Output layer
        x, connections = self.output_layer(x, self.temperature, hard=True)
        connections_log.append(connections)
        
        return x, connections_log
    
    def anneal_temperature(self, step, total_steps):
        # Gradually reduce temperature for sharper connections
        min_temp = 0.1
        max_temp = 5.0
        new_temp = max_temp * (min_temp / max_temp) ** (step / total_steps)
        self.temperature.fill_(new_temp)
    
    def get_topology_stats(self, connections_log):
        stats = {}
        for i, connections in enumerate(connections_log):
            active_connections = (connections > 0.5).float().sum().item()
            total_connections = connections.numel()
            stats[f'layer_{i}_sparsity'] = 1.0 - (active_connections / total_connections)
            stats[f'layer_{i}_active'] = active_connections
        return stats

def sparsity_loss(connections_log, target_sparsity=0.1):
    """Regularization loss to encourage sparsity"""
    total_loss = 0
    for connections in connections_log:
        current_sparsity = 1.0 - connections.mean()
        total_loss += (current_sparsity - target_sparsity) ** 2
    return total_loss / len(connections_log)

def train_dcn(model, train_loader, epochs=100, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_sparsity_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            output, connections_log = model(data)
            
            # Main loss
            main_loss = criterion(output, target)
            
            # Sparsity regularization
            sparse_loss = sparsity_loss(connections_log, target_sparsity=0.1)
            
            # Combined loss
            loss = main_loss + 0.01 * sparse_loss
            
            loss.backward()
            optimizer.step()
            
            # Anneal temperature
            total_steps = epochs * len(train_loader)
            current_step = epoch * len(train_loader) + batch_idx
            model.anneal_temperature(current_step, total_steps)
            
            total_loss += main_loss.item()
            total_sparsity_loss += sparse_loss.item()
        
        if epoch % 10 == 0:
            stats = model.get_topology_stats(connections_log)
            print(f'Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}, '
                  f'Sparsity Loss={total_sparsity_loss/len(train_loader):.4f}, '
                  f'Temp={model.temperature:.3f}')
            print(f'Topology: {stats}')

if __name__ == "__main__":
    # Example usage
    model = DynamicConnectionNetwork(
        input_size=784,  # MNIST
        hidden_sizes=[256, 128],
        output_size=10,
        sparsity=0.1
    )
    
    # Create dummy data for testing
    dummy_data = torch.randn(32, 784)
    output, connections = model(dummy_data)
    
    print(f"Output shape: {output.shape}")
    print(f"Number of connection matrices: {len(connections)}")
    
    # Show dynamic topology
    stats = model.get_topology_stats(connections)
    print("Initial topology stats:", stats)