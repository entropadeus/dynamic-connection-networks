# DCN Visualizations

Generated visualizations and charts from Dynamic Connection Network experiments.

## Visualization Files

### Network Topology Evolution
- `dcn_routing_evolution.png` - How routing patterns change during training
- `dcn_topology_evolution.png` - Network structure adaptation over time

## Visualization Types

### 1. Routing Pattern Evolution
Shows how different tasks develop different connection patterns:
- **Task-specific routing** - Unique patterns for each problem type
- **Feature specialization** - Which features become important
- **Training dynamics** - How patterns emerge over time

### 2. Cross-Task Transfer Analysis
Visualizes knowledge transfer between tasks:
- **Similarity matrices** - How tasks relate to each other
- **Transfer success rates** - Which topologies transfer well
- **Feature reuse patterns** - Shared vs specialized features

### 3. Performance Metrics
Charts showing system performance:
- **Accuracy trends** - Learning progress across tasks
- **Memory usage** - Resource efficiency over time
- **Convergence speed** - How quickly networks adapt

## Interpreting Visualizations

### Routing Evolution Charts
- **X-axis**: Training epochs
- **Y-axis**: Connection strength (0-1)
- **Different colors**: Different input features
- **Pattern changes**: Show adaptation to task requirements

### Transfer Analysis Heatmaps
- **Bright colors**: Successful topology transfer
- **Dark colors**: Poor transfer (tasks too different)
- **Diagonal patterns**: Task similarity structure

## Generating New Visualizations

Run any example or experiment file to generate new visualizations:
```bash
cd ../examples
python working_dcn_demo.py
```

Charts will be automatically saved to this directory.