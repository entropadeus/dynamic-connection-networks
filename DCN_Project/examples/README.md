# DCN Examples

This directory contains working examples and demonstrations of Dynamic Connection Networks.

## Quick Start Examples

### 1. Simple Demo (`simple_dcn_demo.py`)
Basic proof of concept showing DCN topology adaptation.
```bash
python simple_dcn_demo.py
```

### 2. Working Demo (`working_dcn_demo.py`) ⭐ **RECOMMENDED**
Complete demonstration with multi-task learning, cross-task transfer, and analysis.
```bash
python working_dcn_demo.py
```

### 3. Enhanced System (`enhanced_dcn_system.py`)
Advanced version with meta-learning and shared weight banks.
```bash
python enhanced_dcn_system.py
```

### 4. Complete System (`complete_system_example.py`)
Full multi-domain system with comprehensive features.
```bash
python complete_system_example.py
```

## What Each Example Shows

- **Dynamic topology adaptation** for different problem types
- **Cross-task knowledge transfer** 
- **Learned routing mechanisms**
- **Weight sharing with topology changes**
- **Feature specialization analysis**
- **Real-time visualization** of network adaptation

## Requirements

Make sure you have the required dependencies:
```bash
pip install -r ../requirements.txt
```

## Generated Outputs

Examples will create visualizations in the `../visualizations/` directory showing how network topology evolves during training.