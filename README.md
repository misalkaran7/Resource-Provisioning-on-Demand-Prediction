# Resource Provisioning Based on Demand Prediction

This repository contains an advanced algorithmic testing and simulation environment for cloud resource provisioning, focusing on predictive scaling and fragmentation-aware scheduling. It models industrial-grade solutions used in environments like AWS ECS and Kubernetes.

## Core Components

- **`forecast_engine.py`**: Simulates a Time-Series Foundation Model (like Time-MoE) providing demand forecasts. It implements an industry-standard **safety margin** to intentionally over-provision and prevent SLA violations due to model Mean Absolute Error (MAE).
- **`distribution_mapper.py`**: Converts aggregate demand forecasts into expected pod size probability distributions. It utilizes an **Exponential Moving Average (EMA)** to track demand trends, ensuring smooth and continuous transitions of workload expectations without thrashing.
- **`fgd_scheduler.py`**: Implements a **Fragmentation Gradient Descent (FGD)** scheduler. It features a robust fragmentation calculation using a **Best-Fit Decreasing (BFD)** greedy bin-packing simulation and incorporates a node scoring bias modeled after Kubernetes' `MostAllocated` strategy to tightly consolidate workloads.
- **`simulator.py`**: The main execution engine that generates synthetic trace demands and sequences of incoming pods. It orchestrates the forecast engine, mapper, and scheduler to evaluate predictive provisioning against baseline approaches.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/misalkaran7/Resource-Provisioning-on-Demand-Prediction.git
   ```
2. Navigate to the testing directory where the simulation resides:
   ```bash
   cd testing
   ```
3. Run the simulation to see the impact of predictive FGD compared to a static baseline:
   ```bash
   python simulator.py
   ```

## Background & Research

For a detailed breakdown of the industry research, algorithm choices, and improvements over simplistic implementations, please read the [Research Log](research_log.md).
