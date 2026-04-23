# Research Log: Resource Provisioning Based on Demand Prediction

## 1. Industry Research & Current Landscape
Resource provisioning in modern cloud environments (like AWS ECS, Kubernetes) relies on a combination of reactive scaling and proactive, demand-prediction-based (predictive) scaling.

### 1.1 How Algorithms Work Now
- **Predictive Scaling Mechanisms**: Platforms like AWS Auto Scaling and Karpenter utilize historical time-series data (e.g., using models like Prophet, LSTM, or Time-Series Foundation Models) to predict upcoming demand spikes. 
- **Safety Buffers**: Because predictions carry intrinsic errors (Mean Absolute Error - MAE), industrial algorithms invariably apply an over-provisioning **Safety Margin** (e.g., Target Tracking buffers) to prevent SLA violations and "cold-start" latencies when sudden traffic exceeds forecasts.
- **Continuous Distribution Shifts**: Workload shapes rarely jump discretely. Industry standard algorithms use smoothing functions like **Exponential Moving Average (EMA)** to transition the expected pod sizes and cluster resource allocations continuously.
- **Fragmentation-Aware Bin Packing**: On Kubernetes (particularly with GPU scheduling like Volcano or native `MostAllocated` scoring plugin), nodes are scheduled using greedy bin-packing variants (e.g., Best-Fit Decreasing) rather than naive capability checks. The goal is to maximize the utilization of already-active nodes and leave larger contiguous blocks of compute empty for large-scale distributed tasks (like LLM training).

---

## 2. Evaluation of Existing Implementation & Shortcomings
Before our enhancements, the existing codebase in the `testing` directory had several fundamental shortcomings typical of an academic/prototype setup:

1. **`forecast_engine.py`**:
   - **Shortcoming**: It merely simulated the model with a fixed Gaussian noise based on the theoretical MAE of a Time-MoE model. There was no concept of a safety buffer, meaning under-prediction would immediately lead to pod rejection in a real-world scenario.
2. **`distribution_mapper.py`**:
   - **Shortcoming**: Used a brittle, hardcoded step function (shifting probability rigidly if the demand derivative exceeded a flat `0.05` threshold). This causes "thrashing" during fluctuating but generally stable workloads.
3. **`fgd_scheduler.py`**:
   - **Shortcoming**: The `calculate_fragmentation` method used a highly simplistic evaluation, iterating linearly over the `typical_pods` array and doing a loose fit without actually simulating node packing. Additionally, the node scoring metric used a static linear bias, which is vastly less effective than Kubernetes' aggressive `MostAllocated` node utilization curve.

---

## 3. Improvements Implemented
We enhanced the global, algorithmic infrastructure within the `testing/` directory to mirror the robustness of enterprise-grade scheduling systems:

### 3.1 Over-provisioning via Safety Margins (`forecast_engine.py`)
- **Enhancement**: Introduced an industry-standard `safety_margin` coefficient (defaulting to 1.05) to the `ForecastEngine`.
- **Impact**: Forecasts are padded dynamically to absorb unexpected deviations and prediction noise, practically eliminating pod starvation stemming from minor model mispredictions.

### 3.2 Continuous Moving Averages for Workload Shifts (`distribution_mapper.py`)
- **Enhancement**: Implemented an **Exponential Moving Average (EMA)** with an alpha smoothing factor of `0.3` for tracking demand changes. 
- **Impact**: Instead of static probabilistic jumps, the mapping function now creates a continuous, smoothed shift of pod distributions. This eliminates thrashing and dynamically scales the magnitude of the shift proportionate to the actual severity of the demand surge.

### 3.3 Greedy Best-Fit Decreasing (BFD) Bin Packing (`fgd_scheduler.py`)
- **Enhancement 1**: Rewrote `calculate_fragmentation` to sort the incoming typical pod distributions descending (`reverse=True`) and perform a greedy bin-packing simulation. This accurately computes the true fragmented leftover space.
- **Enhancement 2**: Aligned the `packing_bias` strictly with Kubernetes' `MostAllocated` logic, exponentially prioritizing heavily loaded nodes over empty ones.
- **Impact**: By drastically improving the precision of the fragmentation gradient, the scheduler consolidates pods much more tightly, leaving vast blocks of contiguous GPU memory free for high-burst LLM inference and training pods.
