import numpy as np
import copy
from fgd_scheduler import Node, FGDScheduler
from distribution_mapper import DistributionMapper
from forecast_engine import ForecastEngine

def generate_synthetic_trace(duration_minutes=1440):
    """
    Generates a synthetic aggregate demand trace (1 day = 1440 mins)
    with a morning burst and stable afternoon.
    """
    t = np.linspace(0, 4*np.pi, duration_minutes)
    base_demand = 0.5 + 0.2 * np.sin(t)
    
    # Add a burst around minute 300
    base_demand[300:360] += 0.3 
    
    return np.clip(base_demand, 0, 1).tolist()

def generate_incoming_pods(duration_minutes=1440):
    """
    Generates the actual sequence of pod requests coming into the cluster.
    """
    pods = []
    for m in range(duration_minutes):
        # Base workload: many small and medium pods
        reqs = []
        if np.random.rand() < 0.95:
            reqs.append(np.random.choice([100, 250, 250, 500]))
        if np.random.rand() < 0.8:
            reqs.append(np.random.choice([100, 250]))
            
        # Burst workload around min 300: massive LLM pods
        if 300 <= m <= 400:
            if np.random.rand() < 0.95:
                reqs.extend([np.random.choice([500, 1000]), np.random.choice([500, 1000])])
                
        pods.append(reqs)
    return pods

def run_simulation(name, trace, incoming_pods, is_predictive=False):
    print(f"--- Running {name} ---")
    num_nodes = 4
    nodes = [Node(i) for i in range(num_nodes)]
    scheduler = FGDScheduler(nodes)
    mapper = DistributionMapper()
    
    engine = ForecastEngine(trace)
    
    # Static typical pods (derived from historical averages)
    static_typical = [100, 100, 250, 250, 500] 
    
    total_rejected = 0
    total_allocated = 0
    
    for t in range(len(trace)):
        # 1. Update Typical Pods if Predictive
        if is_predictive and t % 10 == 0:
            # Get forecast for next 60 mins
            forecast = engine.get_forecast(t)
            if forecast:
                typical_pods = mapper.forecast_to_distribution(trace[t], forecast)
            else:
                typical_pods = static_typical
        elif not is_predictive:
            typical_pods = static_typical
            
        # 2. Free up some GPU (simulate pod completion)
        if t % 10 == 0:
            for n in nodes:
                if n.allocated_gpu > 0:
                    # Random completion logic (needs to be deterministic per run to compare fairly)
                    np.random.seed(t) # Seed based on time step for identical completions
                    freed = min(n.allocated_gpu, np.random.choice([0, 100, 250]))
                    n.allocated_gpu -= freed
                    
        # 3. Schedule incoming pods
        for pod_req in incoming_pods[t]:
            # Use FGD to schedule
            best_node = scheduler.schedule(pod_req, typical_pods)
            if best_node:
                total_allocated += 1
            else:
                total_rejected += 1
                
    # Calculate final fragmentation
    total_frag = 0
    for n in nodes:
        # Unusable space (< 250 milli is a fragment for our large workload)
        if n.free_gpu < 250 and n.free_gpu > 0:
            total_frag += n.free_gpu
            
    print(f"Total Pods Allocated: {total_allocated}")
    print(f"Total Pods Rejected (Out of Resources): {total_rejected}")
    print(f"Final Wasted Fragments (gpu_milli): {total_frag}\n")
    return total_frag, total_rejected

if __name__ == "__main__":
    np.random.seed(42) # For reproducibility
    shared_trace = generate_synthetic_trace()
    shared_pods = generate_incoming_pods()
    
    # Run Static FGD (Baseline)
    frag_static, rej_static = run_simulation("Static FGD (Baseline)", shared_trace, shared_pods, is_predictive=False)
    
    # Run Predictive FGD (Proposed)
    frag_predictive, rej_pred = run_simulation("Predictive FGD (Time-MoE Forecasts)", shared_trace, shared_pods, is_predictive=True)
    
    if frag_static > 0:
        reduction = ((frag_static - frag_predictive) / frag_static) * 100
        print(f"==> Fragmentation Reduction: {reduction:.2f}%")
        
    if rej_static > 0:
        rej_reduction = ((rej_static - rej_pred) / rej_static) * 100
        print(f"==> Rejection Reduction: {rej_reduction:.2f}%")
