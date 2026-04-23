import numpy as np

class DistributionMapper:
    def __init__(self, historical_sizes=[100, 250, 500, 1000], historical_probs=[0.4, 0.3, 0.2, 0.1]):
        """
        Maps aggregate demand forecasts into a probability distribution of expected pod sizes.
        historical_sizes: Common GPU milli requests.
        historical_probs: The default probability of seeing those sizes during stable periods.
        """
        self.sizes = np.array(historical_sizes)
        self.base_probs = np.array(historical_probs)
        
    def forecast_to_distribution(self, current_demand, forecast_demand_window):
        """
        Converts a forecast window of demand into an expected list of 'Typical Pods'
        that will arrive in the near future.
        """
        # Calculate the trend (derivative)
        demand_change = np.mean(np.diff(forecast_demand_window))
        
        # Implement Exponential Moving Average (EMA) for smoother, industry-standard trend tracking
        if not hasattr(self, 'ema_trend'):
            self.ema_trend = 0.0
            self.alpha = 0.3 # Smoothing factor
            
        self.ema_trend = (self.alpha * demand_change) + ((1 - self.alpha) * self.ema_trend)
        
        # If demand is rapidly increasing, we expect larger pods (LLM training bursts)
        # If demand is stable or decreasing, we expect standard/small pods
        probs = self.base_probs.copy()
        
        # Continuous probabilistic shift based on EMA magnitude instead of brittle threshold
        shift_magnitude = np.clip(self.ema_trend * 4.0, -0.4, 0.4)
        
        if shift_magnitude > 0:
            probs[2:] += shift_magnitude / 2
            probs[:2] -= shift_magnitude / 2
        else:
            probs[:2] -= shift_magnitude / 2 # shift_magnitude is negative, so subtract negative = add
            probs[2:] += shift_magnitude / 2
            
        # Ensure valid probability distribution
        probs = np.clip(probs, 0, 1)
        probs = probs / np.sum(probs)
        
        # Generate the dynamic typical pods list (e.g., 10 representative pods)
        # We sample sizes based on the dynamic probabilities
        typical_pods = np.random.choice(self.sizes, size=10, p=probs).tolist()
        return typical_pods

if __name__ == "__main__":
    mapper = DistributionMapper()
    # Test stable demand
    print("Stable:", mapper.forecast_to_distribution(0.5, [0.5, 0.51, 0.5, 0.49, 0.5]))
    # Test surging demand
    print("Surge:", mapper.forecast_to_distribution(0.5, [0.5, 0.6, 0.7, 0.8, 0.9]))
