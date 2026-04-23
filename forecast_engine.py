import numpy as np

class ForecastEngine:
    def __init__(self, trace_data, safety_margin=1.05):
        """
        Simulates the Time-MoE forecast engine. 
        In a production setup, this would load the Time-MoE model and run inference.
        For simulation, we use the actual trace data but add some noise to simulate 
        the MAE of the model (MAE = ~0.012 for Time-MoE-50M).
        Incorporates an industry-standard safety_margin to over-provision and prevent SLA violations.
        """
        self.trace_data = trace_data
        self.safety_margin = safety_margin
        
    def get_forecast(self, current_time_step, lookahead=60):
        """
        Returns the forecast for the next `lookahead` minutes.
        """
        if current_time_step + lookahead >= len(self.trace_data):
            # Out of bounds
            return []
            
        actual_future = self.trace_data[current_time_step : current_time_step + lookahead]
        
        # Simulate Time-MoE forecast by adding small noise (simulating MAE 0.012)
        noise = np.random.normal(0, 0.012, size=lookahead)
        forecast = actual_future + noise
        
        # Apply safety margin to prevent under-provisioning during unexpected spikes
        forecast = forecast * self.safety_margin
        
        # Demand cannot be negative
        forecast = np.clip(forecast, 0, 1.0)
        
        return forecast.tolist()
