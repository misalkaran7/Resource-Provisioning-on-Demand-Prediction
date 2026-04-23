import math

class Node:
    def __init__(self, id, total_gpu=1000):
        self.id = id
        self.total_gpu = total_gpu
        self.allocated_gpu = 0
        
    @property
    def free_gpu(self):
        return self.total_gpu - self.allocated_gpu
        
    def can_fit(self, pod_gpu):
        return self.free_gpu >= pod_gpu
        
    def allocate(self, pod_gpu):
        if self.can_fit(pod_gpu):
            self.allocated_gpu += pod_gpu
            return True
        return False

class FGDScheduler:
    def __init__(self, nodes):
        self.nodes = nodes
        
    def calculate_fragmentation(self, node, typical_pods):
        """
        Calculates the fragmentation score of a node using a greedy bin-packing simulation.
        Fragmentation is the total free GPU space that CANNOT be used 
        by the typical pods because it's too fragmented.
        """
        wasted_space = node.free_gpu
        
        # Sort pods descending to simulate Best-Fit Decreasing (BFD) logic
        sorted_pods = sorted(typical_pods, reverse=True)
        
        # Greedily pack pods to find true leftover (fragmented) space
        for pod in sorted_pods:
            if wasted_space >= pod:
                wasted_space -= pod
                
        return max(0, wasted_space)

    def score_node(self, node, pod_req, typical_pods):
        """
        Score a node based on the Fragmentation Gradient Descent (FGD) logic.
        Score = sigmoid((OldFrag - NewFrag) / 1000) * 100
        """
        if not node.can_fit(pod_req):
            return -1 # Invalid node
            
        old_frag = self.calculate_fragmentation(node, typical_pods)
        
        # Simulate placement
        node.allocated_gpu += pod_req
        new_frag = self.calculate_fragmentation(node, typical_pods)
        
        # Revert simulation
        node.allocated_gpu -= pod_req
        
        # Calculate score (Steepest descent of fragmentation)
        frag_diff = old_frag - new_frag
        
        # Sigmoid normalization
        try:
            score = (1 / (1 + math.exp(-frag_diff / 50.0))) * 100
        except OverflowError:
            score = 100 if frag_diff > 0 else 0
            
        # Kubernetes MostAllocated style packing bias
        # Strongly prioritizes nodes that are already heavily utilized to free up empty nodes
        node_utilization = node.allocated_gpu / node.total_gpu
        packing_bias = node_utilization * 25
        
        return score + packing_bias

    def schedule(self, pod_req, typical_pods):
        """
        Selects the best node for the incoming pod.
        """
        best_node = None
        best_score = -float('inf')
        
        for node in self.nodes:
            score = self.score_node(node, pod_req, typical_pods)
            if score > best_score:
                best_score = score
                best_node = node
                
        if best_node:
            best_node.allocate(pod_req)
            
        return best_node

if __name__ == "__main__":
    nodes = [Node(1), Node(2)]
    nodes[0].allocated_gpu = 800 # 200 free
    nodes[1].allocated_gpu = 200 # 800 free
    
    scheduler = FGDScheduler(nodes)
    
    # Static list expecting 250milli pods
    static_typical = [250, 250, 250]
    
    # We want to place a 200milli pod
    best = scheduler.schedule(200, static_typical)
    print(f"Scheduled on Node {best.id if best else 'None'}")
