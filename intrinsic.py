import math
from collections import defaultdict

class VisitCounter:
    """
    Tracks state visit counts during training and computes
    count-based intrinsic exploration bonuses (Bellemare et al. 2016).
    
    Bonus formula: β / sqrt(N(s))
    
    where N(s) is how many times state s has been visited.
    High bonus for rarely visited states (encourages exploration).
    Low bonus for frequently visited states (stops rewarding known territory).
    
    Beta decays toward zero over training so the intrinsic signal
    fades as the Q-table matures and exploration becomes less important.
    """
    
    def __init__(self, beta=0.5):
        """
        Parameters:
            beta : initial exploration bonus scale. default 0.5.
                   decays toward 0 over training.
        """
        self.beta = beta
        self.count = defaultdict(int) 

    def get_bonus(self, state):
        """
        Returns the exploration bonus for a given state.
        
        Parameters:
            state : hashable state tuple from obs_to_state
        
        Returns:
            float — beta if never visited, beta/sqrt(N) otherwise
        """
        n = self.count[state]
        if n == 0:
            return self.beta
        return self.beta/(math.sqrt(n))
    
    def update(self, state):
        """
        Increments the visit count for a state by 1.
        Call this after every training step.
        
        Parameters:
            state : hashable state tuple from obs_to_state
        """
        self.count[state] += 1 
    
    def decay_beta(self, factor=0.9999):
        """
        Reduces beta by multiplying by factor.
        Call this after every training step.
        Beta is clamped to minimum of 0.01 so bonus never
        fully disappears.
        
        Parameters:
            factor : decay multiplier, default 0.9999
        """
        self.beta = max(0.01,self.beta*factor)