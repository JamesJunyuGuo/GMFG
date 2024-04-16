import numpy as np 
from abc import ABC, abstractmethod
from scipy.stats import entropy
class controller(ABC):
    def __init__(self,action_space):
        self.action_space = action_space
        
    @abstractmethod
    def random_policy(self):
        pass 
        
    @abstractmethod
    def reward(self, action, mean_field):
        pass 
    


class ring_control(controller):
    def __init__(self, action_space):
        super().__init__(action_space)
        self.naction = len(action_space)
        self.current_action = 0
    
    def random_policy(self):
        return np.ones((self.naction))/self.naction
    
    def reward(self,action,mean_field):
        return entropy(mean_field) 
    
    

class SIR_controller(controller):
    def __init__(self, action_space):
        super().__init__(action_space)
        self.naction = len(action_space)
        self.current_action = 0
    
    def random_policy(self):
        return np.ones((self.naction))/self.naction
    
    def reward(self,action,mean_field):
        return 2*(1-mean_field[1]) + (self.naction-action)
    
    
        
    
    
        
        
        
    