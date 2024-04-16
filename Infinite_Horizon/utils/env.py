import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym import spaces
import numpy as np

class InfiniteHorizonEnv(gym.Env):
    def __init__(self, gamma=0.99):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.state = 0
        self.gamma = gamma  # Discount rate
        self.current_step = 0

    def step(self, action):
        self.state += -1 if action == 0 else 1
        immediate_reward = abs(self.state)  # Immediate reward without discounting
        
        # Apply discounting to the immediate reward
        discounted_reward = immediate_reward * (self.gamma ** self.current_step)
        self.current_step += 1

        info = {'immediate_reward': immediate_reward}  # Keep the immediate reward in the info for reference
        return np.array([self.state]).astype(np.float32), discounted_reward, False, info

    def reset(self):
        self.state = 0
        self.current_step = 0
        return np.array([self.state]).astype(np.float32)
    
    def render(self, mode='human'):
        print(f"State: {self.state}")


