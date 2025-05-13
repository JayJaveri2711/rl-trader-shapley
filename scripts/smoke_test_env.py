import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from envs.stock_risk_env import StockRiskEnv

tensor = np.load("data/price_tensor.npy")
env = StockRiskEnv(tensor)

obs, _ = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    print(f"Reward: {reward:.4f}, Info: {info}")
