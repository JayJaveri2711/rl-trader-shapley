# scripts/train_baseline.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.stock_risk_env import StockRiskEnv

# Load the price tensor
price_tensor = np.load("data/price_tensor.npy")

# Create environment (wrapped in VecEnv for SB3)
def make_env():
    return StockRiskEnv(price_tensor, lam=0.0)

env = DummyVecEnv([make_env])

# Initialize PPO agent
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    n_steps=2048,
    batch_size=512,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
)

# Train
model.learn(total_timesteps=50_000)

# Save model
os.makedirs("checkpoints", exist_ok=True)
model.save("checkpoints/ppo_baseline")
print("✅ Model saved to checkpoints/ppo_baseline.zip")
