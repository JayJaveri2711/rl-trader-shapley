# scripts/train_riskppo.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.stock_risk_env import StockRiskEnv

# Load data
price_tensor = np.load("data/price_tensor.npy")

# Create environment with risk penalty (λ = 1.0 is a good starting point)
def make_env():
    return StockRiskEnv(price_tensor, lam=1.0)

env = DummyVecEnv([make_env])

# Train PPO
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=2048,
    batch_size=512,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
)

model.learn(total_timesteps=50_000)

# Save
os.makedirs("checkpoints", exist_ok=True)
model.save("checkpoints/ppo_riskaware")
print("✅ Risk-aware model saved to checkpoints/ppo_riskaware.zip")
