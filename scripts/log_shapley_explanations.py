# scripts/log_shapley_explanations.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from envs.stock_risk_env import StockRiskEnv
from csv_explainer import counterfactual_shapley

# Load model and environment
price_tensor = np.load("data/price_tensor.npy")
model = PPO.load("checkpoints/ppo_riskaware")  # or ppo_baseline

env = StockRiskEnv(price_tensor, lam=1.0)  # use same λ as model was trained with

obs, _ = env.reset()
shapley_logs = []

print("⏳ Logging Shapley explanations...")

for t in range(env.T - env.window_size - 1):
    obs_input = obs.copy()

    # Predict action (what the agent would do)
    def wrapped_policy(o):
        return model.policy.predict(np.array(o).reshape(1, -1), deterministic=True)[0][0]

    shap_values = counterfactual_shapley(obs_input, wrapped_policy, num_samples=30)

    # Save timestep, portfolio return, volatility, and shapley vector
    _, _, _, _, info = env.step(wrapped_policy(obs_input))
    shapley_logs.append({
        "t": t,
        "return": info["return"],
        "sigma": info["sigma"],
        "shap_values": shap_values
    })


# Convert to DataFrame and unpack shapley values
df = pd.DataFrame(shapley_logs)
shap_df = pd.DataFrame(df["shap_values"].to_list())
shap_df.columns = [f"phi_{i}" for i in range(shap_df.shape[1])]

result = pd.concat([df[["t", "return", "sigma"]], shap_df], axis=1)
result.to_parquet("data/shapley_logs_riskaware.parquet")
