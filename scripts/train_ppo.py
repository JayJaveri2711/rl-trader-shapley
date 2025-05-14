# envs/stock_risk_env.py
## run via command : python scripts/train_ppo.py --lam 1 --tag msft_only

import gymnasium as gym
import numpy as np

class StockRiskEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, price_tensor, lam=1.0, window_size=30):
        super().__init__()
        self.price_tensor = price_tensor  # shape (T, N, F)
        self.T, self.N, self.F = price_tensor.shape
        self.lam = lam
        self.window_size = window_size
        self.reset()

        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.N,))
        obs_dim = self.N * self.F * self.window_size + 1  # +1 for cashs
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,))

    def reset(self, seed=None, options=None):
        self.t = self.window_size
        self.done = False
        self.cash = 1.0
        self.weights = np.zeros(self.N)
        self.returns_history = []
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, 0, 1)
        action = action / (np.sum(action) + 1e-8)  # normalize to portfolio weights

        # calculate portfolio return
        prev_prices = self.price_tensor[self.t - 1, :, 3]  # close price yesterday
        curr_prices = self.price_tensor[self.t, :, 3]      # close price today
        price_rel = curr_prices / prev_prices
        portfolio_return = np.dot(action, price_rel - 1)

        self.returns_history.append(portfolio_return)
        self.cash *= (1 + portfolio_return)

        # rolling volatility (optional: change window)
        if len(self.returns_history) >= 30:
            sigma = np.std(self.returns_history[-30:])
        else:
            sigma = np.std(self.returns_history)

        reward = portfolio_return - self.lam * sigma

        self.weights = action
        self.t += 1
        terminated = self.t >= self.T - 1
        obs = self._get_obs()
        info = {"return": portfolio_return, "sigma": sigma}
        return obs, reward, terminated, False, info

    def _get_obs(self):
        price_window = self.price_tensor[self.t - self.window_size : self.t]  # (W, N, F)
        obs = price_window.transpose(1, 2, 0).reshape(-1)  # → (N*F*W,)
        return np.concatenate(([self.cash], obs))
