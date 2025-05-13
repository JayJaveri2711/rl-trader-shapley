# Risk-Calibrated RL-Trader with Shapley Explanations

## 1. Overview

This project builds a reinforcement‑learning‑based equities trader that optimizes risk‑adjusted returns and produces per‑decision Shapley attributions for model interpretability. We train PPO agents with a volatility penalty and analyze both performance and explanation stability across market regimes.

## 2. Motivation & Justification

Quantitative trading desks demand both profitability and transparency. Vanilla RL traders can maximize PnL, but risk behavior and decision logic remain opaque. By integrating Shapley‑based explainability, we aim to:

* **Improve trust** in automated decision systems.
* **Audit risk behavior** under different market conditions.
* **Demonstrate reproducibility** for industrial interview showcases.

## 3. Objectives & Expectations

1. **Baseline PPO**: Establish a vanilla PPO trader’s performance (Sharpe, Sortino).
2. **Risk‑Calibrated PPO**: Add a penalty term for realized volatility; tune λ to balance return vs. drawdown.
3. **Shapley Explainer**: Attach per‑step Counterfactual Shapley Values (CSV) and record attributions.
4. **Stability Study**: Measure explanation consistency (Spearman ρ, cosine similarity) across bull, bear, and choppy regimes.
5. **Deliverables**: Code, backtest notebook, explanation logs, white paper.

By Day 10, expect a working pipeline with end‑to‑end training, CSV logging, and preliminary stability plots.

## 4. Background Knowledge & Statistics

* **Reinforcement Learning**: PPO, policy gradients, GAE.
* **Financial Metrics**: Sharpe Ratio, Sortino Ratio, max‑drawdown, expected shortfall.
* **Explainable ML**: Shapley values, model‑agnostic attributions, Counterfactual Shapley Values for RL.
* **Time Series Data**: OHLCV bars, volatility estimation, minute‑bar alignment.

Required familiarity: PyTorch, Gymnasium, stable‑baselines3, pandas, NumPy.

## 5. Technical Methodology

1. **Data Pipeline**: Fetch minute‑bar OHLCV, clean, feature‑engineer rolling returns and volatilities.
2. **Custom Gym Environment**: Reward = portfolio\_return − λ·realized\_volatility; action = position weights.
3. **Model Training**: Train vanilla and risk‑calibrated PPO agents; log performance.
4. **Shapley Attribution**: Implement CSV per step; batch compute attributions; store to Parquet.
5. **Evaluation & Analysis**: Backtest, regime segmentation, compute performance and stability metrics; visualize results.

## 6. Project Structure

```
rl-trader-shapley/
├── data/                  # raw and processed minute-bar data
├── envs/                  # custom Gym environment code
├── scripts/               # data download and training scripts
├── notebooks/             # analysis and backtesting notebooks
├── csv_explainer.py       # CSV implementation
├── requirements.txt       # pip dependencies
└── README.md              # this document
```

## 7. Usage

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Download data**: `python scripts/download_data.py --symbols AAPL MSFT ...`
3. **Train baseline**: `python scripts/train_baseline.py`
4. **Train risk‑PPO**: `python scripts/train_riskppo.py --lambda 1.0`
5. **Run CSV hook**: integrated in training scripts or via `csv_explainer.py`.
6. **Analyze**: Open `notebooks/quick_backtest.ipynb`.

## 8. Expected Results

* PnL improvement over vanilla PPO at optimized λ.
* Stable Shapley rankings across market regimes (target Spearman ρ > 0.7).
* Clear visualizations: equity curves, violin plots of attribution stability.

## 9. References

* Counterfactual Shapley Values for RL (ArXiv, Aug 2024)
* Explainable Post‑Hoc Portfolio RL (PLOS ONE, 2025)
* Schulman et al., Proximal Policy Optimization, 2017
* Buehler & Haas, Volatility‑Targeting Rewards, ICAIF 2022
