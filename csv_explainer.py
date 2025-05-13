import numpy as np

def counterfactual_shapley(obs, model, baseline=None, num_samples=100):
    """
    Computes per-feature Shapley values for a single observation.
    """

    obs = obs.copy()
    d = obs.shape[0]
    baseline = baseline if baseline is not None else np.mean(obs)
    full_pred = model(obs)
    shap_values = np.zeros(d)

    for _ in range(num_samples):
        perm = np.random.permutation(d)
        masked = np.full_like(obs, baseline)

        for i, idx in enumerate(perm):
            masked[idx] = obs[idx]
            with_feat = model(masked)
            without_feat = model(np.where(np.arange(d) == idx, baseline, masked))
            shap_values[idx] += (with_feat - without_feat)

    shap_values /= num_samples
    return shap_values
