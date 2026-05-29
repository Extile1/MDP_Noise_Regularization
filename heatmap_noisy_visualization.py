#!/usr/bin/env python3
import glob
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.wrappers import TimeLimit

from reacher_env import make_reacher_env, ResetNoiseWrapper, NLinkReacherEnv

def find_gamma_models(n_links):
    models = {}
    for path in glob.glob(f"./experiments_{n_links}links*/**/best_model.zip", recursive=True):
        m = re.search(r"gamma_([0-9\.]+)", path)
        if m: models[float(m.group(1))] = path
    return dict(sorted(models.items()))

def evaluate(model_path, n_links, noise_prob):
    try:
        model = PPO.load(model_path, device="cpu")
        # Manually construct to ensure ResetNoise is applied
        env = NLinkReacherEnv(n_links=n_links)
        if noise_prob > 0: env = ResetNoiseWrapper(env, prob=noise_prob)
        env = TimeLimit(env, max_episode_steps=2000)
        env.reset(seed=123)
        mean, _ = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
        env.close()
        return mean
    except: return np.nan

def main():
    N_LINKS = 2
    noise_lvls = [0.01, 0.05, 0.1, 0.2, 0.3]
    models = find_gamma_models(N_LINKS)
    
    if not models: return print("No models found")
    
    data = []
    gammas = list(models.keys())
    
    for g in tqdm(gammas):
        row = [evaluate(models[g], N_LINKS, p) for p in noise_lvls]
        data.append(row)
        
    df = pd.DataFrame(data, index=gammas, columns=noise_lvls)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt=".0f", cmap="viridis")
    plt.title("Gamma vs Reset Noise Robustness")
    plt.ylabel("Gamma")
    plt.xlabel("Reset Prob")
    plt.gca().invert_yaxis()
    plt.savefig("heatmap_robustness.png")

if __name__ == "__main__":
    main()