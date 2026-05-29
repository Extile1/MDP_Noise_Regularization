#!/usr/bin/env python3
"""
JSD Analysis: Compares Reward Distributions between Gamma Models (Set A) and Noise Models (Set B).
This script builds a similarity heatmap to see if high-gamma models behave similarly to noise-robust models.
"""
import os
import glob
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from stable_baselines3 import PPO
from tqdm import tqdm

# Import shared env from reacher_env.py
from reacher_env import make_reacher_env

def find_gamma_models(n_links):
    """Set A: Models trained with different Gamma values (Clean Env)."""
    models = {}
    # Matches structure from train_basic.py: experiments_{n}links_5_2000_new/gamma_{val}/...
    pattern = f"./experiments_{n_links}links_5_2000_new/gamma_*/best_model.zip"
    
    files = glob.glob(pattern, recursive=True)
    for path in files:
        match = re.search(r"gamma_([0-9\.]+)", path)
        if match:
            gamma = float(match.group(1))
            models[gamma] = path
    return dict(sorted(models.items()))

def find_noise_models(n_links):
    """Set B: Models trained with different Noise levels (e.g. Reset or Random)."""
    models = {}
    # Matches structure from train_noisy.py: experiments_{n}links_5_2000_noise_reset/g{gamma}/noise_{type}/{prob}/...
    pattern = f"./experiments_{n_links}links_5_2000_noise_reset/**/best_model.zip"
    
    files = glob.glob(pattern, recursive=True)
    for path in files:
        parts = path.split(os.sep)
        # We look for the part that defines the noise probability
        # Structure is usually: .../noise_{type}/{prob}/...
        try:
            # Iterate parts to find noise definition
            for i, part in enumerate(parts):
                if part.startswith("noise_") and i + 1 < len(parts):
                    n_type = part.replace("noise_", "")
                    try:
                        prob = float(parts[i+1])
                        # Create a key like "reset_0.05"
                        key = f"{n_type}_{prob}"
                        models[key] = path
                    except ValueError:
                        continue # Next part might match
        except Exception:
            continue
            
    # Sort keys (custom sort to handle strings with numbers)
    return dict(sorted(models.items(), key=lambda x: (x[0].split('_')[0], float(x[0].split('_')[1]))))

def collect_rewards(model_path, env, n_seeds=50, steps=2000):
    """
    Runs policy on multiple seeds on a CLEAN environment to get its behavior distribution.
    We use a Clean environment for both sets to compare their fundamental policy behavior.
    """
    try:
        model = PPO.load(model_path, device="cpu")
    except Exception as e:
        print(f"Failed to load {model_path}: {e}")
        return np.array([])

    rewards = []
    start_seed = 5000 # Use validation seeds
    
    for i in range(n_seeds):
        obs, _ = env.reset(seed=start_seed + i)
        episode_rewards = []
        for _ in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            episode_rewards.append(r)
            if term or trunc:
                break
        rewards.extend(episode_rewards)
        
    return np.array(rewards)

def main():
    N_LINKS = 2
    
    print(f"--- JSD Policy Similarity Analysis (Links: {N_LINKS}) ---")

    # 1. Find Models
    gamma_models = find_gamma_models(N_LINKS) # Set A
    noise_models = find_noise_models(N_LINKS) # Set B
    
    if not gamma_models:
        print("Error: No Gamma models found (Set A). Run train_basic.py first.")
        return
    if not noise_models:
        print("Error: No Noise models found (Set B). Run train_noisy.py first.")
        return
        
    print(f"Set A (Gamma): Found {len(gamma_models)} models -> {list(gamma_models.keys())}")
    print(f"Set B (Noise): Found {len(noise_models)} models -> {list(noise_models.keys())}")

    # 2. Collect Data
    # We compare policies on the standard (clean) environment to see if they learned similar behaviors
    env = make_reacher_env(N_LINKS, 2000, noise_type="none")
    
    cache_gamma = {}
    cache_noise = {}
    
    print("\n[1/2] Collecting reward distributions from Gamma Models...")
    for g, path in tqdm(gamma_models.items()):
        cache_gamma[g] = collect_rewards(path, env)
        
    print("\n[2/2] Collecting reward distributions from Noise Models...")
    for n, path in tqdm(noise_models.items()):
        cache_noise[n] = collect_rewards(path, env)

    env.close()

    # 3. Global Binning (Crucial for JSD)
    # Concatenate all data to determine global range
    valid_arrays = [x for x in list(cache_gamma.values()) + list(cache_noise.values()) if x.size > 0]
    if not valid_arrays:
        print("No valid reward data collected.")
        return

    all_rewards = np.concatenate(valid_arrays)
    global_min, global_max = all_rewards.min(), all_rewards.max()
    bins = np.linspace(global_min, global_max, 100)
    
    # 4. Compute JSD Matrix
    gamma_keys = list(gamma_models.keys()) 
    noise_keys = list(noise_models.keys())
    
    matrix = np.zeros((len(gamma_keys), len(noise_keys)))
    
    print(f"\nComputing JSD Matrix ({len(gamma_keys)}x{len(noise_keys)})...")
    for i, g in enumerate(gamma_keys):
        for j, n in enumerate(noise_keys):
            p = cache_gamma[g]
            q = cache_noise[n]
            
            if p.size == 0 or q.size == 0:
                matrix[i, j] = np.nan
                continue
            
            hp, _ = np.histogram(p, bins=bins, density=True)
            hq, _ = np.histogram(q, bins=bins, density=True)
            
            # Add epsilon to avoid zero division/log issues in JSD calculation
            hp += 1e-12
            hq += 1e-12
            
            # Jensen-Shannon Distance (0 = Identical, 1 = Disjoint)
            matrix[i, j] = jensenshannon(hp, hq)

    # 5. Plotting
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        matrix, 
        annot=True, 
        fmt=".3f", 
        xticklabels=noise_keys, 
        yticklabels=gamma_keys, 
        cmap="viridis_r", # Reversed: Dark/Blue (Low JSD) = High Similarity
        cbar_kws={'label': 'Jensen-Shannon Distance (Lower is More Similar)'}
    )
    plt.title(f"Policy Similarity Heatmap ({N_LINKS}-Link)\nGamma Models vs. Noise-Robust Models")
    plt.ylabel("Gamma (Discount Factor)")
    plt.xlabel("Noise Configuration")
    plt.tight_layout()
    
    out_file = "jsd_analysis.png"
    plt.savefig(out_file, dpi=150)
    print(f"\nDone! Heatmap saved to: {out_file}")

if __name__ == "__main__":
    main()