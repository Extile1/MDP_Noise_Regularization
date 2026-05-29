#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Accelerated PPO training for custom N-link Reacher (Noise)."""

import os
import shutil
import argparse
from dataclasses import dataclass
import numpy as np
import optuna
import pandas as pd
import imageio
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# Import shared env
from reacher_env import make_reacher_env

@dataclass
class Config:
    output_dir: str = "./experiments_custom_reacher"
    n_links: int = 4
    episode_len: int = 2000
    noise_type: str = "none"
    noise_prob: float = 0.0
    n_envs: int = 12
    device: str = "cpu"
    tuning_trials: int = 10
    tuning_steps: int = 1_000_000
    optuna_jobs: int = 1
    eval_freq: int = 50_000
    n_eval_episodes: int = 10
    total_timesteps: int = 200_000_000
    gamma: float = 0.995
    n_steps: int = 1024
    n_epochs: int = 10
    video_frames: int = 2000
    video_fps: int = 30
    video_seed: int = 999

CFG = Config()

def make_subproc_vec_env(n_envs, n_links, episode_len, noise_type, noise_prob, log_dir=None, base_seed=0):
    def _make(rank):
        def _init():
            env = make_reacher_env(n_links, episode_len, noise_type, noise_prob, seed=base_seed + rank)
            if log_dir is not None:
                os.makedirs(log_dir, exist_ok=True)
                env = Monitor(env, filename=os.path.join(log_dir, f"monitor_{rank}.csv"))
            return env
        return _init
    return SubprocVecEnv([_make(i) for i in range(n_envs)])

def make_eval_vec_env(n_links, episode_len, noise_type, noise_prob, seed=0):
    def _init():
        return make_reacher_env(n_links, episode_len, noise_type, noise_prob, seed=seed)
    return DummyVecEnv([_init])

class OptunaEvalCallback(EvalCallback):
    def __init__(self, eval_env, trial, n_eval_episodes=10, eval_freq=50_000, deterministic=True, verbose=0):
        super().__init__(eval_env, n_eval_episodes=n_eval_episodes, eval_freq=eval_freq, deterministic=deterministic, verbose=verbose)
        self.trial = trial
        self.eval_idx = 0

    def _on_step(self) -> bool:
        cont = super()._on_step()
        if self.n_calls % self.eval_freq == 0 and self.last_mean_reward is not None:
            self.eval_idx += 1
            self.trial.report(float(self.last_mean_reward), step=self.eval_idx)
            if self.trial.should_prune():
                raise optuna.TrialPruned()
        return cont

def load_monitor_logs_time_sorted(log_dir: str) -> pd.DataFrame:
    # (Same as train_basic.py)
    paths = []
    for root, _, files in os.walk(log_dir):
        for f in files:
            if f.lower().endswith(".csv") and "monitor" in f.lower():
                paths.append(os.path.join(root, f))
    if not paths: raise FileNotFoundError(f"No monitor CSV files found in {log_dir}")
    dfs = []
    for p in sorted(paths):
        try:
            df = pd.read_csv(p, skiprows=1)
            if {"r", "l", "t"}.issubset(df.columns):
                dfs.append(df[["r", "l", "t"]].copy())
        except Exception: pass
    if not dfs: return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df["timesteps"] = df["l"].cumsum()
    return df

def plot_reward_curves(cfg: Config, log_dir: str):
    df = load_monitor_logs_time_sorted(log_dir)
    if df.empty: return
    window = max(50, len(df) // 20)
    df["r_smooth"] = df["r"].rolling(window=window, min_periods=1).mean()
    plt.figure(figsize=(10, 6))
    plt.plot(df["timesteps"], df["r"], alpha=0.25)
    plt.plot(df["timesteps"], df["r_smooth"], linewidth=2, label=f"Avg ({window})")
    plt.title(f"Reward (p={cfg.noise_prob})")
    plt.savefig(os.path.join(cfg.output_dir, "reward_vs_timesteps.png"), dpi=150)
    plt.close()

def run_tuning(cfg: Config):
    print(f"\n[1/4] Optuna tuning: {cfg.tuning_trials} trials")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(), sampler=optuna.samplers.TPESampler(seed=0))
    def objective(trial: optuna.Trial):
        lr = trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True)
        ent_coef = trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])
        
        train_env = make_subproc_vec_env(cfg.n_envs, cfg.n_links, cfg.episode_len, cfg.noise_type, cfg.noise_prob, log_dir=None, base_seed=10000+trial.number*100)
        eval_env = make_eval_vec_env(cfg.n_links, cfg.episode_len, cfg.noise_type, cfg.noise_prob, seed=20000+trial.number)
        
        model = PPO("MlpPolicy", train_env, learning_rate=lr, ent_coef=ent_coef, gamma=cfg.gamma, 
                    n_steps=cfg.n_steps, batch_size=batch_size, n_epochs=cfg.n_epochs, verbose=0, device=cfg.device)
        cb = OptunaEvalCallback(eval_env, trial, cfg.n_eval_episodes, cfg.eval_freq)
        try:
            model.learn(total_timesteps=cfg.tuning_steps, callback=cb)
            mean_r, _ = evaluate_policy(model, eval_env, n_eval_episodes=cfg.n_eval_episodes, deterministic=True)
        finally:
            train_env.close()
            eval_env.close()
        return float(mean_r)
    study.optimize(objective, n_trials=cfg.tuning_trials, n_jobs=cfg.optuna_jobs)
    return study.best_params

def train_final_model(cfg: Config, best_params: dict):
    print(f"\n[2/4] Training final: total_timesteps={cfg.total_timesteps}")
    os.makedirs(cfg.output_dir, exist_ok=True)
    log_dir = os.path.join(cfg.output_dir, "logs")
    if os.path.exists(log_dir): shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    train_env = make_subproc_vec_env(cfg.n_envs, cfg.n_links, cfg.episode_len, cfg.noise_type, cfg.noise_prob, log_dir=log_dir, base_seed=12345)
    eval_env = make_eval_vec_env(cfg.n_links, cfg.episode_len, cfg.noise_type, cfg.noise_prob, seed=54321)
    
    best_path = os.path.join(cfg.output_dir, "best_checkpoints")
    cb = EvalCallback(eval_env, best_model_save_path=best_path, log_path=log_dir, eval_freq=cfg.eval_freq, deterministic=True, verbose=1)
    
    model = PPO("MlpPolicy", train_env, gamma=cfg.gamma, n_steps=cfg.n_steps, n_epochs=cfg.n_epochs, verbose=1, device=cfg.device, **best_params)
    model.learn(total_timesteps=cfg.total_timesteps, callback=cb)
    
    final_path = os.path.join(cfg.output_dir, f"ppo_reacher_{cfg.n_links}links_final")
    model.save(final_path)
    
    best_file = os.path.join(best_path, "best_model.zip")
    return_path = os.path.join(best_path, "best_model") if os.path.exists(best_file) else final_path
    
    train_env.close()
    eval_env.close()
    return return_path, log_dir

def validate_noise_impact(cfg: Config, model_path: str):
    print(f"\n[5/5] Validation (Noisy vs Clean)")
    model = PPO.load(model_path, device=cfg.device)
    
    # 1. Noisy
    env1 = make_reacher_env(cfg.n_links, cfg.episode_len, cfg.noise_type, cfg.noise_prob, seed=123)
    mean1, std1 = evaluate_policy(model, env1, n_eval_episodes=20, deterministic=True)
    env1.close()
    
    # 2. Clean
    env2 = make_reacher_env(cfg.n_links, cfg.episode_len, "none", 0.0, seed=123)
    mean2, std2 = evaluate_policy(model, env2, n_eval_episodes=20, deterministic=True)
    env2.close()
    
    print(f"  Noisy ({cfg.noise_type}, {cfg.noise_prob}): {mean1:.2f} +/- {std1:.2f}")
    print(f"  Clean: {mean2:.2f} +/- {std2:.2f}")
    with open(os.path.join(cfg.output_dir, "noise_val.txt"), "w") as f:
        f.write(f"Noisy: {mean1}\nClean: {mean2}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--n_links", type=int, default=2)
    parser.add_argument("--gemma", type=float, default=0.99, dest="gamma")
    parser.add_argument("--video_seed", type=int, default=999)
    parser.add_argument("--noise_type", type=str, default="none", choices=["none", "sticky", "reset", "random"])
    parser.add_argument("--noise_prob", type=float, default=0.1)
    args = parser.parse_args()

    CFG.n_links = args.n_links
    CFG.gamma = args.gamma
    CFG.video_seed = args.video_seed
    CFG.noise_type = args.noise_type
    CFG.noise_prob = args.noise_prob
    CFG.output_dir = f"./experiments_{CFG.n_links}links_5_2000_noise_reset/g{CFG.gamma}/noise_{CFG.noise_type}/{CFG.noise_prob}"
    
    if os.environ.get("MUJOCO_GL") != "egl": os.environ["MUJOCO_GL"] = "egl"
    os.makedirs(CFG.output_dir, exist_ok=True)

    best_params = run_tuning(CFG)
    model_path, log_dir = train_final_model(CFG, best_params)
    plot_reward_curves(CFG, log_dir)
    
    # Video
    load_path = model_path if model_path.endswith(".zip") else model_path + ".zip"
    vid_env = make_reacher_env(CFG.n_links, CFG.episode_len, CFG.noise_type, CFG.noise_prob, render_mode="rgb_array", seed=CFG.video_seed)
    model = PPO.load(load_path, device=CFG.device)
    obs, _ = vid_env.reset()
    frames = []
    for _ in range(CFG.video_frames):
        frames.append(vid_env.render())
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = vid_env.step(action)
        if term or trunc: obs, _ = vid_env.reset()
    vid_env.close()
    imageio.mimsave(os.path.join(CFG.output_dir, f"noise_{CFG.noise_type}.mp4"), frames, fps=CFG.video_fps)

    validate_noise_impact(CFG, load_path)

if __name__ == "__main__":
    main()