#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Accelerated PPO training for custom N-link Reacher (Basic)."""

import os
import shutil
import argparse
from dataclasses import dataclass

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
    n_envs: int = 12
    device: str = "cpu"
    tuning_trials: int = 10
    tuning_steps: int = 1_000_000
    optuna_jobs: int = 1
    eval_freq: int = 100_000
    n_eval_episodes: int = 20
    n_eval_envs: int = 5
    total_timesteps: int = 500_000_000
    gamma: float = 0.995
    n_steps: int = 1024
    n_epochs: int = 10
    video_frames: int = 2000
    video_fps: int = 30
    video_seed: int = 999

CFG = Config()

def make_subproc_vec_env(n_envs, n_links, episode_len, log_dir=None, base_seed=0):
    def _make(rank):
        def _init():
            env = make_reacher_env(n_links, episode_len, render_mode=None, seed=base_seed + rank)
            if log_dir is not None:
                os.makedirs(log_dir, exist_ok=True)
                env = Monitor(env, filename=os.path.join(log_dir, f"monitor_{rank}.csv"))
            return env
        return _init
    return SubprocVecEnv([_make(i) for i in range(n_envs)])

def make_eval_vec_env(n_links, episode_len, n_envs=5, base_seed=123):
    def _make(rank):
        def _init():
            return make_reacher_env(n_links, episode_len, render_mode=None, seed=base_seed + rank)
        return _init
    return DummyVecEnv([_make(i) for i in range(n_envs)])

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
    paths = []
    for root, _, files in os.walk(log_dir):
        for f in files:
            if f.lower().endswith(".csv") and "monitor" in f.lower():
                paths.append(os.path.join(root, f))
    if not paths: raise FileNotFoundError(f"No monitor CSV files found under: {log_dir}")

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
    plt.plot(df["timesteps"], df["r"], alpha=0.25, label="Episode reward")
    plt.plot(df["timesteps"], df["r_smooth"], linewidth=2, label=f"Moving avg ({window})")
    plt.title(f"Reward vs timesteps ({cfg.n_links}-link)")
    plt.savefig(os.path.join(cfg.output_dir, "reward_vs_timesteps.png"), dpi=150)
    plt.close()

def run_tuning(cfg: Config):
    print(f"\n[1/4] Optuna tuning: {cfg.tuning_trials} trials")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(), sampler=optuna.samplers.TPESampler(seed=0))

    def objective(trial: optuna.Trial):
        lr = trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True)
        ent_coef = trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])
        
        train_env = make_subproc_vec_env(cfg.n_envs, cfg.n_links, cfg.episode_len, log_dir=None, base_seed=10000+trial.number*100)
        eval_env = make_eval_vec_env(cfg.n_links, cfg.episode_len, n_envs=3, base_seed=20000+trial.number)
        
        model = PPO("MlpPolicy", train_env, learning_rate=lr, ent_coef=ent_coef, gamma=cfg.gamma, 
                    n_steps=cfg.n_steps, batch_size=batch_size, n_epochs=cfg.n_epochs, verbose=0, device=cfg.device)
        
        cb = OptunaEvalCallback(eval_env, trial, cfg.n_eval_episodes, cfg.eval_freq)
        try:
            model.learn(total_timesteps=cfg.tuning_steps, callback=cb)
            mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=cfg.n_eval_episodes, deterministic=True)
        finally:
            train_env.close()
            eval_env.close()
        return float(mean_reward)

    study.optimize(objective, n_trials=cfg.tuning_trials, n_jobs=cfg.optuna_jobs)
    print("Best params:", study.best_params)
    return study.best_params

def train_final_model(cfg: Config, best_params: dict):
    print(f"\n[2/4] Training final PPO: total_timesteps={cfg.total_timesteps}")
    os.makedirs(cfg.output_dir, exist_ok=True)
    log_dir = os.path.join(cfg.output_dir, "logs")
    if os.path.exists(log_dir): shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    train_env = make_subproc_vec_env(cfg.n_envs, cfg.n_links, cfg.episode_len, log_dir=log_dir, base_seed=12345)
    eval_env = make_eval_vec_env(cfg.n_links, cfg.episode_len, n_envs=cfg.n_eval_envs, base_seed=60000)

    eval_callback = EvalCallback(eval_env, best_model_save_path=cfg.output_dir, log_path=cfg.output_dir,
                                 eval_freq=max(cfg.eval_freq // cfg.n_envs, 1), n_eval_episodes=cfg.n_eval_episodes,
                                 deterministic=True, verbose=1)

    model = PPO("MlpPolicy", train_env, gamma=cfg.gamma, n_steps=cfg.n_steps, n_epochs=cfg.n_epochs,
                verbose=1, device=cfg.device, **best_params)
    model.learn(total_timesteps=cfg.total_timesteps, callback=eval_callback)
    
    final_path = os.path.join(cfg.output_dir, f"ppo_reacher_{cfg.n_links}links_final")
    model.save(final_path)
    train_env.close()
    eval_env.close()
    return final_path, log_dir

def generate_video(cfg: Config, model_zip_path: str):
    print(f"\n[4/4] Generating video (Seed: {cfg.video_seed})")
    best_model_path = os.path.join(cfg.output_dir, "best_model.zip")
    load_path = best_model_path if os.path.exists(best_model_path) else model_zip_path
    
    model = PPO.load(load_path, device=cfg.device)
    env = make_reacher_env(cfg.n_links, cfg.episode_len, render_mode="rgb_array", seed=cfg.video_seed)
    
    obs, _ = env.reset()
    frames = []
    for _ in range(cfg.video_frames):
        frames.append(env.render())
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc: obs, _ = env.reset()
    
    env.close()
    out_path = os.path.join(cfg.output_dir, f"reacher_{cfg.n_links}links_fast.mp4")
    imageio.mimsave(out_path, frames, fps=cfg.video_fps)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--n_links", type=int, default=2)
    parser.add_argument("--gemma", type=float, default=0.995, dest="gamma")
    parser.add_argument("--video_seed", type=int, default=999)
    args = parser.parse_args()

    CFG.n_links = args.n_links
    CFG.gamma = args.gamma
    CFG.video_seed = args.video_seed
    CFG.output_dir = f"./experiments_{CFG.n_links}links_5_2000_new/gamma_{CFG.gamma}"
    
    if os.environ.get("MUJOCO_GL") != "egl": os.environ["MUJOCO_GL"] = "egl"
    os.makedirs(CFG.output_dir, exist_ok=True)

    best_params = run_tuning(CFG)
    model_path, log_dir = train_final_model(CFG, best_params)
    plot_reward_curves(CFG, log_dir)
    generate_video(CFG, model_path + ".zip")

if __name__ == "__main__":
    main()