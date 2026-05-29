import os
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Import shared env
from reacher_env import NLinkReacherEnv, make_reacher_env

def get_arm_joints(env, n_links):
    """Returns list of (x, y) coordinates for arm."""
    points = [[0.0, 0.0]] # Root
    for i in range(1, n_links):
        pos = env.data.body(f"body{i}").xpos
        points.append([pos[0], pos[1]])
    tip = env.data.body("fingertip").xpos
    points.append([tip[0], tip[1]])
    return np.array(points)

def plot_shadow_trajectory(ax, model_path, n_links, seed, title, cmap_name="Blues"):
    print(f"Plotting: {title}")
    # We use base environment for viz (no noise wrappers usually desired for clean traj check)
    env = NLinkReacherEnv(n_links=n_links)
    env.reset(seed=seed)
    
    model = None
    if model_path and os.path.exists(model_path):
        try:
            model = PPO.load(model_path)
        except:
            print(f"Warning: Could not load {model_path}")

    obs, _ = env.reset(seed=seed)
    target_pos = env.get_body_com("target")[:2].copy()
    
    arm_positions = []
    for _ in range(2000):
        if (_+1) % 10 != 0:
            arm_positions.append(get_arm_joints(env, n_links))
        
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)

    # Plot Target
    ax.scatter(target_pos[0], target_pos[1], c='red', s=200, marker='*', zorder=10)
    
    # Plot Trajectory
    cmap = plt.get_cmap(cmap_name)
    n_frames = len(arm_positions)
    for t, joints in enumerate(arm_positions):
        prog = t / n_frames
        color = cmap(0.4 + 0.6 * prog)
        ax.plot(joints[:, 0], joints[:, 1], color=color, alpha=0.2+0.8*prog, linewidth=3)
        ax.scatter(joints[:, 0], joints[:, 1], color=color, alpha=0.2+0.8*prog, s=20)

    # Start/End
    ax.plot(arm_positions[0][:, 0], arm_positions[0][:, 1], 'k--', alpha=0.3)
    ax.plot(arm_positions[-1][:, 0], arm_positions[-1][:, 1], color=cmap(1.0), linewidth=3)
    
    ax.set_title(title)
    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.25, 0.25)
    ax.axis('off')
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs='+', required=True)
    parser.add_argument("--titles", nargs='+')
    parser.add_argument("--n_links", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2000)
    args = parser.parse_args()

    num = len(args.models)
    titles = args.titles if args.titles and len(args.titles) == num else [os.path.basename(p) for p in args.models]
    
    n_cols = 5 if num > 2 else num
    n_rows = math.ceil(num / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    axes_flat = [axes] if num == 1 else axes.flatten()
    
    cmaps = ["Blues", "Greens", "Purples", "Oranges", "Greys", "Reds", "YlOrBr"]

    for i in range(num):
        plot_shadow_trajectory(axes_flat[i], args.models[i], args.n_links, args.seed, titles[i], cmaps[i%len(cmaps)])
    
    for j in range(num, len(axes_flat)): axes_flat[j].axis('off')
    
    plt.tight_layout()
    plt.savefig("trajectory_comparison.png", dpi=150)
    print("Saved trajectory_comparison.png")