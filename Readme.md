# Reacher PPO Training & Analysis

This repository contains code to train PPO agents on a custom 2-Link Reacher MuJoCo environment, robustly handle noise (random actions, reset).

## File Structure

- `reacher_env.py`: Core file. Contains the NLinkReacherEnv class and Noise Wrappers.

- `train_basic.py`: Train standard PPO agents with Optuna tuning.

- `train_noise.py`: Train PPO agents specifically robust to noise (Reset, Sticky, Random).

- `visualize.py`: Visualize agent trajectories compared to target.

- `analyze_jsd.py`: Calculate Jensen-Shannon Divergence between policies.

- `heatmap_noisy_visualizatiopn.py`: Generate heatmaps comparing Gamma performance against Noise levels under clean environment.

## Basic Usage

Train a standard agent with 2 links and Gamma 0.99.

```
python train_basic.py -k 2 --gemma 0.99
```

Train an agent on a 2-link arm with 5% "Reset" noise.

```
python train_noise.py -k 2 --gemma 0.99 --noise_type reset --noise_prob 0.05
```

Compare multiple trained models visually based on a fixed random seed

```
python visualize.py --models path/to/model1.zip path/to/model2.zip --titles "Model A" "Model B"
```

Run the analysis scripts to generate plots

```
# Replicate Figure 7
python analyze_jsd.py

# Replicate Appendix D
python heatmap_noisy_visualization.py
```