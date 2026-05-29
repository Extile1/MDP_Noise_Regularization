#!/bin/bash

# Example usage for running the visualization script
# Make sure to update the paths to point to your actual .zip files

# python visualize.py --models \
# "path/to/gamma_0.99/best_model.zip" \
# "path/to/gamma_0.9/best_model.zip" \
# --titles "Gamma=0.99" "Gamma=0.9"

# Example for noise comparison
# python visualize.py --models \
# "experiments_2links_5_2000/gamma_0.99/best_model.zip" \
# "experiments_2links_5_2000_noise_reset/g0.99/noise_reset/0.05/best_checkpoints/best_model.zip" \
# --titles "Baseline" "Reset Noise 0.05"