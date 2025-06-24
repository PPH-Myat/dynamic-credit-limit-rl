import numpy as np
import torch
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

from src.rl_env import CreditLimitEnv
from src.dqn import DQNTrainer
from src.evaluation import EvaluationUtils

# === CONFIG ===
EPISODES = 500
N_RUNS = 20
MODEL_PATH = "models/best_dqn_model.pth"
DATA_PATH = "data/processed/cleaned_df.csv"

print("=== RL Pipeline Started ===")

# === Step 1: Load dataset ===
print("[INFO] Loading dataset...")
final_df = pd.read_csv(DATA_PATH)
print(f"[INFO] Loaded {len(final_df):,} rows.")

# === Step 2: Setup environment ===
provision_bins = np.arange(-0.5, 1.51, 0.01)
globals()["provision_bins"] = provision_bins
env = CreditLimitEnv(final_df, provision_bins=provision_bins)
print("[INFO] Environment initialized.")

# === Step 3: Initialize trainer ===
trainer = DQNTrainer(env, model_path=MODEL_PATH)
model = trainer.policy_net

# === Step 4: Train only if no pretrained model was loaded ===
if not trainer.pretrained_loaded:
    model, _, reward_history, _ = trainer.train(episodes=EPISODES)
    EvaluationUtils.plot_training_reward_history(reward_history)

# === Step 5: Wrap model as RL agent ===
agent = EvaluationUtils.DQNAgent(model, env.action_space)

# === Step 6: Evaluate policies ===
print("[INFO] Evaluating agent and benchmarks...")
rl_sim, rl_real     = EvaluationUtils.evaluate_policy(agent, env, n_runs=N_RUNS)
rand_sim, rand_real = EvaluationUtils.evaluate_policy(None, env, n_runs=N_RUNS, benchmark="random")
no_sim, no_real     = EvaluationUtils.evaluate_policy(None, env, n_runs=N_RUNS, benchmark="never_increase")
all_sim, all_real   = EvaluationUtils.evaluate_policy(None, env, n_runs=N_RUNS, benchmark="always_increase")

# === Step 7: Package results ===
sim_results = {
    "RL Agent": rl_sim,
    "Random": rand_sim,
    "Never Increase": no_sim,
    "Always Increase": all_sim
}

real_results = {
    "RL Agent": rl_real,
    "Random": rand_real,
    "Never Increase": no_real,
    "Always Increase": all_real
}

# === Step 8: Print summary ===
print("\n=== Simulation Reward Summary ===")
for name, rewards in sim_results.items():
    print(f"{name:17s}: {np.mean(rewards):.4f}")

print("\n=== Real Reward Summary ===")
for name, rewards in real_results.items():
    print(f"{name:17s}: {np.mean(rewards):.4f}")

# === Step 9: Plot comparisons ===
EvaluationUtils.plot_policy_comparison(sim_results, real_results)

print("=== RL Pipeline Completed ===")