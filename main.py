import numpy as np
import torch
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

from src.rl_env import CreditLimitEnv
from src.dqn import DQN, train_dqn
from src.evaluation import EvaluationUtils

# === CONFIG ===
EPISODES = 500     # RL episodes
N_RUNS = 20        # Evaluation runs
MODEL_PATH = "models/best_dqn_model.pth"
DATA_PATH = "data/processed/cleaned_df.csv"

print("=== RL Pipeline Started ===")

# === Step 1: Load dataset ===
print("[INFO] Loading dataset...")
final_df = pd.read_csv(DATA_PATH)
print(f"[INFO] Loaded {len(final_df):,} rows.")

# === Step 2: Configure device (GPU/CPU) ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# === Step 3: Setup environment ===
provision_bins = np.arange(-0.5, 1.51, 0.01)
globals()["provision_bins"] = provision_bins
env = CreditLimitEnv(final_df, provision_bins=provision_bins)
print("[INFO] Environment initialized.")

# === Step 4: Initialize or load model ===
input_dim = len(env.state_space)
output_dim = len(env.action_space)
model = DQN(input_dim, output_dim).to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    reward_history = None
    print(f"[INFO] Loaded trained model from {MODEL_PATH}")
else:
    print("[INFO] No saved model found. Starting training...")

# === Step 5: Plot training rewards ===
if reward_history is not None and len(reward_history) > 0:
    EvaluationUtils.plot_training_reward_history(reward_history)

# === Step 6: Wrap trained model as agent ===
agent = EvaluationUtils.DQNAgent(model, env.action_space)

# === Step 7: Evaluate policies ===
print("[INFO] Evaluating agent and benchmarks...")
rl_sim, rl_real     = EvaluationUtils.evaluate_policy(agent, env, n_runs=N_RUNS)
rand_sim, rand_real = EvaluationUtils.evaluate_policy(None, env, n_runs=N_RUNS, benchmark="random")
no_sim, no_real     = EvaluationUtils.evaluate_policy(None, env, n_runs=N_RUNS, benchmark="never_increase")
all_sim, all_real   = EvaluationUtils.evaluate_policy(None, env, n_runs=N_RUNS, benchmark="always_increase")

# === Step 8: Package results ===
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

# === Step 9: Print summary ===
print("\n=== Simulation Reward Summary ===")
for name, rewards in sim_results.items():
    print(f"{name:17s}: {np.mean(rewards):.4f}")

print("\n=== Real Reward Summary ===")
for name, rewards in real_results.items():
    print(f"{name:17s}: {np.mean(rewards):.4f}")

# === Step 10: Plot comparisons ===
EvaluationUtils.plot_policy_comparison(sim_results, real_results)

print("=== RL Pipeline Completed ===")