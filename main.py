import numpy as np
import torch
import pandas as pd
import os

from src.rl_env import CreditLimitEnv
from src.dqn import DQN, train_dqn
from src.evaluation import (
    DQNAgent,
    evaluate_policy,
    plot_policy_comparison,
    plot_training_reward_history
)

# === Step 1: Load processed dataset ===
final_df = pd.read_csv("data/processed/cleaned_df.csv")

# === Step 2: Define global provision bins (required by env) ===
provision_bins = np.arange(-0.5, 1.51, 0.01)
globals()["provision_bins"] = provision_bins

# === Step 3: Initialize environment ===
env = CreditLimitEnv(final_df, provision_bins=provision_bins)

# === Step 4: Load or train DQN model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = len(env.state_space)
output_dim = len(env.action_space)
model_path = "models/best_dqn_model.pth"

model = DQN(input_dim, output_dim).to(device)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Loaded pretrained model.")
    reward_history = None
else:
    model, reward_history, _, _ = train_dqn(env, episodes=50, model_path=model_path)
    print("✅ Trained new model.")

# === Step 5: Plot training reward history if available ===
if reward_history:
    plot_training_reward_history(reward_history)

# === Step 6: Wrap model into agent ===
agent = DQNAgent(model, env.action_space)

# === Step 7: Evaluate RL agent and benchmarks ===
rl_sim, rl_real     = evaluate_policy(agent, env, n_runs=5)
rand_sim, rand_real = evaluate_policy(None, env, n_runs=5, benchmark="random")
no_sim, no_real     = evaluate_policy(None, env, n_runs=5, benchmark="never_increase")
all_sim, all_real   = evaluate_policy(None, env, n_runs=5, benchmark="always_increase")

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

# === Step 9: Print evaluation summary ===
print("\n=== Average Total Reward by Policy ===")
print("Simulated Reward:")
for name, rewards in sim_results.items():
    print(f"{name:17s}: {np.mean(rewards):.4f}")

print("\nGround-Truth Reward:")
for name, rewards in real_results.items():
    print(f"{name:17s}: {np.mean(rewards):.4f}")

# === Step 10: Plot reward comparison chart ===
plot_policy_comparison(sim_results, real_results, title="Policy Evaluation: Simulated vs Real Reward")