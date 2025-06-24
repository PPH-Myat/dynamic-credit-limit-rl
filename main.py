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
EPISODES = 500  # RL episodes
N_RUNS = 5      # Evaluation runs

# === Step 1: Load processed dataset ===
final_df = pd.read_csv("data/processed/cleaned_df.csv")

# === Step 2: Configure device (GPU or CPU) ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Step 3: Set global provision bins & initialize environment ===
provision_bins = np.arange(-0.5, 1.51, 0.01)
globals()["provision_bins"] = provision_bins
env = CreditLimitEnv(final_df, provision_bins=provision_bins)

# === Step 4: Load or train DQN model ===
input_dim = len(env.state_space)
output_dim = len(env.action_space)
model_path = "models/best_dqn_model.pth"
model = DQN(input_dim, output_dim).to(device)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    reward_history = None
else:
    model, _, reward_history, _ = train_dqn(env, episodes=EPISODES, model_path=model_path)

# === Step 5: Plot training rewards (if available) ===
if reward_history is not None and len(reward_history) > 0:
    EvaluationUtils.plot_training_reward_history(reward_history)

# === Step 6: Wrap trained model as an agent ===
agent = EvaluationUtils.DQNAgent(model, env.action_space)

# === Step 7: Evaluate RL agent and benchmarks ===
rl_sim, rl_real     = EvaluationUtils.evaluate_policy(agent, env, n_runs=N_RUNS)
rand_sim, rand_real = EvaluationUtils.evaluate_policy(None, env, n_runs=N_RUNS, benchmark="random")
no_sim, no_real     = EvaluationUtils.evaluate_policy(None, env, n_runs=N_RUNS, benchmark="never_increase")
all_sim, all_real   = EvaluationUtils.evaluate_policy(None, env, n_runs=N_RUNS, benchmark="always_increase")

# === Step 8: Collect simulation and real-world reward results ===
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

# === Step 9: Print evaluation summaries ===
for name, rewards in sim_results.items():
    print(f"{name:17s}: {np.mean(rewards):.4f}")

for name, rewards in real_results.items():
    print(f"{name:17s}: {np.mean(rewards):.4f}")

# === Step 10: Visualize policy comparisons ===
EvaluationUtils.plot_policy_comparison(sim_results, real_results)
