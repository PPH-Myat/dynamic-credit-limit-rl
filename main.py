import numpy as np
import torch
import pandas as pd

from src.rl_env import CreditLimitEnv
from src.dqn import DQN
from src.evaluation import DQNAgent, evaluate_policy, plot_policy_comparison

# === Step 1: Load processed dataset ===
final_df = pd.read_csv("data/processed/final_df.csv")

# === Step 2: Define provision bins globally (must be set before env init) ===
provision_bins = np.arange(-0.5, 1.51, 0.01)
globals()["provision_bins"] = provision_bins

# === Step 3: Initialize environment ===
env = CreditLimitEnv(final_df)

# === Step 4: Load trained DQN model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = len(env.state_space)
output_dim = len(env.action_space)

model = DQN(input_dim, output_dim).to(device)
model.load_state_dict(torch.load("models/best_dqn_model.pth", map_location=device))
model.eval()

agent = DQNAgent(model, env.action_space)

# === Step 5: Evaluate agent and benchmark policies ===
rl_rewards   = evaluate_policy(agent, env, n_runs=20)
rand_rewards = evaluate_policy(None, env, n_runs=20, benchmark="random")
no_rewards   = evaluate_policy(None, env, n_runs=20, benchmark="never_increase")
all_rewards  = evaluate_policy(None, env, n_runs=20, benchmark="always_increase")

# === Step 6: Print average reward results ===
results = {
    "RL Agent": rl_rewards,
    "Random": rand_rewards,
    "Never Increase": no_rewards,
    "Always Increase": all_rewards
}

print("\n=== Average Total Reward by Policy ===")
for name, rewards in results.items():
    print(f"{name:17s}: {np.mean(rewards):.4f}")

# === Step 7: Plot comparison ===
plot_policy_comparison(results, title="Policy Evaluation (Avg Reward over 20 runs)")