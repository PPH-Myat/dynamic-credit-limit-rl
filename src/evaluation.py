import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# Ensure results folder exists
os.makedirs("results", exist_ok=True)

class EvaluationUtils:

    # ------------------ Agent Wrapper ------------------ #
    class DQNAgent:
        def __init__(self, model, action_space):
            self.model = model
            self.device = next(model.parameters()).device
            self.action_space = action_space

        def choose_action(self, state):
            state_array = np.asarray(state, dtype=np.float32).reshape(1, -1)
            state_tensor = torch.from_numpy(state_array).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()

    # ------------------ Policy Evaluation ------------------ #
    @staticmethod
    def evaluate_policy(agent, env, n_runs=10, benchmark=None):
        synthetic_rewards = []
        real_rewards = []

        for run in range(n_runs):
            state = env.reset()
            done = False
            total_syn = 0.0
            total_real = 0.0

            while not done:
                if benchmark == "always_increase":
                    action = 1
                elif benchmark == "never_increase":
                    action = 0
                elif benchmark == "random":
                    action = np.random.choice(env.action_space)
                else:
                    action = agent.choose_action(state)

                next_state, reward, done, info = env.step(action)
                total_syn += float(reward)

                # Real reward calculation
                bal   = float(info.get("actual_balance", 0))
                rate  = float(info.get("interest_rate", 0))
                pd    = float(info.get("pd", 0.05))
                lgd   = float(info.get("lgd", 0.5))
                limit = float(info.get("new_limit", 0))
                ccf   = float(info.get("ccf", 0.8))

                ead = bal + ccf * (limit - bal)
                real_reward = 3 * rate * bal * (1 - pd) - pd * lgd * ead
                real_reward = np.clip(real_reward, -1e6, 1e6) / 1e6
                total_real += real_reward

                state = next_state

            synthetic_rewards.append(total_syn)
            real_rewards.append(total_real)

            if os.environ.get("DEBUG", "0") == "1":
                print(f"[EVAL] Run {run+1:02d} | Benchmark={benchmark or 'RL'} | "
                      f"SimReward={total_syn:.4f} | RealReward={total_real:.4f}")

        return synthetic_rewards, real_rewards

    # ------------------ Plot: Sim vs Real Comparison ------------------ #
    @staticmethod
    def plot_policy_comparison(results_dict_sim, results_dict_real):
        labels = list(results_dict_sim.keys())
        x = np.arange(len(labels))
        width = 0.35

        means_sim = [np.mean(results_dict_sim[k]) for k in labels]
        std_sim = [np.std(results_dict_sim[k]) / np.sqrt(len(results_dict_sim[k])) for k in labels]

        means_real = [np.mean(results_dict_real[k]) for k in labels]
        std_real = [np.std(results_dict_real[k]) / np.sqrt(len(results_dict_real[k])) for k in labels]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, means_sim, width, yerr=std_sim, label='Simulated', capsize=4)
        ax.bar(x + width/2, means_real, width, yerr=std_real, label='Real', capsize=4)

        ax.set_ylabel("Average Total Reward")
        ax.set_title("Policy Evaluation: Simulated vs Real Rewards")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig("results/policy_comparison.png")
        plt.show()

    # ------------------ Plot: Training Reward Curve ------------------ #
    @staticmethod
    def plot_training_reward_history(reward_history):
        if reward_history is None or len(reward_history) == 0:
            print("[WARNING] No reward history available to plot.")
            return

        reward_array = np.asarray(reward_history, dtype=np.float32)
        if reward_array.ndim != 1:
            print(f"[ERROR] reward_history must be 1D, got shape: {reward_array.shape}")
            return

        plt.figure(figsize=(8, 4))
        plt.plot(reward_array, marker='o', linewidth=1.5)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("DQN Training Reward Curve")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("results/training_reward.png")
        plt.show()