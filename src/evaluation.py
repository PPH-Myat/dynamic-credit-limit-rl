import numpy as np
import torch
import matplotlib.pyplot as plt

# ------------------------- DQN Agent Wrapper ------------------------- #
class DQNAgent:
    def __init__(self, model, action_space):
        self.model = model
        self.device = next(model.parameters()).device
        self.action_space = action_space

    def choose_action(self, state):
        state_tensor = torch.tensor([state], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()


# ------------------------- Policy Evaluation Function ------------------------- #
def evaluate_policy(agent, env, n_runs=10, benchmark=None):
    """
    Evaluate a policy (DQN agent or benchmark) in the given environment.
    Returns both simulated and ground-truth rewards.
    """
    synthetic_rewards = []
    real_rewards = []

    for _ in range(n_runs):
        state = env.reset()
        done = False
        total_syn = 0
        total_real = 0

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

            # Simulated reward
            total_syn += reward

            # Real reward
            bal = info["actual_balance"]
            rate = info["interest_rate"]
            pd = info["pd"]
            lgd = info["lgd"]
            limit = info["new_limit"]
            ccf = info["ccf"]

            ead = bal + ccf * (limit - bal)
            real_reward = 3 * rate * bal * (1 - pd) - pd * lgd * ead
            real_reward = np.clip(real_reward, -1e6, 1e6) / 1e6
            total_real += real_reward

            state = next_state

        synthetic_rewards.append(total_syn)
        real_rewards.append(total_real)

    return synthetic_rewards, real_rewards


# ------------------------- Comparison Bar Plot ------------------------- #
def plot_policy_comparison(results_dict_sim, results_dict_real, title="Policy Evaluation: Sim vs Real Rewards"):
    """
    Grouped bar chart comparing simulated and real rewards for each policy.
    """
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
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()


# ------------------------- Training Reward Plot ------------------------- #
def plot_training_reward_history(reward_history, title="DQN Training Reward Curve"):
    """
    Plot reward per episode during training to visualize learning trend.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(reward_history, marker='o', linewidth=1.5)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()