import pandas as pd

from src.classifier import
from src.rl_env import
from src.dqn import DQN
from src.evaluation import

final_df = pd.read_csv('data/processed/final_data.csv')

# Initialize environment
env = CreditLimitEnv(final_df)

# Quick environment sanity check
print("Testing env.reset() and env.step() once before training...")
state = env.reset()
action = np.random.choice(env.action_space)
next_state, reward, done, _ = env.step(action)
print(f"Initial state: {state} → corresponds to ['BALANCE_CLASS', 'UR', 'PR', 'D_PROVISION_bin']")
print(f"Action: {action}, Next state: {next_state}, Reward: {reward}, Done: {done}")
print("Environment test passed, proceeding to training...\n")

# === Training Phase ===
if __name__ == "__main__":
    env = CreditLimitEnv(final_df)
    trained_model, best_cases, reward_history, action_counter = train_dqn(
        env, episodes=150, gamma=0.9
    )

    # Save trained model and results
    torch.save(trained_model.state_dict(), "best_dqn_model.pth")

    with open("best_state_action.pkl", "wb") as f:
        pickle.dump(best_cases, f)

    # Plot training reward curve
    plt.plot(reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Reward Curve")
    plt.grid(True)
    plt.show()

    print("Best state-action pairs saved to best_state_action.pkl\n")

# === Evaluation Phase ===
if __name__ == "__main__":
    env = CreditLimitEnv(final_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = len(env.state_space)
    output_dim = len(env.action_space)

    model = DQN(input_dim, output_dim).to(device)
    model.load_state_dict(torch.load("best_dqn_model.pth", map_location=device))
    model.eval()

    agent = DQNAgent(model, env.action_space)

    # Evaluate trained agent vs benchmarks
    rl_rewards = evaluate_policy(agent, env, n_runs=20)
    rand_rewards = evaluate_policy(None, env, n_runs=20, benchmark="random")
    no_rewards = evaluate_policy(None, env, n_runs=20, benchmark="never_increase")
    all_rewards = evaluate_policy(None, env, n_runs=20, benchmark="always_increase")

    print(f"RL Avg Reward:          {np.mean(rl_rewards):.2f}")
    print(f"Random Avg Reward:      {np.mean(rand_rewards):.2f}")
    print(f"Never Increase Reward:  {np.mean(no_rewards):.2f}")
    print(f"Always Increase Reward: {np.mean(all_rewards):.2f}")

    # Plot evaluation results
    plt.figure(figsize=(8, 5))
    plt.bar(
        ['RL', 'Random', 'Never ↑', 'Always ↑'],
        [np.mean(rl_rewards), np.mean(rand_rewards), np.mean(no_rewards), np.mean(all_rewards)],
        color=['blue', 'gray', 'red', 'green']
    )
    plt.title("Policy Evaluation (Avg Rewards over 20 runs)")
    plt.ylabel("Average Total Reward")
    plt.grid(True)
    plt.show()
