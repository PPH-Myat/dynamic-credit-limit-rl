from src.classifier_regressor import
from src.rl_env import
from src.dqn import DQN
from src.evaluation import

# RL environment
# add provision_bins
# provision_bins = np.arange(-0.5, 1.51, 0.01)
env = CreditLimitEnv(final_df)
# test env
state = env.reset()
action = np.random.choice(env.action_space)
next_state, reward, done, _ = env.step(action)
print(f"初始状态: {state} -> 表示:['BALANCE_CLASS', 'UR', 'PR', 'D_PROVISION_bin']")
print(f"action: {action}, next_state: {next_state}, reward: {reward}, done: {done}")

print("Testing env.reset() and env.step() once before training...")
test_state = env.reset()
test_next, test_reward, test_done, _ = env.step(random.choice(env.action_space))
print("env works, starting training...")

# DQN neural network
if __name__ == '__main__':
    env = CreditLimitEnv(final_df)
    trained_model, best_cases, reward_history, action_counter = train_dqn(env, episodes=150, gamma=0.9)

    torch.save(trained_model.state_dict(), 'best_dqn_model.pth')

    with open('best_state_action.pkl', 'wb') as f:
        pickle.dump(best_cases, f)

    plt.plot(reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Reward Curve")
    plt.grid(True)
    plt.show()

    print("\nBest state-action pairs saved to best_state_action.pkl")

# Evaluation
if __name__ == "__main__":
    # Ensure the environment is instantiated
    env = CreditLimitEnv(final_df)  # Replace with your actual env constructor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = len(env.state_space)
    output_dim = len(env.action_space)

    # Load trained model
    model = DQN(input_dim, output_dim).to(device)
    model.load_state_dict(torch.load("best_dqn_model.pth", map_location=device))
    model.eval()

    agent = DQNAgent(model, env.action_space)

    # Evaluate RL agent and benchmarks
    rl_rewards = evaluate_policy(agent, env, n_runs=20)
    rand_rewards = evaluate_policy(None, env, n_runs=20, benchmark="random")
    no_rewards = evaluate_policy(None, env, n_runs=20, benchmark="never_increase")
    all_rewards = evaluate_policy(None, env, n_runs=20, benchmark="always_increase")

    # Print average rewards
    print(f"RL Avg Reward:          {np.mean(rl_rewards):.2f}")
    print(f"Random Avg Reward:      {np.mean(rand_rewards):.2f}")
    print(f"Never Increase Reward:  {np.mean(no_rewards):.2f}")
    print(f"Always Increase Reward: {np.mean(all_rewards):.2f}")

    # Plot comparison
    plt.figure(figsize=(8, 5))
    plt.bar(['RL', 'Random', 'Never ↑', 'Always ↑'],
            [np.mean(rl_rewards), np.mean(rand_rewards), np.mean(no_rewards), np.mean(all_rewards)],
            color=['blue', 'gray', 'red', 'green'])
    plt.title("Policy Evaluation (Avg Rewards over 20 runs)")
    plt.ylabel("Average Total Reward")
    plt.grid(True)
    plt.show()

