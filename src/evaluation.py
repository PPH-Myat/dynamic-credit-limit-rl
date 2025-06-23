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
    rewards = []
    for _ in range(n_runs):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            if benchmark == "always_increase":
                action = 1
            elif benchmark == "never_increase":
                action = 0
            elif benchmark == "random":
                action = np.random.choice(env.action_space)
            else:
                action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
        rewards.append(total_reward)
    return rewards
