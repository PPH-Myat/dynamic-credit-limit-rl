# ------------------------- Neural Network ------------------------- #
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# ------------------------- Experience Replay Buffer ------------------------- #
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        states = np.array(states).astype(np.float32)
        next_states = np.array(next_states).astype(np.float32)

        return (torch.tensor(states, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.int64).unsqueeze(1),
                torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
                torch.tensor(next_states, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32).unsqueeze(1))

    def __len__(self):
        return len(self.buffer)

# ------------------------- Main Training Function ------------------------- #
def train_dqn(env, episodes=10, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05,
              epsilon_decay=0.995, batch_size=64, target_update_freq=10,
              model_path='best_dqn_model.pth'):
    global action, last_action
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = len(env.state_space)
    output_dim = len(env.action_space)

    policy_net = DQN(input_dim, output_dim).to(device)
    target_net = DQN(input_dim, output_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    buffer = ReplayBuffer()
    epsilon = epsilon_start

    max_reward = -float('inf')
    best_state_action_pairs = []
    reward_history = []
    action_counter = {a: 0 for a in env.action_space}

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if np.random.rand() < epsilon:
                action = np.random.choice(env.action_space)
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor([state], dtype=torch.float32).to(device)
                    q_values = policy_net(state_tensor)
                    action = torch.argmax(q_values).item()

            last_action = action

            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            buffer.push(state, action, reward, next_state, done)
            action_counter[action] += 1
            state = next_state

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)

                with torch.no_grad():
                    next_actions = policy_net(next_states).argmax(1, keepdim=True)
                    next_q_values = target_net(next_states).gather(1, next_actions)
                    target_q = rewards + gamma * next_q_values * (1 - dones)

                current_q = policy_net(states).gather(1, actions)
                loss = F.mse_loss(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if total_reward > max_reward:
            max_reward = total_reward
            best_state_action_pairs.append((state,  last_action))
            torch.save(policy_net.state_dict(), model_path)

        reward_history.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        print(f"Episode {episode} | Reward: {total_reward:.3f} | Epsilon: {epsilon:.3f} | Best Reward: {max_reward:.3f}")

    print(f"\nTraining complete! Best Reward: {max_reward:.3f}")
    print("Action Selection Distribution:", action_counter)
    print("Model trained on device:", next(policy_net.parameters()).device)

    return policy_net, best_state_action_pairs, reward_history, action_counter