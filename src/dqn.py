import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time
import os
from collections import deque
from tqdm import trange

# ------------------------- Neural Network ------------------------- #
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

# ------------------------- Experience Replay Buffer ------------------------- #
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, float(reward), next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        return (
            torch.tensor(np.stack(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.stack(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)

# ------------------------- Main Training Function ------------------------- #
def train_dqn(env, episodes=300, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05,
              epsilon_decay=0.99, batch_size=128, target_update_freq=5,
              model_path='best_dqn_model.pth'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = len(env.state_space)
    output_dim = len(env.action_space)

    policy_net = DQN(input_dim, output_dim).to(device)
    target_net = DQN(input_dim, output_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    buffer = ReplayBuffer()
    epsilon = epsilon_start

    max_reward = -float('inf')
    reward_history = []
    best_state_action_pairs = []
    action_counter = {a: 0 for a in env.action_space}

    start_time = time.time()

    for episode in trange(episodes, desc="Training Episodes"):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if np.random.rand() < epsilon:
                action = np.random.choice(env.action_space)
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    q_values = policy_net(state_tensor)
                    action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            total_reward += reward
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
                    next_q = target_net(next_states).gather(1, policy_net(next_states).argmax(1, keepdim=True))
                    target_q = rewards + gamma * next_q * (1 - dones)

                current_q = policy_net(states).gather(1, actions)
                loss = F.mse_loss(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)  # gradient clipping
                optimizer.step()

        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if total_reward > max_reward:
            max_reward = total_reward
            best_state_action_pairs.append((state.copy(), action))
            if model_path:
                torch.save(policy_net.state_dict(), model_path)

        reward_history.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if os.environ.get("DEBUG", "0") == "1":
            print(f"[DEBUG] Episode {episode+1} | Reward: {total_reward:.3f} | Buffer: {len(buffer)} | Action Count: {action_counter}")

    duration = time.time() - start_time
    print(f"\nTraining complete in {duration / 60:.2f} minutes")
    print("Best Episode Reward :", max_reward)
    print("Final Action Count   :", action_counter)

    return policy_net, best_state_action_pairs, reward_history, action_counter