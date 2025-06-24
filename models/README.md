# DQNTrainer: Deep Q-Learning Agent for Credit Limit Optimization

This module implements a Deep Q-Network (DQN) training framework using PyTorch. It is designed to train agents in environments such as `CreditLimitEnv` for tasks like dynamic credit limit adjustment.

---

## Overview

- Builds and trains a DQN agent to learn optimal actions based on environment feedback.
- Supports experience replay and target network stabilization.
- Automatically loads a saved model if available.
- Trains a new model if no saved checkpoint is found.

---

## Components

### 1. `DQN` (Neural Network)
A three-layer fully connected feedforward network:
- Inputs: the current environment state
- Outputs: Q-values for each available action

This architecture is shared between the main (policy) network and the target network.

---

### 2. `ReplayBuffer`
Stores past experiences to enable randomized training batches:
- `push(...)`: stores a tuple of (state, action, reward, next_state, done)
- `sample(batch_size)`: retrieves a random batch for learning

This helps break the correlation between consecutive experiences and improves learning stability.

---

### 3. `DQNTrainer` Class

#### `__init__(...)`
- Initializes the environment, neural networks, optimizer, and replay buffer.
- Loads a model from `model_path` if the file exists.
- Sets the flag `pretrained_loaded = True` if loading succeeds.

#### `train(...)`
Performs training over a number of episodes:
- Selects actions using epsilon-greedy strategy
- Interacts with the environment and stores transitions in the replay buffer
- Updates the policy network using a random sample from the buffer
- Periodically syncs the target network with the policy network
- Saves the model if a new maximum reward is observed

---

### 4. `_optimize_model(...)`
Handles the learning step:
- Computes target Q-values from the target network
- Computes predicted Q-values from the policy network
- Calculates mean squared error loss
- Applies backpropagation with gradient clipping

---

## Configurable Parameters

These arguments can be passed to the `train(...)` method:
- `episodes`: total number of training episodes
- `gamma`: discount factor for future rewards
- `epsilon_start`, `epsilon_end`, `epsilon_decay`: controls exploration
- `batch_size`: size of minibatch sampled from the buffer
- `target_update_freq`: frequency (in episodes) to update the target network

---

## Model Persistence

- If `model_path` is specified, the model is saved to that path after improved performance.
- If `model_path` is `None` (e.g., in debug mode), no model is saved.

---

## Example Usage

```python
trainer = DQNTrainer(env, model_path="models/best_dqn_model.pth")

if not trainer.pretrained_loaded:
    model, _, reward_history, _ = trainer.train(episodes=300)
