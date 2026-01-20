import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

from spirob_env import SpiRobEnv

# -------------------------------------------------
# Safety: enforce float32 everywhere
# -------------------------------------------------
torch.set_default_dtype(torch.float32)

# -------------------------------------------------
# DQN Network
# -------------------------------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------
# Replay Buffer
# -------------------------------------------------
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = []
        self.size = size

    def push(self, transition):
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# -------------------------------------------------
# Rule-Based Baseline (for comparison)
# -------------------------------------------------
def rule_based_policy(state):
    l, phi, contact, d, alpha = state

    if contact:
        return 0  # curl
    if abs(alpha) > 0.2:
        return 0 if alpha > 0 else 1
    return 2  # uncurl


# -------------------------------------------------
# Training Parameters
# -------------------------------------------------
EPISODES = 400
GAMMA = 0.98
LR = 1e-3

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 300

BATCH_SIZE = 64
MEM_SIZE = 5000


# -------------------------------------------------
# Environment & Networks
# -------------------------------------------------
env = SpiRobEnv()

state_dim = 5
action_dim = 4

policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
buffer = ReplayBuffer(MEM_SIZE)


# -------------------------------------------------
# Training Loop
# -------------------------------------------------
episode_rewards = []
success_flags = []

for ep in range(EPISODES):
    state = env.reset()
    total_reward = 0.0

    eps = EPS_END + (EPS_START - EPS_END) * np.exp(-ep / EPS_DECAY)
    done = False

    while not done:
        # ε-greedy action
        if random.random() < eps:
            action = random.randint(0, action_dim - 1)
        else:
            with torch.no_grad():
                q_vals = policy_net(torch.tensor(state))
                action = q_vals.argmax().item()

        next_state, reward, done, _ = env.step(action)

        buffer.push((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        # ---- Learn ----
        if len(buffer) >= BATCH_SIZE:
            batch = buffer.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.tensor(np.array(states), dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
            q_next = target_net(next_states).max(1)[0].detach()
            q_target = rewards + GAMMA * q_next * (1 - dones)

            loss = nn.MSELoss()(q, q_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    target_net.load_state_dict(policy_net.state_dict())

    episode_rewards.append(total_reward)
    success_flags.append(1 if env.contact and env.l < 0.3 else 0)

    if (ep + 1) % 50 == 0:
        print(f"Episode {ep+1}/{EPISODES} | Reward: {total_reward:.2f}")


# -------------------------------------------------
# Evaluation
# -------------------------------------------------
def evaluate(policy_fn, episodes=100):
    success = 0
    for _ in range(episodes):
        s = env.reset()
        done = False
        while not done:
            a = policy_fn(s)
            s, _, done, _ = env.step(a)
        if env.contact and env.l < 0.3:
            success += 1
    return success / episodes


rl_success = evaluate(
    lambda s: policy_net(torch.tensor(s)).argmax().item()
)

baseline_success = evaluate(rule_based_policy)

print("\nEvaluation Results:")
print(f"Baseline success rate: {baseline_success:.2f}")
print(f"RL success rate:       {rl_success:.2f}")


# -------------------------------------------------
# Plots
# -------------------------------------------------
plt.figure()
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Learning Curve")
plt.grid()

plt.figure()
plt.bar(["Baseline", "RL"], [baseline_success, rl_success])
plt.ylabel("Success Rate")
plt.title("Wrapping Success Comparison")
plt.grid()

plt.show()
