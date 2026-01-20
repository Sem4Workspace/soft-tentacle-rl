import gymnasium as gym
from stable_baselines3 import PPO
from envs.spiral_env import SpiralEnv
import os

# Create environment
env = SpiralEnv()

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    gamma=0.99,
    n_steps=2048,
    batch_size=64,
    ent_coef=0.01,
)

# Train
model.learn(total_timesteps=300_000)

# Save trained model
os.makedirs("trained_models", exist_ok=True)
model.save("trained_models/spiral_ppo")

print("Training complete. Model saved.")
