import time
from stable_baselines3 import PPO
from envs.spiral_env import SpiralEnv
import mujoco
import mujoco.viewer

# Load environment
env = SpiralEnv()

# Load trained model
model = PPO.load("trained_models/spiral_ppo")

# Reset environment
obs, _ = env.reset()

# Create viewer
viewer = mujoco.viewer.launch_passive(env.model, env.data)

print("Running trained policy...")

while viewer.is_running():
    # Predict action from PPO
    action, _ = model.predict(obs, deterministic=True)

    # Step environment
    obs, reward, done, _, _ = env.step(action)

    # Small delay for smooth viewing
    time.sleep(0.01)

    # Reset if success
    if done:
        obs, _ = env.reset()
