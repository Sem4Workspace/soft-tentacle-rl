from envs.spiral_env import SpiralEnv
import numpy as np

env = SpiralEnv()

obs, _ = env.reset()

for i in range(1000):
    action = np.random.uniform(-1, 1, 2)
    obs, reward, done, _, _ = env.step(action)
    print(reward)
    if done:
        break
