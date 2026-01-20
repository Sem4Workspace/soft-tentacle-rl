import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import time


class ContinuumReachEnv(gym.Env):
    """
    Stage 4:
    - Gymnasium wrapper
    - Random actions
    - NO PPO
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render=True):
        super().__init__()

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path("stage3_chain_with_ball.xml")
        self.data = mujoco.MjData(self.model)

        # IDs
        self.tip_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "tip"
        )
        self.target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )

        # Action space: 5 joint motors
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # Observation space:
        # joint positions (5)
        # joint velocities (5)
        # tip position (3)
        # target position (3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32
        )

        self.render_enabled = render
        self.viewer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Clip action for safety (hard guarantee)
        action = np.clip(action, -1.0, 1.0)

        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        # Distance-based reward (simple, stable)
        tip_pos = self.data.site_xpos[self.tip_site_id]
        target_pos = self.data.xpos[self.target_body_id]
        distance = np.linalg.norm(tip_pos - target_pos)

        reward = -distance

        terminated = False
        truncated = False
        info = {"distance": distance}

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        tip_pos = self.data.site_xpos[self.tip_site_id].copy()
        target_pos = self.data.xpos[self.target_body_id].copy()

        return np.concatenate([qpos, qvel, tip_pos, target_pos])

    def render(self):
        if not self.render_enabled:
            return

        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# ============================
# RANDOM ACTION TEST
# ============================
if __name__ == "__main__":
    env = ContinuumReachEnv(render=True)

    obs, _ = env.reset()
    print("\nSTAGE 4 RANDOM ACTION TEST RUNNING")
    print("Expected behavior:")
    print("- Robot moves randomly")
    print("- No instability")
    print("- Distance stays finite")
    print("- Viewer remains responsive\n")

    for step in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 100 == 0:
            print(f"Step {step} | Distance: {info['distance']:.4f}")

        env.render()
        time.sleep(env.model.opt.timestep)

    env.close()
