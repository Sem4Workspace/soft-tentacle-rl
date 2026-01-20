import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import time


class TendonReachEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render=True):
        super().__init__()

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(
            "stage7_dual_tendon_with_ball.xml"
        )
        self.data = mujoco.MjData(self.model)

        # IDs (NO MAGIC NUMBERS)
        self.tip_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "tip"
        )
        self.target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )

        assert self.tip_site_id != -1
        assert self.target_body_id != -1

        # Action space: tendon length commands
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(2,), dtype=np.float32
        )

        # Observation:
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
        return self._get_obs(), {}

    def step(self, action):
        # Hard safety clip
        action = np.clip(action, -0.1, 0.1)

        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        tip_pos = self.data.site_xpos[self.tip_site_id]
        target_pos = self.data.xpos[self.target_body_id]
        distance = np.linalg.norm(tip_pos - target_pos)

        reward = -distance

        terminated = False
        truncated = False
        info = {"distance": distance}

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.copy(),
            self.data.qvel.copy(),
            self.data.site_xpos[self.tip_site_id].copy(),
            self.data.xpos[self.target_body_id].copy(),
        ])

    def render(self):
        if not self.render_enabled:
            return
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data
            )
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# ============================
# RANDOM ACTION TEST
# ============================
if __name__ == "__main__":
    env = TendonReachEnv(render=True)

    obs, _ = env.reset()

    print("\nSTAGE 8 TENDON RANDOM ACTION TEST")
    print("Expected behavior:")
    print("- Robot bends randomly (left/right)")
    print("- Distance always finite")
    print("- No instability or warnings\n")

    for step in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 100 == 0:
            print(f"Step {step} | Distance: {info['distance']:.4f}")

        env.render()
        time.sleep(env.model.opt.timestep)

    env.close()
