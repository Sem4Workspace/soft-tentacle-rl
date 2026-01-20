import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class TendonReachEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render=False):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(
            "stage7_dual_tendon_with_ball.xml"
        )
        self.data = mujoco.MjData(self.model)

        self.tip_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "tip"
        )
        self.target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )

        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(2,), dtype=np.float32
        )

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
# PPO TRAINING (FINAL)
# ============================
if __name__ == "__main__":

    def make_env():
        return TendonReachEnv(render=False)

    env = DummyVecEnv([make_env])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        device="cpu",
    )

    print("\nSTAGE 9 PPO TRAINING (TENDON REACHING)")
    print("Expected behavior:")
    print("- No NaNs")
    print("- Mean reward improves")
    print("- Distance decreases over time")
    print("- No MuJoCo warnings\n")

    model.learn(total_timesteps=60_000)

    print("\nTraining complete. Visualizing learned policy...\n")

    test_env = TendonReachEnv(render=True)
    obs, _ = test_env.reset()

    for _ in range(1500):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _, info = test_env.step(action)
        test_env.render()
        time.sleep(test_env.model.opt.timestep)

    test_env.close()
