import numpy as np

class SpiRobEnv:
    """
    Geometry-based RL environment for a 2-cable SpiRob.
    No forces, no physics, no dynamics.
    """

    def __init__(self):
        # Target (fixed in workspace)
        self.target_pos = np.array([2.0, 0.5])

        # State variables
        self.l = None        # extension ratio [0,1]
        self.phi = None      # orientation (rad)
        self.contact = None  # contact flag
        self.done = False

        self.max_steps = 100
        self.step_count = 0

        self.reset()

    def reset(self):
        """Reset episode"""
        self.l = 0.2                    # start curled
        self.phi = 0.0                  # facing forward
        self.contact = 0
        self.done = False
        self.step_count = 0

        return self._get_state()

    def _tip_position(self):
        """
        Geometric tip position.
        Extension controls reach, orientation controls direction.
        """
        reach = 3.0 * self.l
        x = reach * np.cos(self.phi)
        y = reach * np.sin(self.phi)
        return np.array([x, y])

    def _get_state(self):
        """Return state vector"""
        tip = self._tip_position()
        diff = self.target_pos - tip

        d = np.linalg.norm(diff)
        alpha = np.arctan2(diff[1], diff[0]) - self.phi

        return np.array([
            self.l,
            self.phi,
            self.contact,
            d,
            alpha
        ], dtype=np.float32)

    def step(self, action):
        """
        Actions:
        0 = pull left cable
        1 = pull right cable
        2 = pull both cables (uncurl)
        3 = hold
        """
        if self.done:
            raise RuntimeError("Episode already terminated")

        self.step_count += 1

        # --- Action effects (geometry only) ---
        if action == 0:  # left pull
            self.phi += 0.15
            self.l = max(0.0, self.l - 0.05)

        elif action == 1:  # right pull
            self.phi -= 0.15
            self.l = max(0.0, self.l - 0.05)

        elif action == 2:  # symmetric pull (uncurl)
            self.l = min(1.0, self.l + 0.08)

        elif action == 3:  # hold
            pass

        else:
            raise ValueError("Invalid action")

        # --- Geometry-based contact detection ---
        tip = self._tip_position()
        dist = np.linalg.norm(self.target_pos - tip)

        if dist < 0.3 and self.contact == 0:
            self.contact = 1   # first contact

        # --- Reward ---
        reward = -dist

        if self.contact:
            reward += 2.0

        # Successful wrap condition
        if self.contact and self.l < 0.3:
            reward += 10.0
            self.done = True

        # Timeout
        if self.step_count >= self.max_steps:
            self.done = True

        return self._get_state(), reward, self.done, {}
