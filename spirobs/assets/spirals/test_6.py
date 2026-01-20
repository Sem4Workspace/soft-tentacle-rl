import mujoco
import mujoco.viewer
import numpy as np
import time

# Load model
model = mujoco.MjModel.from_xml_path("stage6_tendon_chain.xml")
data = mujoco.MjData(model)

# Sanity checks
assert model.ntendon == 1, "Expected exactly one tendon"
assert model.nu == 1, "Expected exactly one tendon actuator"

print("\nSTAGE 6 MANUAL TENDON TEST")
print("Expected behavior:")
print("- Chain bends smoothly as a whole")
print("- Motion is slow and elastic")
print("- No jitter, no snapping")
print("- NO NaN / QVEL / QACC warnings\n")

t = 0.0
dt = model.opt.timestep

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        # VERY SMALL length oscillation (safety first)
        length_ctrl = 0.05 * np.sin(0.3 * t)

        data.ctrl[0] = length_ctrl

        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(dt)
        t += dt
