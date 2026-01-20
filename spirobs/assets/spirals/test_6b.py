import mujoco
import mujoco.viewer
import numpy as np
import time

# Load model
model = mujoco.MjModel.from_xml_path("stage6b_dual_tendon_chain.xml")
data = mujoco.MjData(model)

# Sanity checks
assert model.ntendon == 2, "Expected exactly two tendons"
assert model.nu == 2, "Expected exactly two tendon actuators"

print("\nSTAGE 6B DUAL TENDON MANUAL TEST")
print("Expected behavior:")
print("- Tendon_pos shortens → robot bends +X direction")
print("- Tendon_neg shortens → robot bends -X direction")
print("- Motion is smooth and elastic")
print("- NO jitter, NO snapping, NO warnings\n")

t = 0.0
dt = model.opt.timestep

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        # Differential tendon control (SAFE AMPLITUDE)
        ctrl_pos =  0.05 * np.sin(0.3 * t)
        ctrl_neg = -0.05 * np.sin(0.3 * t)

        data.ctrl[0] = ctrl_pos
        data.ctrl[1] = ctrl_neg

        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(dt)
        t += dt
