import mujoco
import mujoco.viewer
import numpy as np
import time

# Load model
model = mujoco.MjModel.from_xml_path("stage3_chain_with_ball.xml")
data = mujoco.MjData(model)

# Get IDs (NO MAGIC NUMBERS)
tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")

assert tip_site_id != -1, "Tip site not found"
assert target_body_id != -1, "Target body not found"

print("\nSTAGE 3 DISTANCE TEST RUNNING")
print("Expected behavior:")
print("- Robot bends as before")
print("- Printed distance changes smoothly")
print("- Distance decreases when tip moves toward ball")
print("- NO warnings\n")

t = 0.0
dt = model.opt.timestep

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        # === Manual bending (same as Stage 2) ===
        bend = 0.6 * np.sin(0.5 * t)
        for i in range(5):
            data.ctrl[i] = bend

        mujoco.mj_step(model, data)

        # === Read positions ===
        tip_pos = data.site_xpos[tip_site_id]
        target_pos = data.xpos[target_body_id]

        # === Euclidean distance ===
        distance = np.linalg.norm(tip_pos - target_pos)

        # Print occasionally (not every step)
        if int(t * 1000) % 200 == 0:
            print(f"Distance to target: {distance:.4f} m")

        viewer.sync()
        time.sleep(dt)
        t += dt
