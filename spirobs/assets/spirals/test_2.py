import mujoco
import mujoco.viewer
import numpy as np
import time

# Load the model
model = mujoco.MjModel.from_xml_path("spiral_5link.xml")
data = mujoco.MjData(model)

# Sanity check
assert model.nu == 5, "Expected exactly 5 actuators"

# Launch viewer
with mujoco.viewer.launch_passive(model, data) as viewer:

    print("\nSTAGE 2 MANUAL TEST RUNNING")
    print("Expected behavior:")
    print("- The chain should bend smoothly like a snake / continuum")
    print("- All links bend in the SAME direction")
    print("- No shaking, no snapping, no warnings\n")

    t = 0.0
    dt = model.opt.timestep

    while viewer.is_running():

        # === Coordinated bending pattern ===
        # All joints get the SAME control signal
        bend = 0.6 * np.sin(0.5 * t)

        for i in range(5):
            data.ctrl[i] = bend

        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(dt)
        t += dt
