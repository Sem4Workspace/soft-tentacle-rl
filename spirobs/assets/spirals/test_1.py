import mujoco
import mujoco.viewer
import time

# Load model
model = mujoco.MjModel.from_xml_path("stage1_single_hinge.xml")
data = mujoco.MjData(model)

# Launch viewer
with mujoco.viewer.launch_passive(model, data) as viewer:

    print("Manual test running...")
    print("Expected behavior:")
    print("- The blue link should rotate back and forth smoothly")
    print("- NO warnings, NO NaNs, NO instability")

    t = 0.0
    while viewer.is_running():
        # Simple sinusoidal control
        data.ctrl[0] = 0.8 * (1 if int(t) % 2 == 0 else -1)

        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(0.002)
        t += 0.002
