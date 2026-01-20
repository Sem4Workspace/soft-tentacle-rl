import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("assets/spirals/spiral_5link.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data):
    while True:
        data.ctrl[:] = [0.8, 0.8, 0.8, 0.8, 0.8]
        mujoco.mj_step(model, data)
