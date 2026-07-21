import numpy as np
from crazyflow.sim import Sim
from crazyflow.control import Control
import matplotlib.pyplot as plt
from mppi import MPPI

sim = Sim(n_worlds=1, n_drones=1, freq=500, control=Control.attitude)
sim.reset()
rgb = sim.render(mode="rgb_array")             # numpy array (H, W, 3)
depth = sim.render(mode="depth_array")         # numpy array (H, W)
rgb, depth = sim.render(mode="rgbd_tuple", camera="fpv_cam:0", width=320, height=240)

goal = np.array([0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

ctrl = MPPI(sim, 200, 100, goal)

history = []

for i in range(1000):
    ctrl.generate_gaussian_noise(np.array([0.03, 0.08, 0.02, 0.15]))
    ctrl.rollout(sim.data.states)
    cmd = ctrl.update_command(100)
    sim.attitude_control(cmd)
    sim.step(5)
    sim.render()
    history.append(sim.data.states.pos[0, 0].copy())

history = np.array(history)
plt.plot(history[:, 0], label="x")
plt.plot(history[:, 1], label="y")
plt.plot(history[:, 2], label="z")
plt.axhline(goal[0], color="C0", linestyle="--")
plt.axhline(goal[1], color="C1", linestyle="--")
plt.axhline(goal[2], color="C2", linestyle="--")
plt.legend()
plt.show()