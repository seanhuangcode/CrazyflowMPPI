import numpy as np
from crazyflow.sim import Sim
from crazyflow.control import Control
import matplotlib.pyplot as plt
import pickle
from mppi import MPPI
import time

with open("counter.pkl", "rb") as file:
    file_number = pickle.load(file)

with open(f'commands/command_data{file_number - 1}.pkl', "rb") as file:
    command_list = pickle.load(file)

sim = Sim(n_worlds=1, n_drones=1, freq=500, control=Control.attitude)
sim.reset()
rgb = sim.render(mode="rgb_array")             # numpy array (H, W, 3)
depth = sim.render(mode="depth_array")         # numpy array (H, W)
rgb, depth = sim.render(mode="rgbd_tuple", camera="fpv_cam:0", width=320, height=240)

history = []

for i in range(len(command_list)):
    sim.attitude_control(command_list[i])
    sim.step(5)
    time.sleep(0.02)
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