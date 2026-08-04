import numpy as np
from crazyflow.sim import Sim
from crazyflow.control import Control
import matplotlib.pyplot as plt
import pickle
from mppi import MPPI

with open("counter.pkl", "rb") as file:
    file_counter = pickle.load(file)

sim = Sim(n_worlds=1, n_drones=1, freq=500, control=Control.attitude)
sim.reset()
rgb = sim.render(mode="rgb_array")             # numpy array (H, W, 3)
depth = sim.render(mode="depth_array")         # numpy array (H, W)
rgb, depth = sim.render(mode="rgbd_tuple", camera="fpv_cam:0", width=320, height=240)

goal = np.array([7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

ctrl = MPPI(sim, 2000, 60, np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

history = []
command_history = []

print(sim.data.controls.attitude.freq)

ctrl.goal = goal

for i in range(400):
    ctrl.generate_gaussian_noise(np.array([0.05,0.05,0.05,0.1]))
    ctrl.rollout(sim.data.states)
    cmd = ctrl.update_command(25)
    sim.attitude_control(cmd)
    command_history.append(cmd)
    sim.step(7)
    history.append(sim.data.states.pos[0, 0].copy())

command_history = np.array(command_history)

with open(f'commands/command_data{file_counter}.pkl', "wb") as file:
    pickle.dump(command_history, file)

with open("counter.pkl", "wb") as file:
    pickle.dump(file_counter + 1, file)

history = np.array(history)
plt.plot(history[:, 0], label="x")
plt.plot(history[:, 1], label="y")
plt.plot(history[:, 2], label="z")
plt.axhline(goal[0], color="C0", linestyle="--")
plt.axhline(goal[1], color="C1", linestyle="--")
plt.axhline(goal[2], color="C2", linestyle="--")
plt.legend()
plt.show()