import numpy as np
from crazyflow.sim import Sim
from crazyflow.control import Control
import matplotlib.pyplot as plt

class MPPI():
    def __init__(self, simulator, samples, horizon, goal):

        cmd = np.array([[[0.0, 0.0, 0.0, 1]]])

        self.simulator = simulator
        self.samples = samples

        self.horizon = horizon
        self.goal = goal

        self.cmd = cmd
        self.nominal_cmd = np.tile(self.cmd, (1, self.horizon, 1))


        self.costs = np.zeros(self.samples)

        self.sim_world = Sim(n_worlds=samples, n_drones=1, freq=200, control=Control.attitude)
    

    def generate_gaussian_noise(self, dev):
        self.noise = np.random.normal(0, dev, (self.samples, self.horizon, 4))
        # Keep thrust perturbations zero-mean instead of adding a large upward bias.
        self.noise[:, :, 3] = np.random.normal(0, dev[3], size=(self.samples, self.horizon))
        self.cmd_list = np.tile(self.nominal_cmd, (self.samples, 1, 1)) #(samples, horizon, 4)

    
        
    def calculate_cost(self):
        pos = self.sim_world.data.states.pos[:,0,:]
        vel = self.sim_world.data.states.vel[:,0,:]

        error = pos - self.goal[:3]

        pos_cost = (
            5.0 * abs(error[:,0]) +
            5.0 * abs(error[:,1]) +
            10.0 * abs(error[:,2])
        )

        vel_cost = (
            1.0 * vel[:,0]**2 +   # x velocity
            1.0 * vel[:,1]**2 +   # y velocity
            0.5 * vel[:,2]**2     # z velocity
        )

        self.costs += pos_cost + vel_cost


    def calculate_terminal_cost(self):
        pos = self.sim_world.data.states.pos[:,0,:]
        vel = self.sim_world.data.states.vel[:,0,:]
        quat = self.sim_world.data.states.quat[:,0,:]

        dist_to_goal = np.linalg.norm(pos - self.goal[:3], axis=1)
        speed_sq = np.sum(vel**2, axis=1)

        near_goal_radius = 0.3  # meters, tune to your goal tolerance
        closeness = np.clip(1.0 - dist_to_goal / near_goal_radius, 0.0, 1.0)

        terminal_vel_weight = 20.0
        self.costs += terminal_vel_weight * closeness * speed_sq

        # Orientation only penalized at the end, gated by closeness — free to
        # tilt during travel, expected to be level once it's actually arrived.
        qx, qy = quat[:,0], quat[:,1]
        terminal_tilt_weight = 15.0
        self.costs += terminal_tilt_weight * closeness * (qx**2 + qy**2)
    
    def rollout(self, real_states):
        self.sim_world.data = self.sim_world.data.replace(
            states=self.sim_world.data.states.replace(
                pos=np.repeat(real_states.pos, self.samples, axis=0),
                quat=np.repeat(real_states.quat, self.samples, axis=0),
                vel=np.repeat(real_states.vel, self.samples, axis=0),
                ang_vel=np.repeat(real_states.ang_vel, self.samples, axis=0)
            )
        )

        self.costs = np.zeros(self.samples)
        perturbed_cmd_list = self.cmd_list + self.noise

        for i in range(self.horizon):
            step_cmd = perturbed_cmd_list[:, i, :].reshape(self.samples, 1, 4)
            self.sim_world.attitude_control(step_cmd)
            self.sim_world.step(1)
            self.calculate_cost()

        self.calculate_terminal_cost() 

        print(np.min(self.costs) / self.samples, real_states.pos[2])
        

    def update_command(self, lamb):

        low_cost = np.min(self.costs)

        weights = np.exp(-1/lamb * (self.costs - low_cost))

        delta_u = (np.sum(np.reshape(weights, (self.samples, 1, 1)) * self.noise.reshape((self.samples, self.horizon, 4)), axis=0)) \
                / np.sum(weights)

        self.nominal_cmd[0] += delta_u

        cmd = self.nominal_cmd[:, 0:1, :].copy()

        self.nominal_cmd[:, :-1, :] = self.nominal_cmd[:, 1:, :]
        self.nominal_cmd[:, -1, :] = self.cmd


        return cmd
    
