import numpy as np
from crazyflow.sim import Sim
from crazyflow.control import Control
import matplotlib.pyplot as plt

class MPPI():
    def __init__(self, simulator, samples, horizon, goal):

        cmd = np.array([[[0.0, 0.0, 0.0, 0.5]]])

        self.simulator = simulator
        self.samples = samples

        self.horizon = horizon
        self.goal = goal

        self.cmd = cmd
        self.nominal_cmd = np.tile(self.cmd, (1, self.horizon, 1))
        print (self.nominal_cmd.shape)


        self.costs = np.zeros(self.samples)

        self.sim_world = Sim(n_worlds=samples, n_drones=1, freq=200, control=Control.attitude)
    
    def get_state(self):
        self.pos = self.simulator.data.states.pos        # (world, drone, 3)
        self.quat = self.simulator.data.states.quat      # (world, drone, 4)
        self.vel = self.simulator.data.states.vel        # (world, drone, 3)
        return self.pos

    def generate_gaussian_noise(self, dev):
        self.noise = np.random.normal(0, dev, (self.samples, self.horizon, 4))
        self.cmd_list = np.tile(self.nominal_cmd, (self.samples, 1, 1)) #(samples, horizon, 4)

 
        print (self.noise.shape)

    
    def calculate_cost(self):
        pos = self.sim_world.data.states.pos[:,0,:]
        vel = self.sim_world.data.states.vel[:,0,:]

        error = pos - self.goal[:3]

        pos_cost = (
            5.0 * error[:,0]**2 +   # x
            5.0 * error[:,1]**2 +   # y
            10.0 * error[:,2]**2   # z
        )

        vel_cost = (
            1.0 * vel[:,0]**2 +
            1.0 * vel[:,1]**2 +
            20.0 * vel[:,2]**2
        )


        control_cost = np.sum(self.cmd**2, axis=-1)

        self.costs += pos_cost + vel_cost + 0.01 * control_cost


    
    def rollout(self, real_states):

        self.sim_world.data = self.sim_world.data.replace(
            states=self.sim_world.data.states.replace(
                pos=np.repeat(real_states.pos, self.samples, axis=0),
                quat=np.repeat(real_states.quat, self.samples, axis=0),
                vel=np.repeat(real_states.vel, self.samples, axis=0),
            )
        )
        self.costs = np.zeros(self.samples)
        perturbed_cmd_list = self.cmd_list + self.noise

        for i in range(self.horizon):
            step_cmd = perturbed_cmd_list[:, i, :].reshape(self.samples, 1, 4)
            self.sim_world.attitude_control(step_cmd)

            self.sim_world.step(1)
            self.calculate_cost()
        
        self.sim_world.reset()
            

    def update_command(self, lamb):

        low_cost = np.min(self.costs)

        weights = np.exp(-1/lamb * (self.costs - low_cost))

        delta_u = (np.sum(np.reshape(weights, (self.samples, 1, 1)) * self.noise.reshape((self.samples, self.horizon, 4)), axis=0)) \
                / np.sum(weights)

        self.nominal_cmd[0] += delta_u

        cmd = self.nominal_cmd[:, 0:1, :].copy()

        self.nominal_cmd[:, :-1, :] = self.nominal_cmd[:, 1:, :]
        self.nominal_cmd[:, -1, :] = 0

        return cmd
    
