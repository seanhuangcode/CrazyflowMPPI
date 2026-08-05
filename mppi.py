import numpy as np
from crazyflow.sim import Sim
from crazyflow.control import Control
import matplotlib.pyplot as plt

class MPPI():
    def __init__(self, simulator, samples, horizon, goal, obstacles):

        cmd = np.array([[[0.0, 0.0, 0.0, 0.427]]])

        self.simulator = simulator
        self.samples = samples

        self.horizon = horizon
        self.goal = goal

        self.obstacle = obstacles

        self.cmd = cmd
        self.nominal_cmd = np.tile(self.cmd, (1, self.horizon, 1))

        self.costs = np.zeros(self.samples)

        self.sim_world = Sim(n_worlds=samples, n_drones=1, freq=500, control=Control.attitude)
    

    def generate_gaussian_noise(self, dev):
        self.noise = np.random.normal(0, dev, (self.samples, self.horizon, 4))
        self.noise[:, :, 3] = np.random.normal(0, dev[3], size=(self.samples, self.horizon))
        self.cmd_list = np.tile(self.nominal_cmd, (self.samples, 1, 1)) #(samples, horizon, 4)

    def calculate_obstacle_cost(self):

        pos = self.sim_world.data.states.pos[:,0,:]

        radius = 1

        distance = np.linalg.norm(pos - self.obstacle, axis=1) - radius
        safe_distance = 1
        cost = np.maximum(0, safe_distance-distance)**2

        return 1e6 * cost

        
    def calculate_cost(self):
        pos = self.sim_world.data.states.pos[:,0,:]
        vel = self.sim_world.data.states.vel[:,0,:]

        error = pos - self.goal[:3]

        pos_cost = (
            10.0 * error[:,0]**2 +
            10.0 * error[:,1]**2 +
            20.0 * error[:,2]**2
        )

        z = pos[:, 2]

        below_height = np.maximum(0, 0.5 - z)

        self.costs += 100 * below_height**2

        vel_cost = (
            4.0 * vel[:,0]**2 +
            4.0 * vel[:,1]**2 +
            10 * vel[:,2]**2
        )

        obstacle_cost = self.calculate_obstacle_cost()

        self.costs += pos_cost + 0.75 * vel_cost + obstacle_cost

    def calculate_terminal_cost(self):
        pos = self.sim_world.data.states.pos[:,0,:]
        vel = self.sim_world.data.states.vel[:,0,:]
        quat = self.sim_world.data.states.quat[:,0,:]

        dist_to_goal = np.linalg.norm(pos - self.goal[:3], axis=1)
        speed_sq = np.sum(vel**2, axis=1)

        FACTOR = 1

        terminal_pos_weight = 150.0
        error = pos - self.goal[:3]
        pos_cost = (
                    1* error[:,0]**2 +
                    1 * error[:,1]**2 +
                    1 * error[:,2]**2
                )

        self.costs += FACTOR * terminal_pos_weight * pos_cost

        near_goal_radius = 0.3
        closeness = np.clip(1.0 - dist_to_goal / near_goal_radius, 0.0, 1.0)

        terminal_vel_weight = 20.0
        self.costs += FACTOR * terminal_vel_weight * closeness * speed_sq

        qx, qy = quat[:,0], quat[:,1]

        terminal_tilt_weight = 50
        self.costs += FACTOR * terminal_tilt_weight * closeness * (qx**2 + qy**2)
    
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
            self.sim_world.step(7)
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
    
def sphere_sdf(points, center, radius):
    """
    points: (N,3)
    returns: (N,)
    """

    return np.linalg.norm(points - center, axis=1) - radius