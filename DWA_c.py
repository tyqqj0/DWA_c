# -*- CODING: UTF-8 -*-
# @time 2023/3/26 19:41
# @Author tyqqj
# @File DWA_c.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

__all__ = ["DWA"]


class DWA:
    def __init__(self, x_init, goal, height_map, obstacles, config):
        self.x = np.array(x_init)
        self.height = height_map[x_init[0], x_init[1]]
        self.goal = np.array(goal)
        self.height_map = height_map
        self.obstacles = obstacles
        self.config = config
        self.path = [x_init]
        self.stagnation_counter = 0
        self.previous_position = np.array(x_init[:2])

    def update(self, x):
        self.path.append(x)
        self.x = np.array(x)
        x, y = int(self.x[0]), int(self.x[1])
        if 0 <= x < self.height_map.shape[0] and 0 <= y < self.height_map.shape[1]:
            self.height = self.height_map[x, y]
        distance_moved = np.linalg.norm(self.x[:2] - self.previous_position)
        if distance_moved < 0.1:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        self.previous_position = self.x[:2]

    def reached_goal(self):
        distance_to_goal = np.linalg.norm(self.x[:2] - self.goal)
        return distance_to_goal < self.config["goal_threshold"]

    def search_control(self):
        best_u = None
        best_trajectory = None
        max_score = -float("inf")
        # pers = []
        for u in self.generate_controls():
            trajectory = self.simulate_trajectory(u)  # 模拟轨迹
            score = self.evaluate_trajectory(trajectory)  # 评估轨迹

            if score > max_score:
                max_score = score
                best_u = u
                # pers = per
                best_trajectory = trajectory
        # print("per:{}".format(pers))
        print("best_u:{}, {} best_score:{}".format(best_u[0], best_u[1], max_score))
        # 打印目标方向
        print("goal_direction:{}".format(np.arctan2(self.goal[1] - self.x[1], self.goal[0] - self.x[0])))
        return best_u, best_trajectory

    def generate_controls(self):
        '''
        Generate all possible controls
        :return:
        '''
        config = self.config
        v_min, v_max = config["v_range"]
        w_min, w_max = config["w_range"]
        v_step, w_step = config["control_steps"]

        v_samples = np.arange(v_min, v_max + v_step, v_step)
        w_samples = np.arange(w_min, w_max + w_step, w_step)

        # 考虑当前速度
        current_v = self.x[3]
        v_samples = np.clip(v_samples + current_v, v_min, v_max)

        controls = [(v, w) for v in v_samples for w in w_samples]

        return controls

    def simulate_trajectory(self, u):
        '''
        Simulate the trajectory of the robot given a control input
        :param u: control input
        :return:
        '''
        x = self.x.copy()
        v, w = u
        dt = self.config["dt"]
        trajectory = [x]

        for _ in range(self.config["prediction_steps"]):
            x = self.move(x, v, w, dt)
            trajectory.append(x)

        return np.array(trajectory)

    def move(self, x, v, w, dt):
        x_new = x.copy()
        x_new[0] += v * np.cos(x[2]) * dt
        x_new[1] += v * np.sin(x[2]) * dt
        x_new[2] += w * dt
        x_new[3] = v
        return x_new

    def evaluate_trajectory(self, trajectory):
        config = self.config
        # Calculate height delta cost
        height_d_cost = 0
        last_height = self.height
        for point in trajectory:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < self.height_map.shape[0] and 0 <= y < self.height_map.shape[1]:
                # Calculate the height delta cost
                height_d_cost += ((self.height_map[x, y] - last_height) ** 2) * config["height_weight"]
                last_height = self.height_map[x, y]
            else:
                # Add a large penalty for out-of-bound positions
                # print("Out of bounds: ({}, {})".format(x, y))
                height_d_cost += 1e6
        # print(height_d_cost)
        # Check for collisions
        for point in trajectory:
            for obstacle in self.obstacles:
                if np.linalg.norm(point[:2] - obstacle) < config["robot_radius"]:
                    return -float("inf")

        # Calculate goal heading cost
        dx = self.goal[0] - trajectory[-1, 0]
        dy = self.goal[1] - trajectory[-1, 1]
        heading = np.arctan2(dy, dx)
        goal_heading_cost = config["heading_weight"] * (heading - trajectory[-1, 2]) ** 2

        # Calculate goal distance cost
        goal_distance_cost = config["distance_weight"] * np.linalg.norm(trajectory[-1, :2] - self.goal)

        # Calculate speed cost
        speed_cost = config["speed_weight"] * (config["v_range"][1] - trajectory[-1, 3])

        # Calculate stagnation cost
        stagnation_cost = self.config["stagnation_weight"] * self.stagnation_counter

        goal_distance_cost = goal_distance_cost + 1e-7
        goal_heading_cost = goal_heading_cost + 1e-7
        speed_cost = speed_cost + 1e-7
        height_d_cost = height_d_cost + 1e-7
        stagnation_cost = stagnation_cost + 1e-7

        cost = goal_heading_cost + goal_distance_cost + speed_cost + height_d_cost + stagnation_cost
        # print('height_d_cost / cost', height_d_cost / (cost + 1e-6))
        # 计算每个cost的占比
        percent = [goal_heading_cost / (cost + 1e-6), goal_distance_cost / (cost + 1e-6), speed_cost / (cost + 1e-6),
                   height_d_cost / (cost + 1e-6), stagnation_cost / (cost + 1e-6)]
        # print('percent', percent)
        return -cost


def display_dwa_search(dwa, trajectory):
    plt.cla()
    path_array = np.array(dwa.path)
    plt.plot(path_array[:, 0], path_array[:, 1], "-r")

    plt.plot(trajectory[:, 0], trajectory[:, 1], "-g")
    plt.plot(dwa.goal[0], dwa.goal[1], "rx")

    for obstacle in dwa.obstacles:
        circle = Circle(obstacle, dwa.config["robot_radius"], color="b", fill=False)
        plt.gca().add_patch(circle)

    plt.xlim(-5, 15)
    plt.ylim(-5, 15)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.draw()
    plt.pause(0.1)
