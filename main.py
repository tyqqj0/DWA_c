# -*- CODING: UTF-8 -*-
# @time 2023/3/26 18:37
# @Author tyqqj
# @File main.py

import numpy as np
import torch
from torch.utils import data
from torch import nn

# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision

import numpy as np
import matplotlib.pyplot as plt

import terrain
import DWA_c as DWA

args = {  # 参数
    'size': 256,
    'scale': 100.0,
    'octaves': 6,
    'persistence': 0.5,
    'lacunarity': 2.0,
    'seed': 1,
}


# def main():
#     height_map = terrain.generate_terrain(**args)
#     terrain.display_terrain(height_map)


def plot_path_on_height_map(height_map, path, goal, obstacles, robot_radius):
    path_array = np.array(path)

    plt.figure()
    plt.imshow(height_map.T, origin='lower', cmap='terrain', extent=[0, height_map.shape[0], 0, height_map.shape[1]])

    plt.plot(path_array[:, 0], path_array[:, 1], "-r")  # 画出路径
    plt.plot(goal[0], goal[1], "rx")  # 画出目标点

    for obstacle in obstacles:
        circle = plt.Circle(obstacle, robot_radius, color="b", fill=False)
        plt.gca().add_patch(circle)

    plt.xlim(0, height_map.shape[0])  # 设置x轴范围
    plt.ylim(height_map.shape[1], 0)  # 设置y轴范围
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


def main():
    # 创建地图
    height_map = terrain.generate_terrain(**args)
    terrain.display_terrain(height_map)
    print(height_map.shape)

    # 初始化
    x_init = [0, 0, 0, 0.8]  # x, y, theta: 机器人的位置和朝向, v: 速度
    goal = [100, 100]  # 目标点
    obstacles = np.array([[2, 2], [4, 4], [6, 6], [8, 8]])  # 这个是障碍物的位置
    config = {  # 配置
        "v_range": (-0.5, 6),  # 速度范围
        "w_range": (-2, 2),  # 角速度范围
        "control_steps": (0.1, 0.1),  # 速度和角速度的步长
        "prediction_steps": 10,  # 预测步数
        "dt": 0.2,  # 时间步长
        "robot_radius": 0.5,  # 机器人半径
        "goal_threshold": 2,  # 目标阈值
        "heading_weight": 0.2,  # 朝向权重
        "height_weight": 2000,  # 高度权重
        "distance_weight": 1,  # 距离权重
        "speed_weight": 0.05,  # 速度权重
        "stagnation_weight": 0.0  # 停滞权重
    }

    dwa = DWA.DWA(x_init, goal, height_map, obstacles, config)  # 初始化

    # 搜索控制
    while not dwa.reached_goal():
        u, trajectory = dwa.search_control()
        print("#########################\n")
        # 两位小数
        print('x,y: {:.2f},{:.2f}， v,w: {:.2f},{:.2f}'.format(dwa.x[0], dwa.x[1], dwa.x[3], dwa.x[2]))
        dwa.update(dwa.move(dwa.x, *u, dwa.config["dt"]))

    ###############################################################################
    # 绘制路径
    plot_path_on_height_map(height_map, dwa.path, goal, obstacles, config["robot_radius"])


if __name__ == "__main__":
    main()
