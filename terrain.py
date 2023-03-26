# -*- CODING: UTF-8 -*-
# @time 2023/3/26 19:42
# @Author tyqqj
# @File terrain.py

import numpy as np
import torch
from torch.utils import data
from torch import nn

# import os
import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision

from opensimplex import OpenSimplex

def generate_terrain(size, scale, octaves, persistence, lacunarity, seed):
    height_map = np.zeros((size, size))
    noise_generator = OpenSimplex(seed=seed)

    for i in range(size):
        for j in range(size):
            frequency = 1
            amplitude = 1
            total = 0

            for _ in range(octaves):
                total += noise_generator.noise2(i * frequency / scale, j * frequency / scale) * amplitude
                frequency *= lacunarity
                amplitude *= persistence

            height_map[i][j] = total

    return (height_map + 1) / 2


def display_terrain(height_map):
    plt.imshow(height_map, cmap='terrain')
    plt.colorbar()
    plt.show()
