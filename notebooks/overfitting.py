import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
import torch

# Transition between weights and activations: T_1 = T_0-T_1
# Bayesian model
pdb.set_trace()
# T is the state dictionary
path_T_0 = '/visinf/home/shamidi/Projects/rainbow-memory-Bayesian/checkpoints/cub200/Run_10_node10_1_logs/task_0/checkpoint_latest_.ckpt'
T_0 = torch.load(path_T_0, map_location="cpu")
T_0 = T_0['state_dict']
print(len(T_0.keys()))
keys = keys = [k for k in T_0.keys() if 
                ("conv" in k or "weight" in k or "mult_noise_variance" in k) and ("prior" not in k and "bn" not in k and "_copy" not in k)]
print(keys)
print(len(keys))

path_T_1 = '/visinf/home/shamidi/Projects/rainbow-memory-Bayesian/checkpoints/cub200/Run_10_node10_1_logs/task_2/checkpoint_latest_.ckpt'
T_1 = torch.load(path_T_1, map_location="cpu")
D_1 = T_0-T_1
# a distribution of these values: plot also, calculate the mean and std - plot them on top of each other
# D_2 = T_1-T_2
# D_3 = T_2-T_3
# D_4 = T_3-T_4
# D_5 = T_4-T_5
# D_6 = T_5-T_6



