import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch
from torch.utils.tensorboard import SummaryWriter

f = 'tensorboard/overfitting/'
writer = SummaryWriter(f)

path_T_0 = '/visinf/home/shamidi/Projects/rainbow-memory-Bayesian/checkpoints/cub200/Run_10_node10_1_logs/task_0/checkpoint_latest_.ckpt'
path_T_1 = '/visinf/home/shamidi/Projects/rainbow-memory-Bayesian/checkpoints/cub200/Run_10_node10_1_logs/task_1/checkpoint_latest_.ckpt'
path_T_2 = '/visinf/home/shamidi/Projects/rainbow-memory-Bayesian/checkpoints/cub200/Run_10_node10_1_logs/task_2/checkpoint_latest_.ckpt'
path_T_3 = '/visinf/home/shamidi/Projects/rainbow-memory-Bayesian/checkpoints/cub200/Run_10_node10_1_logs/task_3/checkpoint_latest_.ckpt'
path_T_4 = '/visinf/home/shamidi/Projects/rainbow-memory-Bayesian/checkpoints/cub200/Run_10_node10_1_logs/task_4/checkpoint_latest_.ckpt'
path_T_5 = '/visinf/home/shamidi/Projects/rainbow-memory-Bayesian/checkpoints/cub200/Run_10_node10_1_logs/task_5/checkpoint_latest_.ckpt'

T_0 = torch.load(path_T_0, map_location="cpu")
T_0 = T_0['state_dict']
T_1 = torch.load(path_T_1, map_location="cpu")
T_1 = T_1['state_dict']
T_2 = torch.load(path_T_2, map_location="cpu")
T_2 = T_2['state_dict']
T_3 = torch.load(path_T_3, map_location="cpu")
T_3 = T_3['state_dict']
T_4 = torch.load(path_T_4, map_location="cpu")
T_4 = T_4['state_dict']
T_5 = torch.load(path_T_5, map_location="cpu")
T_5= T_5['state_dict']




keys = [k for k in T_0.keys() if 
                ("conv" in k or "weight" in k or "mult_noise_variance" in k) and ("prior" not in k and "bn" not in k and "_copy" not in k and 'fc.' not in k)]


D_1=[]
D_2=[]
D_3=[]
D_4=[]
D_5=[]
for key in keys:
    
    D_1.extend((T_0[key]-T_1[key]).numpy().flatten())
    D_2.extend((T_1[key]-T_2[key]).numpy().flatten())
    D_3.extend((T_2[key]-T_3[key]).numpy().flatten())
    D_4.extend((T_3[key]-T_4[key]).numpy().flatten())
    D_5.extend((T_4[key]-T_5[key]).numpy().flatten())

Diffs = [D_1, D_2, D_3, D_4, D_5]
'''
for i in range(5):
    i=i+1
    x = np.random.random(1000)
    writer.add_histogram('distribution centers',f'D_{i}' , i)
writer.close()


'''
# Create a figure with 5 subplots arranged in a 1x5 grid
fig, axs = plt.subplots(5, 1, figsize=(4, 30))

# Plot a histogram on each subplot
for i, D in enumerate(Diffs):
    n, bins, patches = axs[i].hist(D, bins=2, range=(min(D), max(D))) 
    axs[i].set_xticks(bins)
    axs[i].set_title(f'Histogram {i+1}')
    #axs[i].set_xlim([-1, 2])
    axs[i].set_xlabel('Value')
    axs[i].set_ylabel('Frequency')

# Add a title to the figure
fig.suptitle('Five Histograms')

plt.tight_layout()
plt.savefig('histograms.png')
