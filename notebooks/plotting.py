import numpy as np
import matplotlib.pyplot as plt
import pdb

# enter dataset name
dataset = 'cifar10'
Path = '/visinf/home/shamidi/Projects/rainbow-memory-Bayesian/results/cifar10/2023-03-23_11-49-54_rm_random_offline_msz500_rnd1_cutmix_autoaug.npy'

data = np.load(Path)
fig, axs = plt.subplots(len(data), 1, figsize=(6, 6*len(data)))
for i, sublist in enumerate(data):
    axs[i].plot(range(len(sublist)), sublist)
    axs[i].set_title(f"Task {i+1}")
    axs[i].set_xlabel("Classes")
    axs[i].set_ylabel("Accuracy per class")
    axs[i].set_xticks(range(len(sublist) + 2)[1:-1])  # set tick locations
    axs[i].set_xticklabels(range(1, len(sublist)+1))  # set tick labels
    axs[i].grid(True)  # add grid
# Show the plot
plt.tight_layout()
plt.savefig('result.png')