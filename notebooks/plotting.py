import numpy as np
import matplotlib.pyplot as plt
import pdb

# enter dataset name
dataset = 'cifar10'
Path_1 = '/visinf/home/shamidi/Projects/rainbow-memory-Bayesian/results/cub200/2023-04-04_15-47-22_rm_uncertainty_offline_msz500_rnd3_cutmix_autoaug.npy' #with pretraining
Path_2 = '/visinf/home/shamidi/Projects/rainbow-memory-Bayesian/results/cub200/2023-03-26_09-42-43_rm_uncertainty_offline_msz500_rnd3_cutmix_autoaug.npy' # Bayesian without pretraining 
Path_3 = '/visinf/home/shamidi/Projects/rainbow-memory-Bayesian/results/cub200/2023-03-26_09-47-18_rm_uncertainty_offline_msz500_rnd3_cutmix_autoaug.npy' # RM without pretraining

data_1 = np.load(Path_1)
data_2 = np.load(Path_2)
data_3 = np.load(Path_3)

fig, axs = plt.subplots(
    len(data_1), 1, figsize=(40, 20*len(data_1))
                        ) # (nrows, ncols, )

for i, (sublist_1, sublist_2, sublist_3) in enumerate(zip(data_1, data_2, data_3)):
    start = min(np.arange(len(sublist_1)))
    end = max(np.arange(len(sublist_1)))
    axs[i].plot(range(len(sublist_1)), sublist_1, 'r--', range(len(sublist_1)), sublist_2, 'b--',  range(len(sublist_3)), sublist_3, 'g--')
    axs[i].set_title(f"Task {i+1}")
    axs[i].set_xlabel("Classes")
    axs[i].set_ylabel("Accuracy per class")


    axs[i].set_xticks(np.arange(start, end+1, 1))  # set tick locations
    axs[i].set_xticklabels(range(1, len(sublist_1)+1), rotation=(90), fontsize=7)  # set tick labels
    
    axs[i].set_yticks(np.arange(0, 1, 0.1))  # set tick locations
    #axs[i].set_yticklabels(np.arange(0, 1, 0.1), fontsize=10)  # set tick labels range(start, stop, step)

    axs[i].grid(True)  # add grid
    
# Show the plot

plt.tight_layout()
plt.savefig('Rm_CUB200_pretrain.png')