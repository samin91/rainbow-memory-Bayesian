import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pdb

'''
You should read json file which follows below format. 

[{"klass": "truck", "file_name": "test/truck/01.jpg"}, ...]

You should change the file name as below. 
'''
#pdb.set_trace()
train = pd.read_json('/visinf/home/shamidi/Projects/My_scripts/SMALL_CUB_train.json')
test = pd.read_json('/visinf/home/shamidi/Projects/My_scripts/SMALL_CUB_test.json')

'''
Change the variables `rnd_seed` and `num_tasks` 
'''
rnd_seed = 3 # random seed 
num_tasks = 17 # the number of tasks. 
np.random.seed(rnd_seed)

klass = train.klass.unique()
num_cls_per_task = len(klass) // num_tasks
np.random.shuffle(klass)

class2label = {cls_:idx for idx, cls_ in enumerate(klass)} 
train["label"] = train.klass.apply(lambda x: class2label[x])
test["label"] = test.klass.apply(lambda x: class2label[x])

task_class = np.split(klass, num_tasks)
task_train = [train[train.klass.isin(tc)] for tc in task_class]
task_test = [test[test.klass.isin(tc)] for tc in task_class]


'''
Configure disjoint dataset which does not share the classes of each task.
'''

origin_name = "cub200" # Need to change the name of your dataset.
root = Path('../collections/cub200')
root.mkdir(exist_ok=True)
# TRAIN
for idx, train_task in enumerate(task_train):
    file_name = origin_name + '_train'
    train_prefix = {'_disjoint':'', 
              '_rand':rnd_seed, 
              '_cls':num_cls_per_task,
              '_task':idx
             }
 
    for name, value in train_prefix.items():
        file_name += name + str(value)
    file_path = (root/file_name).with_suffix('.json')
    train_task.to_json(file_path, orient='records')
    print(f"{file_path}")


# TEST
origin_name = "cub200" # Need to change the name of your dataset.
task_test = [test[test.klass.isin(tc)] for tc in task_class]

root = Path('../collections/cub200')
root.mkdir(exist_ok=True)

for idx, task in enumerate(task_test):
    file_name = origin_name + '_test'
    prefix = {'_rand':rnd_seed, 
              '_cls':num_cls_per_task,
              '_task':idx
             }
    for name, value in prefix.items():
        file_name += name + str(value)
        
    file_path = (root/file_name).with_suffix('.json')
    task.to_json(file_path, orient='records')
    print(f"{file_path}")