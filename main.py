"""
rainbow-memory
Copyright 2021-present NAVER Corp.
GPLv3
"""
import datetime
import logging.config
import os
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.use_deterministic_algorithms(True, warn_only=True)

from randaugment import RandAugment
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from configuration import config
from utils.augment import Cutout, select_autoaugment
from utils.data_loader import get_test_datalist, get_statistics
from utils.data_loader import get_train_datalist
from utils.method_manager import select_method
from utils.bayes_utils import configure_prior_conversion_function
from losses.probabilistic_loss import configure_model_and_loss

# add the bayesian losses
from losses import ClassificationLoss, ClassificationLossVI
import pdb

def main():
    
    args = config.base_parser()
    # time stamp 
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Save file name
    tr_names = ""
    for trans in args.transforms:
        tr_names += "_" + trans
    save_path = f"{args.dataset}/{timestamp}_{args.mode}_{args.mem_manage}_{args.stream_env}_msz{args.memory_size}_rnd{args.rnd_seed}{tr_names}"

    # Logger
    logging.config.fileConfig("./configuration/logging.conf")
    logger = logging.getLogger()
    os.makedirs(f"/visinf/projects/shamidi/RM_modified/logs/{args.dataset}", exist_ok=True)
    fileHandler = logging.FileHandler("/visinf/projects/shamidi/RM_modified/logs/{}.log".format(save_path), mode="w")
    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    # Tensorboard
    #writer = SummaryWriter(f"test/run_{1}")

    # Device
    # add an argument the device args.device="cuda:0" or "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Set the device ({device})")
  

    # Fix the random seeds
    # https://hoya012.github.io/blog/reproducible_pytorch/
    torch.manual_seed(args.rnd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Add more seeding? maybe it will be necessary for the bayesian model
    torch.cuda.manual_seed_all(args.rnd_seed)
    torch.cuda.manual_seed(args.rnd_seed)
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)

    # Transform Definition
    # here I need to add information about CUB_200 as well: For normalization of CUB_200 I use the information of ImageNet
    mean, std, n_classes, inp_size, _ = get_statistics(dataset=args.dataset)
    train_transform = []
    if "cutout" in args.transforms:
        train_transform.append(Cutout(size=16))
    if "randaug" in args.transforms:
        train_transform.append(RandAugment())
    if "autoaug" in args.transforms:
        train_transform.append(select_autoaugment(args.dataset))

    train_transform = transforms.Compose(
        [
            transforms.Resize((inp_size, inp_size)),
            transforms.RandomCrop(inp_size, padding=4),
            transforms.RandomHorizontalFlip(),
            *train_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    logger.info(f"Using train-transforms {train_transform}")

    test_transform = transforms.Compose(
        [
            transforms.Resize((inp_size, inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    # Loss Function
    if args.bayesian_model is True:
        logger.info(f"[1] Training a Bayesian model)")
        criterion = configure_model_and_loss()
    else:
        criterion = nn.CrossEntropyLoss(reduction="mean")
    

    # METHOD SELECTION
    method = select_method(
        args, criterion, device, train_transform, test_transform, n_classes
    )
    logger.info(f"[1] Select a CIL method ({args.mode})")


    # Load the checkpoint or pretrained model
    if args.pretrain is True and args.bayesian_model is True:
        method.checkpoint_saver_loader()


    logger.info(f"[2] Incrementally training {args.n_tasks} tasks")
    task_records = defaultdict(list)

    
    # Main Loop - cur_iter is not the best name for the current task
    for cur_iter in range(args.n_tasks):
        # ----------------------------------------
        # TENSRBOARD
        # ---------------------------------------
        f = 'tensorboard/test/'+'task_' + str(cur_iter)
        writer = SummaryWriter(f)
        

        if args.mode == "joint" and cur_iter > 0:
            return

        print("\n" + "#" * 50)
        print(f"# Task {cur_iter} iteration")
        print("#" * 50 + "\n")
        logger.info("[2-1] Prepare a datalist for the current task")

        task_acc = 0.0
        eval_dict = dict()

        # get datalist
        # what about validation set?
        cur_train_datalist = get_train_datalist(args, cur_iter)
        cur_test_datalist = get_test_datalist(args, args.exp_name, cur_iter)

        # Reduce datalist in Debug mode
        if args.debug:
            random.shuffle(cur_train_datalist)
            random.shuffle(cur_test_datalist)
            cur_train_datalist = cur_train_datalist[:2560]
            cur_test_datalist = cur_test_datalist[:2560]
        
        logger.info("[2-2] Set environment for the current task")
        method.set_current_dataset(cur_train_datalist, cur_test_datalist)
        # Increment known class for current task iteration.
        # reinitalize optimizer and scheduler?
        # Add the trainable parameters for the optimizer
        method.before_task(cur_train_datalist, cur_iter, args.init_model, args.init_opt, args.bayesian_model)

        # The way to handle streamed samles
        logger.info(f"[2-3] Start to train under {args.stream_env}")

        if args.stream_env == "offline" or args.mode == "joint" or args.mode == "gdumb":
            # Offline Train
            # returns best eval accuracy and not the last one
            # task_acc is the evaluation accuracy and not the training accuracy
            task_acc, eval_dict = method.train(
                cur_iter=cur_iter,
                n_epoch=args.n_epoch,
                batch_size=args.batchsize,
                n_worker=args.n_worker,
                writer=writer
            )
            if args.mode == "joint":
                logger.info(f"joint accuracy: {task_acc}")

        elif args.stream_env == "online":
            # Online Train
            logger.info("Train over streamed data once")
            method.train(
                cur_iter=cur_iter,
                n_epoch=1,
                batch_size=args.batchsize,
                n_worker=args.n_worker,
                writer=writer
            )

            method.update_memory(cur_iter)

            # No stremed training data, train with only memory_list
            method.set_current_dataset([], cur_test_datalist)

            logger.info("Train over memory")
            task_acc, eval_dict = method.train(
                cur_iter=cur_iter,
                n_epoch=args.n_epoch,
                batch_size=args.batchsize,
                n_worker=args.n_worker,
                writer=writer
            )

            method.after_task(cur_iter)

        logger.info("[2-4] Update the information for the current task")
        method.after_task(cur_iter)
        
        if args.bayesian_model is True and args.informed_prior is True:
            logger.info("[2-4] Update the prior for the current task: posterior -> prior")
            #prior_conversion = args.prior_conv_func
            logger.info(f"Prior conversion function: {args.prior_conv_function}")
            prior_conv_func = configure_prior_conversion_function(args.prior_conv_function)
            method.update_prior(prior_conv_func)

        task_records["task_acc"].append(task_acc)
        # task_records['cls_acc'][k][j] = break down j-class accuracy from 'task_acc'
        task_records["cls_acc"].append(eval_dict["cls_acc"])

        # Save checkpoint per task
        directory = f"checkpoints/{args.dataset}/Run_1/task_{cur_iter}/"
        os.makedirs(directory, exist_ok=True)
        method.save_checkpoint(directory, task_records, store_as_best=False, store_prefixes="total_loss")
        

        # Notify to NSML
        logger.info("[2-5] Report task result")
        writer.add_scalar("Metrics/TaskAcc", task_acc, cur_iter)
    # I remember there was an error running this line of code

   
    if os.path.exists(f"results/{args.dataset}") is False:
        os.makedirs(f"results/{args.dataset}", exist_ok=True)
    np.save(f"results/{save_path}.npy", task_records["cls_acc"])
    
  
    # Loop through the sublists and plot each one
    for i, sublist in enumerate(task_records["cls_acc"]):
        plt.subplot(1, len(task_records["cls_acc"]), i+1)
        plt.plot(range(len(sublist)), sublist)
        plt.title(f"List {i+1}")
        plt.xlabel("Classes")
        plt.ylabel("Accuracy per class")

    # Show the plot
    plt.tight_layout()
    plt.show()
    
   
    # Accuracy (A)
    A_avg = np.mean(task_records["task_acc"])
    A_last = task_records["task_acc"][args.n_tasks - 1]
    # Forgetting (F) - Read on the formula of Forgetting and understand it - check this code
    acc_arr = np.array(task_records["cls_acc"])
    # cls_acc = (k, j), acc for j at k
    cls_acc = acc_arr.reshape(-1, args.n_cls_a_task).mean(1).reshape(args.n_tasks, -1)
    for k in range(args.n_tasks):
        forget_k = []
        for j in range(args.n_tasks):
            if j < k:
                forget_k.append(cls_acc[:k, j].max() - cls_acc[k, j])
            else:
                forget_k.append(None)
        task_records["forget"].append(forget_k)
    F_last = np.mean(task_records["forget"][-1][:-1])

    # Intrasigence (I) - how much difference from the upper bound
    I_last = args.joint_acc - A_last

    logger.info(f"======== Summary =======")
    logger.info(f"A_last {A_last} | A_avg {A_avg} | F_last {F_last} | I_last {I_last}")


if __name__ == "__main__":
    main()
