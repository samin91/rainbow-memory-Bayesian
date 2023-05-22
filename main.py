"""
rainbow-memory
Copyright 2021-present NAVER Corp.
GPLv3
"""
from functools import partial

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

# Ray
import ray
from ray import tune, air
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.bayesopt import BayesOptSearch

from configuration import config
from utils.augment import Cutout, select_autoaugment

# ray.tune 
from utils.data_loader import get_test_datalist, get_statistics
from utils.data_loader import get_train_datalist, get_valid_datalist

from utils.method_manager import select_method
from utils.bayes_utils import configure_prior_conversion_function
from losses.probabilistic_loss import configure_model_and_loss

# add the bayesian losses
from losses import ClassificationLoss, ClassificationLossVI
import pdb
import pickle

def main():
    
    Exp_name = 'test_ray'
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
    # valid transformation is the same as the test transformation

    
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


    # Load the checkpoint or pretrained model - pretraining the bayesian model for cub200
    if args.pretrain is True and args.bayesian_model is True:
        method.checkpoint_saver_loader()


    logger.info(f"[2] Incrementally training {args.n_tasks} tasks")
    task_records = defaultdict(list)

    
    # Main Loop - cur_iter is not the best name for the current task
    for cur_iter in range(args.n_tasks):
        # ----------------------------------------
        # TENSRBOARD
        # ---------------------------------------
        f = 'tensorboard/'+ Exp_name +'/task_' + str(cur_iter)
        writer = SummaryWriter(f)
        
        # for checkpointing 
        # store_as_best = [False for i in range(num_validation_losses)]

        if args.mode == "joint" and cur_iter > 0:
            return

        print("\n" + "#" * 50)
        print(f"# Task {cur_iter} iteration")
        print("#" * 50 + "\n")
        logger.info("[2-1] Prepare a datalist for the current task")

        task_acc = 0.0
        eval_dict = dict()

        # get datalist
        cur_train_datalist = get_train_datalist(args, cur_iter)
        cur_valid_datalist = get_valid_datalist(args, args.exp_name, cur_iter)
        cur_test_datalist = get_test_datalist(args, args.exp_name, cur_iter)

        # Reduce datalist in Debug mode
        if args.debug:
            random.shuffle(cur_train_datalist)
            random.shuffle(cur_test_datalist)
            cur_train_datalist = cur_train_datalist[:2560]
            cur_test_datalist = cur_test_datalist[:2560]
        
        logger.info("[2-2] Set environment for the current task")
        method.set_current_dataset(cur_train_datalist, cur_test_datalist, cur_valid_datalist)
        # Increment known class for current task iteration.
        # reinitalize optimizer and scheduler?
        # Add the trainable parameters for the optimizer

        method.before_task(cur_train_datalist, cur_iter, args.init_model, args.init_opt, 
                           args.bayesian_model)

        # The way to handle streamed samples
        logger.info(f"[2-3] Start to train under {args.stream_env}")
        
        if args.stream_env == "offline" or args.mode == "joint" or args.mode == "gdumb":
            # Offline Train
            # returns best eval accuracy and not the last one
            # task_acc is the evaluation accuracy and not the training accuracy
        
            # -----------------------------------------------------------------------------------------------------------------
            # Ray Tune for the first task of the blurry case
            # -----------------------------------------------------------------------------------------------------------------
            
            if args.exp_name == "blurry10" and cur_iter==0:
                # configs has already been defined.
                
                configs={"lr": tune.loguniform(1e-4, 1e-1), 
                    "weight_decay": tune.uniform(1e-8, 1e-1)
                    }
                hyperopt_search = HyperOptSearch(metric='accuracy', mode='max')
                #hyperopt_search = BayesOptSearch(metric='loss', mode='min',points_to_evaluate=[{"lamda": 1}, {"lamda": 25}]
                scheduler = ASHAScheduler(
                    metric="accuracy",
                    mode="max",
                    max_t=100,
                    grace_period=5,
                    reduction_factor=2)
                
                reporter = CLIReporter(
                    parameter_columns=["lr", "wd"],
                    metric_columns=["loss", "accuracy", "training_iteration"]
                    )
                
                #pickle.dumps(tune.with_parameters(method.find_hyperparametrs))
                
                '''
                def train_fn(config):
                    method.train(
                        cur_iter=cur_iter,
                        n_epoch=args.n_epoch,
                        batch_size=args.batchsize,
                        n_worker=args.n_worker,
                        writer=writer,
                        **config)
                '''

                result = tune.run(
                                method.find_hyperparametrs,
                                #max_concurrent_trials=2,  # adding this to avoid the paralelism issue? 
                                resources_per_trial={"cpu": 24, "gpu": 0.5},
                                config=configs,
                                num_samples=1,
                                search_alg=hyperopt_search,
                                scheduler=scheduler,
                                #keep_checkpoints_num=2,
                                checkpoint_score_attr="accuracy", 
                                #progress_reporter=reporter
                                )
               
                '''
                # this is another way of doing it! 
                tuner = tune.Tuner(
                    method,
                    run_config = air.RunConfig(stop={"accuracy": 0.95,"training_iteration":1,},checkpoint_config=air.CheckpointConfig(checkpoint_at_end=True, checkpoint_frequency=3),),
                    tune_config=tune.TuneConfig(search_alg=hyperopt_search, scheduler=scheduler, num_samples=1,),
                    param_space={"lr": tune.loguniform(1e-4, 1e-1), "weight_decay":tune.uniform(1e-8, 1e-1)},
                    )
                    #tuner = tune.Tuner.restore(path="/visinf/home/shamidi/ray_results/RM_trainable_2023-05-19_14-48-56")
                result_grid = tuner.fit()   
            
                num_results = len(result_grid)

                # Check if there have been errors
                if result_grid.errors:
                    print("At least one trial failed.")

                # Get the best result
                best_result = result_grid.get_best_result()

                # And the best checkpoint
                best_checkpoint = best_result.checkpoint

                # And the best metrics
                best_metric = best_result.metrics
                '''            
        
                best_trial = result.get_best_trial("accuracy", "max", "last")
                logging.info("Best trial config: {}".format(best_trial.config))
                logging.info("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
                logging.info("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

                #best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="accuracy", mode="max")
                #best_checkpoint_dir = best_checkpoint.to_directory(path="directory")
                #model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))

                #best_trained_model = model._network
                #best_trained_model.load_state_dict(model_state)
                method.self.model.load_state_dict(model_state)
                #model.check_fisher()
                method.self.model.load_state_dict(model_state)

                TEST_LOADER = method.get_dataloader(batch_size=args.batch_size, n_worker=args.n_worker, train_list=None, test_list=method.self.test_list, valid_list=None)
                test_dictionary=method.evaluation(TEST_LOADER, method.self.criterion)
                print("Best trial test set accuracy: {}".format(test_dictionary["avg_acc"]))
            
            # ---------------------------------------------------------------------------------------------------------------------------------------------------------------
            else: 
            
                task_acc, eval_dict = method.find_hyperparametrs(
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

            # No streamed training data, train with only the memory_list
            method.set_current_dataset([], cur_test_datalist, cur_valid_datalist)

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
        directory = f"checkpoints/{args.dataset}/{Exp_name}/task_{cur_iter}/"
        #os.makedirs(directory, exist_ok=True)
        method.save_checkpoint(directory, task_records, store_as_best=False, store_prefixes="total_loss")
        

        # Notify to NSML
        logger.info("[2-5] Report task result")
        writer.add_scalar("Metrics/TaskAcc", task_acc, cur_iter)
    # I remember there was an error running this line of code

   
    if os.path.exists(f"results/{args.dataset}") is False:
        os.makedirs(f"results/{args.dataset}", exist_ok=True)
    np.save(f"results/{save_path}.npy", task_records["cls_acc"])
    

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
