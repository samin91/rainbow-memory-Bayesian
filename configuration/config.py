"""
rainbow-memory
Copyright 2021-present NAVER Corp.
GPLv3
"""
import argparse
from utils import json
from utils import strings


def base_parser():
    parser = argparse.ArgumentParser(description="Class Incremental Learning Research")

    # Mode and Exp. Settings.
    parser.add_argument(
        "--mode",
        type=str,
        default="finetune",
        help="CIL methods [joint, rwalk, icarl, rm,  gdumb, ewc, bic]",
    )
    parser.add_argument(
        "--mem_manage",
        type=str,
        default=None,
        help="memory management [default, random, reservoir, uncertainty, prototype]",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="[mnist, cifar10, cifar100, imagenet]",
    )
    parser.add_argument("--n_tasks", type=int, default="5", help="The number of tasks")
    parser.add_argument(
        "--n_cls_a_task", type=int, default=2, help="The number of class of each task"
    )
    parser.add_argument(
        "--n_init_cls",
        type=int,
        default=1,
        help="The number of classes of initial task",
    )
    parser.add_argument("--rnd_seed", type=int, help="Random seed number.")
    parser.add_argument(
        "--memory_size", type=int, default=500, help="Episodic memory size"
    )
    parser.add_argument(
        "--stream_env",
        type=str,
        default="offline",
        choices=["offline", "online"],
        help="the restriction whether to keep streamed data or not",
    )

    # Dataset
    parser.add_argument(
        "--log_path",
        type=str,
        default="results",
        help="The path logs are saved. Only for local-machine",
    )

    # Model
    parser.add_argument(
        "--model_name", type=str, default="resnet32", help="[resnet18, resnet32]"
    )
    parser.add_argument("--pretrain", action="store_false", help="pretrain model or not")

    # Train
    parser.add_argument("--opt_name", type=str, default="sgd", help="[adam, sgd]")
    parser.add_argument("--sched_name", type=str, default="cos", help="[cos, anneal]")
    parser.add_argument("--batchsize", type=int, default=128, help="batch size")
    parser.add_argument("--n_epoch", type=int, default=30, help="Epoch")

    parser.add_argument("--n_worker", type=int, default=0, help="The number of workers")

    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--initial_annealing_period",
        type=int,
        default=20,
        help="Initial Period that does not anneal",
    )
    parser.add_argument(
        "--annealing_period",
        type=int,
        default=20,
        help="Period (Epochs) of annealing lr",
    )
    parser.add_argument(
        "--learning_anneal", type=float, default=10, help="Divisor for annealing"
    )
    parser.add_argument(
        "--init_model",
        action="store_true",
        help="Initilize model parameters for every iterations",
    )
    parser.add_argument(
        "--init_opt",
        action="store_true",
        help="Initilize optimizer states for every iterations",
    )
    parser.add_argument(
        "--topk", type=int, default=1, help="set k when we want to set topk accuracy"
    )
    parser.add_argument(
        "--joint_acc",
        type=float,
        default=0.0,
        help="Accuracy when training all the tasks at once",
    )
    # Transforms
    parser.add_argument(
        "--transforms",
        nargs="*",
        default=[],
        help="Additional train transforms [cotmix, cutout, randaug]",
    )

    # Benchmark
    parser.add_argument("--exp_name", type=str, default="", help="[disjoint, blurry]")

    # ICARL
    parser.add_argument(
        "--feature_size",
        type=int,
        default=2048,
        help="Feature size when embedding a sample",
    )

    # BiC
    parser.add_argument(
        "--distilling",
        action="store_true",
        help="use distilling loss with classification",
    )

    # Regularization
    parser.add_argument(
        "--reg_coef",
        type=int,
        default=100,
        help="weighting for the regularization loss term",
    )

    # Uncertain
    parser.add_argument(
        "--uncert_metric",
        type=str,
        default="vr",
        choices=["vr", "vr1", "vr_randaug", "loss"],
        help="A type of uncertainty metric",
    )

     # Coreset size of the expanding memory
    parser.add_argument(
        "--expanding_memory",
        action="store_true",
        help="Growing memory replaces the fixed memory when our dataset has few smaples per task",
    )
    # Coreset size for the expanding memory
    parser.add_argument(
        "--coreset_size",
        type=int,
        default=50,
        help="The number of samples in the coreset per task",
    )

    #Bayesian mode
    parser.add_argument(
        "--bayesian_model",
        action="store_true",
        help="Use Bayesian model for uncertainty estimation",
    )


    # Bayesian arguments 
    parser.add_argument(
        "--min_variance",
        type=float,
        default=1e-5,
        help="Minimum variance for Bayesian model",
    )
    parser.add_argument(
        "--mnv_init",
        type=float,
        default=-3.0,
        help="Initial value for Bayesian model",
    )

    parser.add_argument(
        "--prior_precision",
        type=float,
        default=1e0,
        help="Prior precision for Bayesian model",
    )
    parser.add_argument(
        "--prior_mean",
        type=float,
        default=0.0,
        help="Prior mean for Bayesian model",
    )
    parser.add_argument(
        "--model_kl_div_weight",
        type=float,
        default=1.0,
        help="KL divergence weight for Bayesian model",
    )



    parser.add_argument(
        "--prior_conv_function",
        type=str,
        default="none",
        choices=["sqrt", "exp", "mul2", "mul3", "mul4", "mul8", 'log', 'pow2', 'pow3', 'div', 'none'],
        help="Prior conversion function for Bayesian model",
    )

   
    parser.add_argument(
        "--checkpoint_path",
        type=strings.as_string_or_none,
        default="none",
        help="Path to the checkpoint",
    )

    parser.add_argument(
        "--checkpoint_include_params", 
        type=strings.as_stringlist_or_none, 
        default="[*]",
        help="include the following parameters to the checkpoint",
    )
    
    parser.add_argument(
        "--checkpoint_exclude_params", 
        type=strings.as_stringlist_or_none, 
        default="[]",
        help="exclude the following parameters to the checkpoint",
    ) 

    parser.add_argument(
        "--checkpoint_mode", 
        type=str, 
        default="resume_from_latest",
        choices=["resume_from_latest", "resume_from_best"]
    )

    parser.add_argument(
        "--kld_weight_atte",
        action="store_false",
        help="Use KLD weight for attention of the prior effect",   
    )

    parser.add_argument(
        "--informed_prior",
        action="store_false",
        help="update the prior of task_i with the posterior of task_i-1",   
    )

    parser.add_argument(
        "--early_stopping",
        action="store_false",
        help="early stopping on the validation set",   
    )



   
    # Debug
    parser.add_argument("--debug", action="store_true", help="Turn on Debug mode")

    args = parser.parse_args()
    return args
