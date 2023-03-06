"""
rainbow-memory
Copyright 2021-present NAVER Corp.
GPLv3
"""
import torch_optimizer
from easydict import EasyDict as edict
from torch import optim

from models import cub200_mnvi, mnist, cifar, imagenet, cub200
import pdb

def select_optimizer(opt_name, lr, model, sched_name="cos"):
    if opt_name == "adam":
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    elif opt_name == "radam":
        opt = torch_optimizer.RAdam(model.parameters(), lr=lr, weight_decay=0.00001)
    elif opt_name == "sgd":
        #  momentum=0.9, nesterov=True, weight_decay=1e-4
        opt = optim.SGD(
            model.parameters(), lr=lr, momentum=0, nesterov=False, weight_decay=0
        )
    else:
        raise NotImplementedError("Please select the opt_name [adam, sgd]")

    # what is cosine annealing warm restarts?
    if sched_name == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=1, T_mult=2, eta_min=lr * 0.01
        )
    elif sched_name == "anneal":
        scheduler = optim.lr_scheduler.ExponentialLR(opt, 1 / 1.1, last_epoch=-1)
    elif sched_name == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            opt, milestones=[30, 60, 80, 90], gamma=0.1
        )
    elif sched_name == "none":
        scheduler = None
        
    else:
        raise NotImplementedError(
            "Please select the sched_name [cos, anneal, multistep]"
        )

    return opt, scheduler


def select_model(model_name, dataset, num_classes=None, kwargs=None):
    # send the kwarg* to select model, then if bayesian is true, create a new opt
    # for the bayesian model which contains the prior args
    '''
    min_variance=1e-5, 
    mnv_init=-3.0, 
    prior_precision=1e0, 
    prior_mean=0.0,
    model_kl_div_weight
    '''
    opt = edict(
        {
            "depth": 18,
            "num_classes": num_classes,
            "in_channels": 3,
            "bn": True,
            "normtype": "BatchNorm",
            "activetype": "ReLU",
            "pooltype": "MaxPool2d",
            "preact": False,
            "affine_bn": True,
            "bn_eps": 1e-6,
            "compression": 0.5,
            "bayesian": kwargs['bayesian_model'],
            "min_variance": kwargs['min_variance'],
            "mnv_init": kwargs['mnv_init'],
            "prior_precision": kwargs['prior_precision'],
            "prior_mean": kwargs['prior_mean'],
            "model_kl_div_weight": kwargs['model_kl_div_weight'],
        }
    )
    if kwargs['bayesian_model'] == True:
        if "mnist" in dataset:
            model_class = getattr(mnist, "MLP")
        elif "cifar" in dataset:
            model_class = getattr(cifar, "ResNet")
        elif "imagenet" in dataset:
            model_class = getattr(imagenet, "ResNet")
        elif "cub200" in dataset:
            model_class = getattr(cub200_mnvi, "ResNet")
        else:
            raise NotImplementedError(
                "The bayesian model is not implemented for the selected dataset."
            )
    else:    
        if "mnist" in dataset:
            model_class = getattr(mnist, "MLP")
        elif "cifar" in dataset:
            model_class = getattr(cifar, "ResNet")
        elif "imagenet" in dataset:
            model_class = getattr(imagenet, "ResNet")
        elif "cub200" in dataset:
            model_class = getattr(cub200, "ResNet")
        else:
            raise NotImplementedError(
                "Please select the appropriate datasets (mnist, cifar10, cifar100, imagenet)"
            )

    if model_name == "resnet18":
        opt["depth"] = 18
    elif model_name == "resnet32":
        opt["depth"] = 32
    elif model_name == "resnet34":
        opt["depth"] = 34
    elif model_name == "mlp400":
        opt["width"] = 400
    else:
        raise NotImplementedError(
            "Please choose the model name in [resnet18, resnet32, resnet34]"
        )

    model = model_class(opt)

    return model
