#/bin/bash

# CIL CONFIG
MODE="rm" # joint, gdumb, icarl, rm, ewc, rwalk, bic   # here I can add the Bayesian method? althoug the Bayesian method is more of a architecture than a method! the loss maybe is! 
# "default": If you want to use the default memory management method.
MEM_MANAGE="random" # default, random, random_balanced, reservoir, uncertainty, prototype. ToDo: add: consistent_high, consistent_low
RND_SEED=1
DATASET="cifar10" # mnist, cifar10, cifar100, imagenet, cub200
STREAM="offline" # offline, online
EXP="blurry10" # disjoint, blurry10, blurry30
MEM_SIZE=500 # cifar10: k={200, 500, 1000}, mnist: k=500, cifar100: k=2,000, imagenet: k=20,000, cub200:k={340}
TRANS="cutmix autoaug" # multiple choices: cutmix, cutout, randaug, autoaug

N_WORKER=4
JOINT_ACC=0.0 # training all the tasks at once.
# FINISH CIL CONFIG ####################

UNCERT_METRIC="vr_randaug"
PRETRAIN="" INIT_MODEL="" INIT_OPT="--init_opt"

# iCaRL
FEAT_SIZE=2048

# BiC
distilling="--distilling" # Normal BiC. If you do not want to use distilling loss, then "".

# Expanding memory CONFIG
EXP_MEM="" # default: false - {True, Flase}
CORSET_SIZE=100

# Bayesian CONFIG
BAYESIAN="" # True, False
MEAN_VARIANCE=1e-5 # cifar10: 1e-5, cifar100: 1e-5, imagenet: 1e-5, cub200: 1e-5
MNV_INIT=-3.0 # cifar10: -3.0, cifar100: -3.0, imagenet: -3.0, cub200: -3.0
PRIOR_PRECISION=1e4 # cifar10: 1e4, cifar100: 10, imagenet: 10, cub200: 10
PRIOR_MEAN=0.0 # cifar10: 0.0, cifar100: 0.0, imagenet: 0.0, cub200: 0.0
KL_DIV_WEIGHT=5e-8 # cifar10: 5e-8, cifar100: 5e-7, imagenet: 5e-7, cub200: 5e-7
PRIOR_CONVERSION_FUNCTION="none" # {"sqrt", exp, mul2, mul3, mul4, mul8, log, pow2, pow3, div, none}
KLD_WEIGHT_ATTE="" # True, False
INFORMED_PRIOR="" # True, False


# Checkpoint
CHECKPOINT="./cub_split/checkpoint_imagenet_mnvi.ckpt" # path to the checkpoint?
CHECKPOINT_INCLUDE="[*]"
CHECKPOINT_EXCLUDE="[_model.module.resnet.fc.weight,_model.module.resnet.fc.mult_noise_variance,_model.module.resnet.fc.bias,_model.module.resnet.fc.bias_variance]"
CHECKPOINT_MODE="resume_from_latest" # "resume_from_latest", "resume_from_best"

# Regularization: Early_stopping
EARLY_STOPPING="" # True, False

if [ "$DATASET" == "mnist" ]; then
    TOTAL=50000 N_VAL=250 N_CLASS=10 TOPK=1
    MODEL_NAME="mlp400"
    N_EPOCH=5; BATCHSIZE=16; LR=0.05 OPT_NAME="sgd" SCHED_NAME="cos"
    if [ "${MODE_LIST[0]}" == "joint" ]; then
        N_INIT_CLS=10 N_CLS_A_TASK=10 N_TASKS=1
    elif [[ "$EXP" == *"blurry"* ]]; then
        N_INIT_CLS=10 N_CLS_A_TASK=2 N_TASKS=5
    else
        N_INIT_CLS=2 N_CLS_A_TASK=2 N_TASKS=5

    fi
elif [ "$DATASET" == "cifar10" ]; then
    TOTAL=50000 N_VAL=250 N_CLASS=10 TOPK=1
    MODEL_NAME="resnet18"
    N_EPOCH=300; BATCHSIZE=256; LR=0.01 OPT_NAME="sgd" SCHED_NAME="cos" #128
    if [ "${MODE_LIST[0]}" == "joint" ]; then
        N_INIT_CLS=10 N_CLS_A_TASK=10 N_TASKS=1
    elif [[ "$EXP" == *"blurry"* ]]; then
        N_INIT_CLS=10 N_CLS_A_TASK=2 N_TASKS=5
    else
        N_INIT_CLS=2 N_CLS_A_TASK=2 N_TASKS=5

    fi
elif [ "$DATASET" == "cifar100" ]; then
    TOTAL=50000 N_VAL=0 N_CLASS=100 TOPK=1
    MODEL_NAME="resnet18" #resnet32
    N_EPOCH=100; BATCHSIZE=128; LR=0.01 OPT_NAME="sgd" SCHED_NAME="multistep"
    if [ "${MODE_LIST[0]}" == "joint" ]; then
        N_INIT_CLS=100 N_CLS_A_TASK=100 N_TASKS=1
    elif [[ "$EXP" == *"blurry"* ]]; then
        N_INIT_CLS=100 N_CLS_A_TASK=20 N_TASKS=5
    else
        N_INIT_CLS=20 N_CLS_A_TASK=20 N_TASKS=5
    fi

elif [ "$DATASET" == "cub200" ]; then
    TOTAL=50000 N_VAL=0 N_CLASS=180 TOPK=1  # what is TOTAL? how many data points do we have in the training set of the original dataset? 
    MODEL_NAME="resnet18"
    N_EPOCH=500; BATCHSIZE=32; LR=0.01 OPT_NAME="sgd" SCHED_NAME="multistep"  #N_EPOCH=256; BATCHSIZE=16; LR=0.05 OPT_NAME="sgd" SCHED_NAME="cos"
    if [ "${MODE_LIST[0]}" == "joint" ]; then
        N_INIT_CLS=180 N_CLS_A_TASK=180 N_TASKS=1
    elif [[ "$EXP" == *"blurry"* ]]; then
        N_INIT_CLS=180 N_CLS_A_TASK=20 N_TASKS=9
    else
        N_INIT_CLS=20 N_CLS_A_TASK=20 N_TASKS=9
    fi

elif [ "$DATASET" == "imagenet" ]; then
    TOTAL=50000 N_VAL=0 N_CLASS=1000 TOPK=5
    MODEL_NAME="resnet34"
    N_EPOCH=100; BATCHSIZE=256; LR=0.05 OPT_NAME="sgd" SCHED_NAME="multistep"
    if [ "${MODE_LIST[0]}" == "joint" ]; then
        N_INIT_CLS=1000 N_CLS_A_TASK=100 N_TASKS=10
    elif [[ "$EXP" == *"blurry"* ]]; then
        N_INIT_CLS=1000 N_CLS_A_TASK=100 N_TASKS=10
    else
        N_INIT_CLS=100 N_CLS_A_TASK=100 N_TASKS=10
    fi
else
    echo "Undefined setting"
    exit 1
fi

CUDA_VISIBLE_DEVICES=3 CUDA_LAUNCH_BLOCKING=1 CUBLAS_WORKSPACE_CONFIG=:16:8 python main.py --mode $MODE --mem_manage $MEM_MANAGE --exp_name $EXP \
--dataset $DATASET --stream_env $STREAM  $INIT_MODEL $INIT_OPT --topk $TOPK \
--n_tasks $N_TASKS --n_cls_a_task $N_CLS_A_TASK --n_init_cls $N_INIT_CLS \
--rnd_seed $RND_SEED --model_name $MODEL_NAME --opt_name $OPT_NAME --pretrain $PRETRAIN --sched_name $SCHED_NAME \
--lr $LR --batchsize $BATCHSIZE --n_epoch $N_EPOCH --n_worker $N_WORKER \
--memory_size $MEM_SIZE --transform $TRANS --uncert_metric $UNCERT_METRIC \
--feature_size $FEAT_SIZE $distilling --joint_acc $JOINT_ACC \
--expanding_memory $EXP_MEM --coreset_size $CORSET_SIZE --bayesian_model $BAYESIAN --min_variance $MEAN_VARIANCE \
--mnv_init $MNV_INIT --prior_precision $PRIOR_PRECISION --prior_mean $PRIOR_MEAN --model_kl_div_weight $KL_DIV_WEIGHT \
--prior_conv_function $PRIOR_CONVERSION_FUNCTION --kld_weight_atte $KLD_WEIGHT_ATTE --checkpoint_path $CHECKPOINT --checkpoint_include_params $CHECKPOINT_INCLUDE \
 --checkpoint_exclude_params $CHECKPOINT_EXCLUDE --checkpoint_mode $CHECKPOINT_MODE --informed_prior $INFORMED_PRIOR --early_stopping $EARLY_STOPPING