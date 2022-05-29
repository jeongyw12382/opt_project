#!/bin/bash

optimizer="$1"
level="$2"

lr_scheduler=linear_warmup_cosine_annealing
case $optimizer in
    sgd)
        lr_init=1
        ;;
    sgd_momentum)
        lr_init=1
        ;;
    adam)
        lr_init=0.1
        ;;
    rmsprop)
        lr_init=0.01
        ;;
    radam)
        lr_init=0.1
        ;;
esac

python3 -m run --dset cifar10 --noise_level $2 --global_step 50000 --optimizer $1 --lr_init $lr_init --lr_scheduler $lr_scheduler