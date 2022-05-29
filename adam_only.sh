#!/bin/bash

flag="$1"
level="$2"

case $1 in
    dropout)
        python3 -m run --dset cifar10 --dropout_level $level --global_step 50000 --optimizer adam --lr_init 0.1 --lr_scheduler linear_warmup_cosine_annealing
        ;;
    noise)
        python3 -m run --dset cifar10 --noise_level $level --global_step 50000 --optimizer adam --lr_init 0.1 --lr_scheduler linear_warmup_cosine_annealing
        ;;
esac