#!/bin/bash

#PBS -l nodes=1:ppn=4:gpus=1
#PBS -l mem=12gb
#PBS -l walltime=03:00:00
#PBS -e myprog.err
#PBS -o myprog.out
#PBS -q default-cpu
source ~/.bashrc
workon apple

cd /misc/lmbraid19/mittal/ssl_baselines/forked/pytorch-cifar
python main_ft_imagenet.py >> output_resnet50.txt
