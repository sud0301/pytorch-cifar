#!/bin/bash

#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l mem=8000mb
#PBS -l walltime=10:00:00
#PBS -e myprog.err
#PBS -o myprog.out
#PBS -q default-gpu
source ~/.bashrc
workon apple

cd /misc/lmbraid19/mittal/ssl_baselines/forked/pytorch-cifar
python main.py >> badgan_D_output.txt
