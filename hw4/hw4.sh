#!/bin/bash
module load nvhpc-nompi rocm
make clean
make
# TESTCASE="t30"
./hw4 testcases/$1 $1.out
# srun -N1 -n1 --gres=gpu:1 ./hw4 testcases/$1 $1.out