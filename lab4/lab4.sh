#! /bin/bash
TEST="jerry"
module load rocm cuda
make sobel
srun --gres=gpu:1 ./sobel testcases/${TEST}.png ${TEST}.out.png
png-diff $TEST.out.png testcases/$TEST.out.png
