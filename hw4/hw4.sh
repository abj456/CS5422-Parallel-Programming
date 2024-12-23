#!/bin/bash
TESTCASES=$1
echo $TESTCASES
METRICS_ARR=("achieved_occupancy" "sm_efficiency" \
             "shared_load_throughput" "shared_store_throughput" \
             "gld_throughput" "gst_throughput")
DIR_NAME="nvprof_hw4"
mkdir -p ${DIR_NAME}

module load nvhpc-nompi
make clean
make
# srun -p nvidia -N1 -n1 --gres=gpu:1 \
#     nvprof --print-gpu-summary --log-file ${DIR_NAME}/${TESTCASES}_gpu_summary.txt \
#     ./hw4 testcases/${TESTCASES} ${TESTCASES}.out

# for METRIC in ${METRICS_ARR[@]}
# do
#     echo $METRIC
#     srun -p nvidia -N1 -n1 --gres=gpu:1 \
#         nvprof --metrics ${METRIC} --log-file ${DIR_NAME}/${TESTCASES}_${METRIC}.txt \
#         ./hw4 testcases/${TESTCASES} ${TESTCASES}.out
# done

srun -p nvidia -N1 -n1 --gres=gpu:1 \
    ./hw4 testcases/${TESTCASES} ${TESTCASES}_gpu.out

# ./seq-flashattention testcases/${TESTCASES} ${TESTCASES}_cpu.out
rm *.out
