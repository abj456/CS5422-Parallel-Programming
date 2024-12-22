#!/bin/bash
TESTCASES=$1
echo $TESTCASES
module load nvhpc-nompi rocm
make
# METRICS_ARR=("achieved_occupancy" "sm_efficiency" \
#              "shared_load_throughput" "shared_store_throughput" \
#              "gld_throughput" "gst_throughput")
METRICS_ARR=("inst_integer")

for METRIC in ${METRICS_ARR[@]}
do
    echo $METRIC
    srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof \
        --metrics ${METRIC} \
        ./hw3-2 testcases/${TESTCASES} ${TESTCASES}.out
done
# mkdir -p nsys_reports
# nsys profile \
#     -o "./nsys_reports/wewe_phase3.nsys-rep" \
#     --trace cuda,nvtx \
#     --force-overwrite true \
#     ./hw3-2 testcases/${TESTCASES} ${TESTCASES}.out

# diff <(hw3-cat ${TESTCASES}.out) <(hw3-cat testcases/${TESTCASES}.out)
