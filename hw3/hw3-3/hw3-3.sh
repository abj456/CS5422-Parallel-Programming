#!/bin/bash
TESTCASES=$1
BF="32"
OPT="ReduceSyncthreads"
echo $TESTCASES
echo "Blocking Factor = ${BF}"
module load nvhpc-nompi rocm
make
METRICS_ARR=("achieved_occupancy" "sm_efficiency" \
             "shared_load_throughput" "shared_store_throughput" \
             "gld_throughput" "gst_throughput")
# METRICS_ARR=("inst_integer")

DIR_NAME="nvprof_${BF}"
mkdir -p ${DIR_NAME}
srun -p nvidia -N1 -n1 -c2 --gres=gpu:2 \
    nvprof --print-gpu-summary --log-file ${DIR_NAME}/${TESTCASES}_gpu_summary.txt \
    ./hw3-3 ../hw3-2/testcases/${TESTCASES} ${TESTCASES}.out

# for METRIC in ${METRICS_ARR[@]}
# do
#     echo $METRIC
#     srun -p nvidia -N1 -n1 --gres=gpu:1 \
#         nvprof --metrics ${METRIC} --log-file ${DIR_NAME}/${TESTCASES}_${METRIC}.txt \
#         ./hw3-2 testcases/${TESTCASES} ${TESTCASES}.out
# done

rm *.out


# mkdir -p nsys_reports
# nsys profile \
#     -o "./nsys_reports/wewe_phase3.nsys-rep" \
#     --trace cuda,nvtx \
#     --force-overwrite true \
#     ./hw3-2 testcases/${TESTCASES} ${TESTCASES}.out
# diff <(hw3-cat ${TESTCASES}.out) <(hw3-cat testcases/${TESTCASES}.out)
