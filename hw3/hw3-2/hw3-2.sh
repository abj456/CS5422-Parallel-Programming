#!/bin/bash
TESTCASES="p24k1"
echo $TESTCASES
module load nvhpc-nompi rocm
make
srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof \
    --metrics gst_throughput \
    ./hw3-2 testcases/${TESTCASES} ${TESTCASES}.out
    # --metrics achieved_occupancy \
    # --metrics sm_efficiency \
    # --metrics shared_load_throughput \
    # --metrics shared_store_throughput \
    # --metrics gld_throughput \
    # --metrics gst_throughput \

# mkdir -p nsys_reports
# nsys profile \
#     -o "./nsys_reports/wewe_phase3.nsys-rep" \
#     --trace cuda,nvtx \
#     --force-overwrite true \
#     ./hw3-2 testcases/${TESTCASES} ${TESTCASES}.out

# diff <(hw3-cat ${TESTCASES}.out) <(hw3-cat testcases/${TESTCASES}.out)
