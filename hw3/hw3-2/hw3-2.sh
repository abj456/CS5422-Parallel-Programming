#!/bin/bash
TESTCASES="p11k1"
echo $TESTCASES
module load nvhpc-nompi rocm
make
# ./hw3-2 testcases/${TESTCASES} ${TESTCASES}.out

mkdir -p nsys_reports
nsys profile \
    -o "./nsys_reports/wewe_phase3.nsys-rep" \
    --trace cuda,nvtx \
    --force-overwrite true \
    ./hw3-2 testcases/${TESTCASES} ${TESTCASES}.out

# diff <(hw3-cat ${TESTCASES}.out) <(hw3-cat testcases/${TESTCASES}.out)
