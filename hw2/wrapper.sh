#! /bin/bash

mkdir -p nsys_reports

# Output to ./nsys_reports/rank_$N.nsys-rep
nsys profile \
    -o "./nsys_reports/rank_$PMI_RANK.nsys-rep" \
    --mpi-impl mpich \
    --trace mpi,ucx,osrt,nvtx \
    $@
