#! /bin/bash
module load mpi nsys
srun -N1 -n1 ./wrapper.sh ./hw1_nvtx 536869888 ./testcases/40.in 40.out && rename nsys_reports nsys_reports_n1 nsys_reports
srun -N1 -n2 ./wrapper.sh ./hw1_nvtx 536869888 ./testcases/40.in 40.out && rename nsys_reports nsys_reports_n2 nsys_reports
srun -N1 -n4 ./wrapper.sh ./hw1_nvtx 536869888 ./testcases/40.in 40.out && rename nsys_reports nsys_reports_n4 nsys_reports
srun -N1 -n8 ./wrapper.sh ./hw1_nvtx 536869888 ./testcases/40.in 40.out && rename nsys_reports nsys_reports_n8 nsys_reports
srun -N1 -n12 ./wrapper.sh ./hw1_nvtx 536869888 ./testcases/40.in 40.out && rename nsys_reports nsys_reports_n12 nsys_reports
srun -N2 -n24 ./wrapper.sh ./hw1_nvtx 536869888 ./testcases/40.in 40.out && rename nsys_reports nsys_reports_n24 nsys_reports
