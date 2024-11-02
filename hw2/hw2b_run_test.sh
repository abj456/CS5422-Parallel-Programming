#! /bin/bash
module load mpi nsys
proc=1
thds=12
srun -n$proc -c$thds ./wrapper.sh ./hw2b_nvtx out.png 10000 -0.6743255348701748 -0.6742373110336833 0.3623527741766198 0.36230599099654903 7680 4320
mv nsys_reports nsys_reports_n${proc}_c$thds
echo "hw2b proc $proc thread $thds done"

proc=2
thds=12
srun -n$proc -c$thds ./wrapper.sh ./hw2b_nvtx out.png 10000 -0.6743255348701748 -0.6742373110336833 0.3623527741766198 0.36230599099654903 7680 4320
mv nsys_reports nsys_reports_n${proc}_c$thds
echo "hw2b proc $proc thread $thds done"

proc=3
thds=12
srun -n$proc -c$thds ./wrapper.sh ./hw2b_nvtx out.png 10000 -0.6743255348701748 -0.6742373110336833 0.3623527741766198 0.36230599099654903 7680 4320
mv nsys_reports nsys_reports_n${proc}_c$thds
echo "hw2b proc $proc thread $thds done"

proc=4
thds=12
srun -n$proc -c$thds ./wrapper.sh ./hw2b_nvtx out.png 10000 -0.6743255348701748 -0.6742373110336833 0.3623527741766198 0.36230599099654903 7680 4320
mv nsys_reports nsys_reports_n${proc}_c$thds
echo "hw2b proc $proc thread $thds done"
