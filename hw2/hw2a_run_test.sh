#! /bin/bash
module load mpi nsys
thds=1
srun -n1 -c$thds ./wrapper.sh ./hw2a_nvtx out.png 10000 -0.6743255348701748 -0.6742373110336833 0.3623527741766198 0.36230599099654903 7680 4320
mv nsys_reports nsys_reports_c$thds
echo "hw2a thread $thds done"

thds=2
srun -n1 -c$thds ./wrapper.sh ./hw2a_nvtx out.png 10000 -0.6743255348701748 -0.6742373110336833 0.3623527741766198 0.36230599099654903 7680 4320
mv nsys_reports nsys_reports_c$thds
echo "hw2a thread $thds done"

thds=4
srun -n1 -c$thds ./wrapper.sh ./hw2a_nvtx out.png 10000 -0.6743255348701748 -0.6742373110336833 0.3623527741766198 0.36230599099654903 7680 4320
mv nsys_reports nsys_reports_c$thds
echo "hw2a thread $thds done"

thds=8
srun -n1 -c$thds ./wrapper.sh ./hw2a_nvtx out.png 10000 -0.6743255348701748 -0.6742373110336833 0.3623527741766198 0.36230599099654903 7680 4320
mv nsys_reports nsys_reports_c$thds
echo "hw2a thread $thds done"

thds=12
srun -n1 -c$thds ./wrapper.sh ./hw2a_nvtx out.png 10000 -0.6743255348701748 -0.6742373110336833 0.3623527741766198 0.36230599099654903 7680 4320
mv nsys_reports nsys_reports_c$thds
echo "hw2a thread $thds done"

thds=16
srun -n1 -c$thds ./wrapper.sh ./hw2a_nvtx out.png 10000 -0.6743255348701748 -0.6742373110336833 0.3623527741766198 0.36230599099654903 7680 4320
mv nsys_reports nsys_reports_c$thds
echo "hw2a thread $thds done"

thds=24
srun -n1 -c$thds ./wrapper.sh ./hw2a_nvtx out.png 10000 -0.6743255348701748 -0.6742373110336833 0.3623527741766198 0.36230599099654903 7680 4320
mv nsys_reports nsys_reports_c$thds
echo "hw2a thread $thds done"
