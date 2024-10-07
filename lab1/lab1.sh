#! /bin/bash
#SBATCH -N 2
#SBATCH -n 12
#SBATCH -c 2
#SBATCH -J lab1
# make clear
module load mpi
make 
srun time $HOME/lab1/sample/lab1 4294967295 1099511627775
