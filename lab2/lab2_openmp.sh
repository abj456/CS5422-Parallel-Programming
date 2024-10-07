#!/bin/bash
#SBATCH --job-name=my_job           
#SBATCH --output=my_job_%j.out     
#SBATCH --error=my_job_%j.err      
#SBATCH --time=00:05:00           
#SBATCH --nodes=1              
srun -c4 -n1 ./lab2_omp 5 21 #4