#! /bin/bash
#SBATCH -N 3
#SBATCH -n 14
#SBATCH -J hw1
# make clear
TEST_N="4"
TESTCASE="01"
module load mpi
srun -N2 -n4 $HOME/hw1/sample/hw1 ${TEST_N} $HOME/hw1/testcases/${TESTCASE}.in $HOME/hw1/testcases/${TESTCASE}.out
# srun -N2 -n4 $HOME/hw1/sample/hw1 4 $HOME/hw1/testcases/01.in $HOME/hw1/testcases/01.out
# srun -N1 -n1 $HOME/hw1/sample/hw1 50 $HOME/hw1/testcases/04.in $HOME/hw1/testcases/04.out
# srun -N2 -n12 $HOME/hw1/sample/hw1 100 $HOME/hw1/testcases/05.in $HOME/hw1/testcases/05.out
# srun -N2 -n10 $HOME/hw1/sample/hw1 65536 $HOME/hw1/testcases/06.in $HOME/hw1/testcases/06.out
# srun -N3 -n3 $HOME/hw1/sample/hw1 12345 $HOME/hw1/testcases/07.in $HOME/hw1/testcases/07.out
# srun -N2 -n24 $HOME/hw1/sample/hw1 12347 $HOME/hw1/testcases/21.in $HOME/hw1/testcases/21.out
# srun -N2 -n24 $HOME/hw1/sample/hw1 1000003 $HOME/hw1/testcases/28.in ./28.out
hw1-floats /share/testcases/hw1/${TESTCASE}.out $HOME/hw1/testcases/${TESTCASE}.out