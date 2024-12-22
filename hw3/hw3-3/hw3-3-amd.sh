module load nvhpc-nompi rocm
CODE_NAME="hw3-3"
hipify-clang ${CODE_NAME}.cu
mv ${CODE_NAME}.cu.hip ${CODE_NAME}.hip
make ${CODE_NAME}-amd