module load nvhpc-nompi rocm
hipify-clang hw3-2.cu
mv hw3-2.cu.hip hw3-2.hip
make hw3-2-amd