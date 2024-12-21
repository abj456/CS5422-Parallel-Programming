CC = mpicc
CXX = mpicxx 
NVCC = nvcc
HIPCC = hipcc

LDLIBS = -pthread
CFLAGS = -O3 -fopenmp 
CXXFLAGS = -O3 -fopenmp 
EXES1 = hw3-1
EXES2 = hw3-2
EXES3 = hw3-3

NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -Xcompiler -fopenmp
HIPCCFLAGS = -std=c++11 -O3 --offload-arch=gfx90a -Xcompiler -fopenmp

LDFLAGS = -lm

.PHONY: all hw3-2 hw3-3 clean
all: $(EXES1)

hw3-2: hw3-2.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3-3: hw3-3.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

clean:
	rm -f $(EXES1) $(EXES1:=.o) $(EXES2) $(EXES3)
