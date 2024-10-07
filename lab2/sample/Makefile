CC = gcc
CXX = g++
CFLAGS = -lm -O3
hello_pthread: CFLAGS += -pthread
hello_omp: CFLAGS += -fopenmp
hello_hybrid: CC = mpicc
hello_hybrid: CXX = mpicxx
hello_hybrid: CFLAGS += -fopenmp
CXXFLAGS = $(CFLAGS)
TARGETS = hello_pthread hello_omp hello_hybrid

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean: 
	rm -f $(TARGETS) $(TARGETS:=.o)
