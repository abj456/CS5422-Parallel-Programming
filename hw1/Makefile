CC = mpicc
CXX = mpicxx
CXXFLAGS = -O3 -lm
CFLAGS = -O3 -lm
TARGETS = hw1

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)

.PHONY: clear
clear:
	rm -f *.out

.PHONY: mpiio
mpiio: 
	$(CXX) $(CXXFLAGS) mpiio.cc -o mpiio

.PHONY: debug
debug:
	$(CXX) $(CXXFLAGS) debug.cc -o debug