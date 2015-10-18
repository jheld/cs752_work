CUDA_PATH       := /usr/local/cuda
CUDA_INC_PATH   := $(CUDA_PATH)/include
CUDA_BIN_PATH   := $(CUDA_PATH)/bin
CUDA_LIB_PATH   := $(CUDA_PATH)/lib64
NVCC            := $(CUDA_BIN_PATH)/nvcc
GCC             := g++

LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
CCFLAGS   := -m64
NVCCFLAGS := -m64
INCLUDES  := -I$(CUDA_INC_PATH) -I. -I$(CUDA_PATH)/samples/common/inc

matrixmul: matrixmul.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDS) $(LDFLAGS) -O3 -o $@ $< matrixmul_gold.cpp

simpleMatMult: simpleMatMult.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDS) $(LDFLAGS) -O3 -o $@ $<
