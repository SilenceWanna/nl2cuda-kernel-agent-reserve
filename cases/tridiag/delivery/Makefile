# 三对角求解 Thomas 纯 CUDA 交付版 —— 独立编译（仅需 nvcc + CUDA runtime，不依赖 PyTorch）
#
#   make            编译静态库 libtridiag.a 与自测程序 tridiag_test
#   make test       编译并运行自测（CPU 参考对拍，打印 PASS/FAIL）
#   make clean      清理
#
# 目标架构默认 sm_80（A100）。改其他卡：make ARCH=sm_75（T4）等。

NVCC   ?= nvcc
ARCH   ?= sm_80
NVFLAGS = -O3 -arch=$(ARCH) -std=c++17

all: tridiag_test libtridiag.a

# 交付 kernel 目标文件
tridiag_kernels.o: tridiag_kernels.cu
	$(NVCC) $(NVFLAGS) -c tridiag_kernels.cu -o tridiag_kernels.o

# 静态库（供外部 C/C++ 项目链接；接口见 tridiag_kernels.cu 顶部 extern "C" 声明）
libtridiag.a: tridiag_kernels.o
	ar rcs libtridiag.a tridiag_kernels.o

# 自测程序
tridiag_test: tridiag_kernels.cu tridiag_test.cu
	$(NVCC) $(NVFLAGS) tridiag_kernels.cu tridiag_test.cu -o tridiag_test

test: tridiag_test
	./tridiag_test

clean:
	rm -f tridiag_test tridiag_kernels.o libtridiag.a

.PHONY: all test clean
