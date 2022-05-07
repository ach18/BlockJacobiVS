#!/bin/bash
DEFINED_FUNCTIONS="-DPRRBJRS -DCOLBNSVD -DPRRBNSVD -DCOLBNSVD_AVX -DPRRBNSVD_AVX"
FLAGS="${DEFINED_FUNCTIONS} -DMKL_ILP64 -DMKL_INT=size_t -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -qopenmp -lm -ldl -std=c++11"
INCLUDE="-I${MKLROOT}/include"
SRC_FILES="main.cpp ./src/utils/util.cpp ./src/utils/matrix.cpp ./src/utils/block.cpp ./src/svd/one-sided/svd.cpp ./src/svd/two-sided/nsvd.cpp ./src/svd/two-sided/svd_subprocedure.cpp ./src/svd/two-sided/svd_blocked.cpp"
COMP_FILE=main

# icpc main.cpp ./src/utils/util.cpp ./src/utils/matrix.cpp ./src/utils/block.cpp ./src/svd/one-sided/svd.cpp \
#./src/svd/two-sided/nsvd.cpp ./src/svd/two-sided/svd_subprocedure.cpp ./src/svd/two-sided/svd_blocked.cpp \
#-DMKL_INT=size_t -DMKL_ILP64 -DPRRBJRS -DCOLBNSVD -DPRRBNSVD -DCOLBNSVD_AVX -DPRRBNSVD_AVX \
#-L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -std=c++11 -qopenmp  \
#-I${MKLROOT}/include \
#-o main

icpc $SRC_FILES \
$FLAGS  \
$INCLUDE \
-o $COMP_FILE