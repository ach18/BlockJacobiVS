#!/bin/bash

FLAGS="-DMKL_ILP64 -DMKL_INT=size_t -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -std=c++11 -qopenmp"
INCLUDE="-I${MKLROOT}/include"
SRC_FILES="./src/utils/util.cpp ./src/utils/matrix.cpp ./src/utils/block.cpp ./src/svd/one-sided/svd.cpp ./src/svd/two-sided/nsvd.cpp main.cpp ./src/svd/two-sided/svd_subprocedure.cpp ./src/svd/two-sided/svd_blocked.cpp"
OBJ_FILE=main

# icpc -DMKL_ILP64 -DMKL_INT=size_t -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -std=c++11 -qopenmp -I${MKLROOT}/include  ./src/utils/util.cpp ./src/utils/matrix.cpp ./src/utils/block.cpp ./src/svd/one-sided/svd.cpp ./src/svd/two-sided/nsvd.cpp main.cpp ./src/svd/two-sided/svd_subprocedure.cpp ./src/svd/two-sided/svd_blocked.cpp -o main
icpc $FLAGS $INCLUDE $SRC_FILES -o $OBJ_FILE