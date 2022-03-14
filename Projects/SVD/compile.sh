#!/bin/bash

FLAGS="-DMKL_ILP64 -DMKL_INT=size_t -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -std=c++11"
INCLUDE="-I${MKLROOT}/include"
SRC_FILES="./src/utils/util.cpp ./src/svd/one-sided/svd.cpp ./src/utils/matrix.cpp"
OBJ_FILE=main

# icpc -DMKL_ILP64 -DMKL_INT=size_t -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -std=c++11 -I${MKLROOT}/include  ./src/utils/util.cpp ./src/svd/one-sided/svd.cpp ./src/utils/matrix.cpp  main.cpp -o main
icpc $FLAGS $INCLUDE $SRC_FILES -o $OBJ_FILE