#pragma once
#include <cstddef>
#include <vector>
#include <boost/align/aligned_allocator.hpp>

struct vector_t {
    double* ptr;
    size_t len;
};

struct matrix_t {
    double* ptr;
    size_t rows;
    size_t cols;
};

struct index_t {
    size_t i;
    size_t j;
};
