#pragma once
#include <cstddef>
#include <vector>

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

struct compute_params {
    size_t m;
    size_t n;
    size_t threads;
    size_t iterations;
    double time;
};

/**
 * 32-byte (256 bits) aligned vector.
 */
//template <typename T>
//using aligned_vector = std::vector<T, boost::alignment::aligned_allocator<T, 32>>;
