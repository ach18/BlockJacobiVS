#pragma once
#include <cstddef>
#include <vector>

struct vector_t {
    double* ptr;
    std::size_t len;
};

struct matrix_t {
    double* ptr;
    std::size_t rows;
    std::size_t cols;
};

struct index_t {
    std::size_t i;
    std::size_t j;
};

struct compute_params {
    std::size_t m;
    std::size_t n;
    std::size_t threads;
    std::size_t iterations;
    double time;
};

struct string_t {
	char* ptr;
	std::size_t* len;
};