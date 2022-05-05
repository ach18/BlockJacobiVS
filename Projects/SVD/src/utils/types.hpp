#pragma once
#include <cstddef>
#include <vector>

struct vector_t {
    double* ptr;
    std::size_t len;
};

/*
* double* ptr - указатель на одноразмерный массив
* std::size_t rows, cols - число строк и столбцов матрицы
* char storage - тип хранения элементов в памяти: 'R' по строкам, 'C' по столбцам
*/
struct matrix_t {
    double* ptr;
    std::size_t rows;
    std::size_t cols;
    char storage;
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