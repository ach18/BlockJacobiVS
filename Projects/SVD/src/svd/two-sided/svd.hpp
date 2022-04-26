#pragma once

#include "../../utils/types.hpp"

std::size_t colbnsvd(struct matrix_t X, struct matrix_t B, struct matrix_t U, struct matrix_t V, std::size_t block_size, std::size_t ThreadsNum, double* Time, struct string_t errors);
std::size_t rrbnsvd_parallel(struct matrix_t Amat, struct matrix_t Bmat, struct matrix_t Umat, struct matrix_t Vmat, std::size_t ThreadsNum, bool vectorization, double* Time, struct string_t errors);
std::size_t rrbnsvd_seq(struct matrix_t Amat, struct matrix_t Bmat, struct matrix_t Umat, struct matrix_t Vmat, std::size_t ThreadsNum, bool vectorization, double* Time, struct string_t errors);