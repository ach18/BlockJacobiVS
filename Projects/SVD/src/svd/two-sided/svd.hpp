#pragma once

#include "../../utils/types.hpp"

size_t colbnsvd(struct matrix_t X, struct matrix_t B, struct matrix_t U, struct matrix_t V, size_t block_size, size_t ThreadsNum, double* Time);
size_t rrbnsvd(struct matrix_t Amat, struct matrix_t Bmat, struct matrix_t Umat, struct matrix_t Vmat, size_t block_size, size_t ThreadsNum, double* Time);