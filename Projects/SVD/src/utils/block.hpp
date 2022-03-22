#pragma once
#include <cstdlib>
#include "matrix.hpp"

// perform C = AB
void mult_block(struct matrix_t Amat, std::size_t blockA_row, std::size_t blockA_col, struct matrix_t Bmat, std::size_t blockB_row,
    std::size_t blockB_col, struct matrix_t Cmat, std::size_t blockC_row, std::size_t blockC_col, std::size_t block_size);

void copy_block(struct matrix_t S, std::size_t blockS_row, std::size_t blockS_col, struct matrix_t D, std::size_t blockD_row,
    std::size_t blockD_col, std::size_t block_size);
