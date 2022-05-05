#pragma once
#include <cstdlib>
#include "matrix.hpp"

// perform C = AB
void mult_block(struct matrix_t Amat, std::size_t blockA_row, std::size_t blockA_col, struct matrix_t Bmat, std::size_t blockB_row,
    std::size_t blockB_col, struct matrix_t Cmat, std::size_t blockC_row, std::size_t blockC_col, std::size_t block_size);

// perform C = AB + C
void mult_add_block(struct matrix_t Amat, std::size_t blockA_row, std::size_t blockA_col, struct matrix_t Bmat, std::size_t blockB_row,
    std::size_t blockB_col, struct matrix_t Cmat, std::size_t blockC_row, std::size_t blockC_col, std::size_t block_size);

// perform C = (A^t)B
void mult_block_transp(struct matrix_t Amat, std::size_t blockA_row, std::size_t blockA_col, struct matrix_t Bmat, std::size_t blockB_row,
    std::size_t blockB_col, struct matrix_t Cmat, std::size_t blockC_row, std::size_t blockC_col, std::size_t block_size);

void copy_block(struct matrix_t S, std::size_t blockS_row, std::size_t blockS_col, struct matrix_t D, std::size_t blockD_row,
    std::size_t blockD_col, std::size_t block_size);

void copy_subproblem_blocks(struct matrix_t Smat, std::size_t blockS_row, std::size_t blockS_col, struct matrix_t Dmat, std::size_t blockD_row,
    std::size_t blockD_col, std::size_t block_size);
