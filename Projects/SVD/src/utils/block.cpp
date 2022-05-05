#include "block.hpp"

void copy_subproblem_blocks(struct matrix_t Smat, std::size_t blockS_row, std::size_t blockS_col, struct matrix_t Dmat, std::size_t blockD_row,
	std::size_t blockD_col, std::size_t block_size) 
{
	std::size_t nS = Smat.rows;
	std::size_t nD = Dmat.rows;
	double* S = Smat.ptr;
	double* D = Dmat.ptr;

	std::size_t Sbeg0 = blockS_row * block_size * nS + blockS_row * block_size;
	std::size_t Dbeg0 = blockD_row * block_size * nD + blockD_row * block_size;
	std::size_t Sbeg1 = blockS_row * block_size * nS + blockS_col * block_size;
	std::size_t Dbeg1 = blockD_row * block_size * nD + blockD_col * block_size;
	std::size_t Sbeg2 = blockS_col * block_size * nS + blockS_row * block_size;
	std::size_t Dbeg2 = blockD_col * block_size * nD + blockD_row * block_size;
	std::size_t Sbeg3 = blockS_col * block_size * nS + blockS_col * block_size;
	std::size_t Dbeg3 = blockD_col * block_size * nD + blockD_col * block_size;


	std::size_t j = 0;
	for (std::size_t i = 0; i < block_size; ++i) {
		j = 0;
		cblas_dcopy(block_size, &S[Sbeg0 + (i * nS)], 1, &D[Dbeg0 + (i * nD)], 1);
		cblas_dcopy(block_size, &S[Sbeg1 + (i * nS)], 1, &D[Dbeg1 + (i * nD)], 1);
		cblas_dcopy(block_size, &S[Sbeg2 + (i * nS)], 1, &D[Dbeg2 + (i * nD)], 1);
		cblas_dcopy(block_size, &S[Sbeg3 + (i * nS)], 1, &D[Dbeg3 + (i * nD)], 1);
	}
}

void copy_block(struct matrix_t Smat, std::size_t blockS_row, std::size_t blockS_col, struct matrix_t Dmat, std::size_t blockD_row,
	std::size_t blockD_col, std::size_t block_size)
{
	std::size_t nS = Smat.rows;
	std::size_t nD = Dmat.rows;
	double* S = Smat.ptr;
	double* D = Dmat.ptr;

	std::size_t Sbeg = blockS_row * block_size * nS + blockS_col * block_size;
	std::size_t Dbeg = blockD_row * block_size * nD + blockD_col * block_size;
	std::size_t j = 0;
	for (std::size_t i = 0; i < block_size; ++i) {
		for (j = 0; j < block_size; j ++) {
			D[Dbeg + i * nD + j] = S[Sbeg + i * nS + j];
		}
	}
}

// perform C = AB
void mult_block(struct matrix_t Amat, std::size_t blockA_row, std::size_t blockA_col, struct matrix_t Bmat, std::size_t blockB_row,
	std::size_t blockB_col, struct matrix_t Cmat, std::size_t blockC_row, std::size_t blockC_col, std::size_t block_size) 
{
	std::size_t nA = Amat.rows;
	std::size_t nB = Bmat.rows;
	std::size_t nC = Cmat.rows;
	double* A = Amat.ptr;
	double* B = Bmat.ptr;
	double* C = Cmat.ptr;
	std::size_t Abeg = blockA_row * block_size * nA + blockA_col * block_size;
	std::size_t Bbeg = blockB_row * block_size * nB + blockB_col * block_size;
	std::size_t Cbeg = blockC_row * block_size * nC + blockC_col * block_size;

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, block_size, block_size, block_size, 1, &A[Abeg], nA, &B[Bbeg], nB, 0, &C[Cbeg], nC);
}

// perform C = AB + C
void mult_add_block(struct matrix_t Amat, std::size_t blockA_row, std::size_t blockA_col, struct matrix_t Bmat, std::size_t blockB_row,
	std::size_t blockB_col, struct matrix_t Cmat, std::size_t blockC_row, std::size_t blockC_col, std::size_t block_size)
{
	std::size_t nA = Amat.rows;
	std::size_t nB = Bmat.rows;
	std::size_t nC = Cmat.rows;
	double* A = Amat.ptr;
	double* B = Bmat.ptr;
	double* C = Cmat.ptr;
	std::size_t Abeg = blockA_row * block_size * nA + blockA_col * block_size;
	std::size_t Bbeg = blockB_row * block_size * nB + blockB_col * block_size;
	std::size_t Cbeg = blockC_row * block_size * nC + blockC_col * block_size;

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, block_size, block_size, block_size, 1, &A[Abeg], nA, &B[Bbeg], nB, 1, &C[Cbeg], nC);
}