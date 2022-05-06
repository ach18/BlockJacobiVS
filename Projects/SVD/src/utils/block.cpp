#include "block.hpp"

void copy_subproblem_blocks(struct matrix_t Smat, std::size_t blockS_row, std::size_t blockS_col, struct matrix_t Dmat, std::size_t blockD_row,
	std::size_t blockD_col, std::size_t block_size) 
{
	std::size_t nS = Smat.rows;
	std::size_t nD = Dmat.rows;
	double* S = Smat.ptr;
	double* D = Dmat.ptr;
	std::size_t Sbeg0, Sbeg1, Sbeg2, Sbeg3, Dbeg0, Dbeg1, Dbeg2, Dbeg3;

	if (Smat.storage == 'R') {
		Sbeg0 = blockS_row * block_size * nS + blockS_row * block_size;
		Dbeg0 = blockD_row * block_size * nD + blockD_row * block_size;
		Sbeg1 = blockS_row * block_size * nS + blockS_col * block_size;
		Dbeg1 = blockD_row * block_size * nD + blockD_col * block_size;
		Sbeg2 = blockS_col * block_size * nS + blockS_row * block_size;
		Dbeg2 = blockD_col * block_size * nD + blockD_row * block_size;
		Sbeg3 = blockS_col * block_size * nS + blockS_col * block_size;
		Dbeg3 = blockD_col * block_size * nD + blockD_col * block_size;
	}
	else {
		Sbeg0 = blockS_row * block_size + blockS_row * block_size * nS;
		Dbeg0 = blockD_row * block_size + blockD_row * block_size * nD;
		Sbeg1 = blockS_row * block_size + blockS_col * block_size * nS;
		Dbeg1 = blockD_row * block_size + blockD_col * block_size * nD;
		Sbeg2 = blockS_col * block_size + blockS_row * block_size * nS;
		Dbeg2 = blockD_col * block_size + blockD_row * block_size * nD;
		Sbeg3 = blockS_col * block_size + blockS_col * block_size * nS;
		Dbeg3 = blockD_col * block_size + blockD_col * block_size * nD;
	}

	if (Smat.storage == 'R') {
		for (std::size_t i = 0; i < block_size; ++i) {
			cblas_dcopy(block_size, &S[Sbeg0 + (i * nS)], 1, &D[Dbeg0 + (i * nD)], 1);
			cblas_dcopy(block_size, &S[Sbeg1 + (i * nS)], 1, &D[Dbeg1 + (i * nD)], 1);
			cblas_dcopy(block_size, &S[Sbeg2 + (i * nS)], 1, &D[Dbeg2 + (i * nD)], 1);
			cblas_dcopy(block_size, &S[Sbeg3 + (i * nS)], 1, &D[Dbeg3 + (i * nD)], 1);
		}
	}
	else {
		std::size_t j = 0;
		for (std::size_t i = 0; i < block_size; ++i) {
			for (j = 0; j < block_size; j++) {
				D[Dbeg0 + j * nD + i] = S[Sbeg0 + j * nS + i];
				D[Dbeg1 + j * nD + i] = S[Sbeg1 + j * nS + i];
				D[Dbeg2 + j * nD + i] = S[Sbeg2 + j * nS + i];
				D[Dbeg3 + j * nD + i] = S[Sbeg3 + j * nS + i];
			}
		}
	}
}

void copy_block(struct matrix_t Smat, std::size_t blockS_row, std::size_t blockS_col, struct matrix_t Dmat, std::size_t blockD_row,
	std::size_t blockD_col, std::size_t block_size)
{
	std::size_t nS = Smat.rows;
	std::size_t nD = Dmat.rows;
	double* S = Smat.ptr;
	double* D = Dmat.ptr;

	std::size_t Sbeg, Dbeg; 

	if (Smat.storage == 'R') {
		Sbeg = blockS_row * block_size * nS + blockS_col * block_size;
		Dbeg = blockD_row * block_size * nD + blockD_col * block_size;
	}
	else {
		Sbeg = blockS_row * block_size + blockS_col * block_size * nS;
		Dbeg = blockD_row * block_size + blockD_col * block_size * nD;
	}
	std::size_t j = 0;
	if (Smat.storage == 'R') {
		for (std::size_t i = 0; i < block_size; ++i) {
			for (j = 0; j < block_size; j++) {
				D[Dbeg + i * nD + j] = S[Sbeg + i * nS + j];
			}
		}
	}
	else {
		for (std::size_t i = 0; i < block_size; ++i) {
			for (j = 0; j < block_size; j++) {
				D[Dbeg + j * nD + i] = S[Sbeg + j * nS + i];
			}
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
	std::size_t Abeg, Bbeg, Cbeg;
	if (Amat.storage == 'R') {
		Abeg = blockA_row * block_size * nA + blockA_col * block_size;
		Bbeg = blockB_row * block_size * nB + blockB_col * block_size;
		Cbeg = blockC_row * block_size * nC + blockC_col * block_size;
	}
	else {
		Abeg = blockA_row * block_size + blockA_col * block_size * nA;
		Bbeg = blockB_row * block_size + blockB_col * block_size * nB;
		Cbeg = blockC_row * block_size + blockC_col * block_size * nC;
	}

	if (Amat.storage == 'R')
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, block_size, block_size, block_size, 1, &A[Abeg], nA, &B[Bbeg], nB, 0, &C[Cbeg], nC);
	else
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, block_size, block_size, block_size, 1, &A[Abeg], nA, &B[Bbeg], nB, 0, &C[Cbeg], nC);
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
	std::size_t Abeg, Bbeg, Cbeg;
	if (Amat.storage == 'R') {
		Abeg = blockA_row * block_size * nA + blockA_col * block_size;
		Bbeg = blockB_row * block_size * nB + blockB_col * block_size;
		Cbeg = blockC_row * block_size * nC + blockC_col * block_size;
	}
	else {
		Abeg = blockA_row * block_size + blockA_col * block_size * nA;
		Bbeg = blockB_row * block_size + blockB_col * block_size * nB;
		Cbeg = blockC_row * block_size + blockC_col * block_size * nC;
	}
	
	if (Amat.storage == 'R')
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, block_size, block_size, block_size, 1, &A[Abeg], nA, &B[Bbeg], nB, 1, &C[Cbeg], nC);
	else
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, block_size, block_size, block_size, 1, &A[Abeg], nA, &B[Bbeg], nB, 1, &C[Cbeg], nC);
}