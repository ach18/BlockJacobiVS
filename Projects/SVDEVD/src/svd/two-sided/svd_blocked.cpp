#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include "../../utils/block.hpp"
#include "../../utils/matrix.hpp"
#include "nsvd.hpp"
#include "svd.hpp"
#include "svd_subprocedure.hpp"
#include "../../utils/types.hpp"
#include "../../utils/util.hpp"

/**
 * @param matrix_t Amat ������������ ���������� ������� A
 * @param matrix_t Bmat ������� A ����� ������� �� �����
 * @param matrix_t Umat ������� ����� ����������� ��������
 * @param matrix_t Vmat ������� ������ ����������� ��������
 * @param size_t block_size ������ ����� ������� Bmat
 * @return size_t sweeps ����� ��������� ������� �����
 **/
size_t svd_blocked(struct matrix_t Amat, struct matrix_t Bmat, struct matrix_t Umat, struct matrix_t Vmat,
    size_t block_size) {

    size_t sweeps = 0;  //����� ���������� ����� ���������
    size_t block_iter = 0;
    const double tol = 1e-10;  //�������� ������� ����������
    const size_t n = Amat.rows; //������ �������
    double norm = 0.0;      //����� ���������� ������� B
    double off_norm = 0.0;  //����� ���������� ������ �������������� ��������� ������� B

    matrix_copy(Bmat, Amat); //������������� ������� ������� B
    //������� U � V ���������������� ��� ��������� �������
    matrix_identity(Umat);
    matrix_identity(Vmat);
    matrix_frobenius(Bmat, &norm, &off_norm);

    const size_t n_blocks = n / block_size; //����� ������ ����� ������/�������

    //���� ������/������� ������� �� ���� ������, ����� ��������� SVD ������������ �� ������� �����
    if (n <= 2 * block_size) {
        size_t block_iters = svd_subprocedure(Bmat, Umat, Vmat);
        return block_iters;
    }

    //����� ����� ��������� ���� ������ ������/������� ������ ���� ����� ����������� �������
    assert(n_blocks * block_size == n);

    //��������� ������ ��� �������� ������ ������ B U V M1 M2
    //double* memory_block = (double*)malloc((4 + 4 + 4 + 1 + 1) * block_size * block_size * sizeof(double));
    //double* Bblock = memory_block;
    std::vector<double> Bblock(4 * block_size * block_size);
    //double* Ublock = Bblock + 4 * block_size * block_size;
    std::vector<double> Ublock(4 * block_size * block_size);
    //double* Vblock = Ublock + 4 * block_size * block_size;
    std::vector<double> Vblock(4 * block_size * block_size);
    //M1 M2 ������ ������������� �������� ����������
    //double* M1 = Vblock + 4 * block_size * block_size;
    std::vector<double> M1(block_size * block_size);
    //double* M2 = M1 + block_size * block_size;
    std::vector<double> M2(block_size * block_size);

    //�������� ������ � ���� �������� matrix_t
    matrix_t Bblockmat = { &Bblock[0], 2 * block_size, 2 * block_size };
    matrix_t Ublockmat = { &Ublock[0], 2 * block_size, 2 * block_size };
    matrix_t Vblockmat = { &Vblock[0], 2 * block_size, 2 * block_size };
    matrix_t M1mat = { &M1[0], block_size, block_size };
    matrix_t M2mat = { &M2[0], block_size, block_size };

    //�������� ���� ���������, �� ������������ ���� 
    while (sqrt(off_norm) > tol * sqrt(norm)) {
        //���� ������ ���/��������������� ������
        for (size_t i_block = 0; i_block < n_blocks - 1; ++i_block) {
            for (size_t j_block = i_block + 1; j_block < n_blocks; ++j_block) {
                //� Bblockmat ���������� ����� � ��������� ii ij ji jj �� ������� Bmat
                copy_block(Bmat, i_block, i_block, Bblockmat, 0, 0, block_size);
                copy_block(Bmat, i_block, j_block, Bblockmat, 0, 1, block_size);
                copy_block(Bmat, j_block, i_block, Bblockmat, 1, 0, block_size);
                copy_block(Bmat, j_block, j_block, Bblockmat, 1, 1, block_size);

                //���������� SVD ���������� ����������� ������� ����� ��� ������ Bblockmat
                block_iter += svd_subprocedure(Bblockmat, Ublockmat, Vblockmat);

                //��������������� ���� ������� U 
                matrix_transpose(Ublockmat, Ublockmat);

                //���������� ������ ������� B �� ������� i j � ������� U
                for (size_t k_block = 0; k_block < n_blocks; ++k_block) {
                    mult_block(Ublockmat, 0, 0, Bmat, i_block, k_block, M1mat, 0, 0, block_size);
                    mult_block(Ublockmat, 0, 1, Bmat, j_block, k_block, M2mat, 0, 0, block_size);
                    matrix_add(M1mat, M2mat, M2mat);
                    mult_block(Ublockmat, 1, 0, Bmat, i_block, k_block, M1mat, 0, 0, block_size);
                    copy_block(M2mat, 0, 0, Bmat, i_block, k_block, block_size);
                    mult_block(Ublockmat, 1, 1, Bmat, j_block, k_block, M2mat, 0, 0, block_size);
                    matrix_add(M1mat, M2mat, M2mat);
                    copy_block(M2mat, 0, 0, Bmat, j_block, k_block, block_size);
                }

                //���������� ������ ������� B �� �������� i j � ������� V
                for (size_t k_block = 0; k_block < n_blocks; ++k_block) {
                    mult_block(Bmat, k_block, i_block, Vblockmat, 0, 0, M1mat, 0, 0, block_size);
                    mult_block(Bmat, k_block, j_block, Vblockmat, 1, 0, M2mat, 0, 0, block_size);
                    matrix_add(M1mat, M2mat, M2mat);
                    mult_block(Bmat, k_block, i_block, Vblockmat, 0, 1, M1mat, 0, 0, block_size);
                    copy_block(M2mat, 0, 0, Bmat, k_block, i_block, block_size);
                    mult_block(Bmat, k_block, j_block, Vblockmat, 1, 1, M2mat, 0, 0, block_size);
                    matrix_add(M1mat, M2mat, M2mat);
                    copy_block(M2mat, 0, 0, Bmat, k_block, j_block, block_size);
                }

                //����������� ����� ������� U^T � �������� ��������� U 
                matrix_transpose(Ublockmat, Ublockmat);

                //���������� ������ ������� U �� �������� i j
                for (size_t k_block = 0; k_block < n_blocks; ++k_block) {
                    mult_block(Umat, k_block, i_block, Ublockmat, 0, 0, M1mat, 0, 0, block_size);
                    mult_block(Umat, k_block, j_block, Ublockmat, 1, 0, M2mat, 0, 0, block_size);
                    matrix_add(M1mat, M2mat, M2mat);
                    mult_block(Umat, k_block, i_block, Ublockmat, 0, 1, M1mat, 0, 0, block_size);
                    copy_block(M2mat, 0, 0, Umat, k_block, i_block, block_size);
                    mult_block(Umat, k_block, j_block, Ublockmat, 1, 1, M2mat, 0, 0, block_size);
                    matrix_add(M1mat, M2mat, M2mat);
                    copy_block(M2mat, 0, 0, Umat, k_block, j_block, block_size);
                }

                //���������� ������ ������� V �� �������� i j
                for (size_t k_block = 0; k_block < n_blocks; ++k_block) {
                    mult_block(Vmat, k_block, i_block, Vblockmat, 0, 0, M1mat, 0, 0, block_size);
                    mult_block(Vmat, k_block, j_block, Vblockmat, 1, 0, M2mat, 0, 0, block_size);
                    matrix_add(M1mat, M2mat, M2mat);
                    mult_block(Vmat, k_block, i_block, Vblockmat, 0, 1, M1mat, 0, 0, block_size);
                    copy_block(M2mat, 0, 0, Vmat, k_block, i_block, block_size);
                    mult_block(Vmat, k_block, j_block, Vblockmat, 1, 1, M2mat, 0, 0, block_size);
                    matrix_add(M1mat, M2mat, M2mat);
                    copy_block(M2mat, 0, 0, Vmat, k_block, j_block, block_size);
                }
            }
        }

        matrix_frobenius(Bmat, &norm, &off_norm);
        sweeps++;
    }

    //free(memory_block);

    return sweeps;
}
