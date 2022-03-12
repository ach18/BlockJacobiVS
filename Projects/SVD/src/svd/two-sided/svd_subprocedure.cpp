#include "svd_subprocedure.hpp"
#include <math.h>
#include "../../utils/matrix.hpp"
#include "nsvd.hpp"

size_t svd_subprocedure(struct matrix_t Bmat, struct matrix_t Umat, struct matrix_t Vmat) {
    size_t iter = 0;  //����� ���������� ����� ���������
    size_t n = Bmat.rows; //������ �������
    const double tol = 1e-15; //�������� ������� ����������
    double* B = Bmat.ptr;
    double* U = Umat.ptr;
    double* V = Vmat.ptr;
    double norm = 0.0;      //����� ���������� ������� B
    double off_norm = 0.0;  //����� ���������� ������ �������������� ��������� ������� B

    //������� U � V ���������������� ��� ��������� �������
    matrix_identity(Umat);
    matrix_identity(Vmat);
    //���������� ����� ����������, � ����� �������������� ���������
    matrix_frobenius(Bmat, &norm, &off_norm);

    while (sqrt(off_norm) > tol * sqrt(norm)) {
        for (size_t i = 0; i < n - 1; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                double bii = B[n * i + i];
                double bij = B[n * i + j];
                double bji = B[n * j + i];
                double bjj = B[n * j + j];

                //��������� ������������ cos sin ��� �������� bij
                //������� NSVD - https://maths-people.anu.edu.au/~brent/pd/rpb080i.pdf (���. 12)
                struct svd_2x2_params cf = nsvd(bii, bij, bji, bjj);

                //���������� �������� J^T*B*J, �.�. ���������� ����� � ��������
                //J - ������� ������� �������� (������� �������) � cos sin �� ������ i j
                //������ ���� ��������� ������ i j
                for (size_t k = 0; k < n; k++) {
                    double b_ik = B[n * i + k];
                    double b_jk = B[n * j + k];

                    double left = cf.c1 * b_ik - cf.s1 * b_jk;
                    double right = cf.s1 * cf.k * b_ik + cf.c1 * cf.k * b_jk;

                    B[n * i + k] = left;
                    B[n * j + k] = right;
                }

                //������ ���� ��������� ������� i j
                for (size_t k = 0; k < n; k++) {
                    double b_ki = B[n * k + i];
                    double b_kj = B[n * k + j];

                    double left = cf.c2 * b_ki - cf.s2 * b_kj;
                    double right = cf.s2 * b_ki + cf.c2 * b_kj;

                    B[n * k + i] = left;
                    B[n * k + j] = right;
                }

                //��������� ��� ����� ��������� ������� i j � ������ U V 
                for (size_t k = 0; k < n; k++) {
                    double u_ki = U[n * k + i];
                    double u_kj = U[n * k + j];

                    double left = cf.c1 * u_ki - cf.s1 * u_kj;
                    double right = cf.s1 * cf.k * u_ki + cf.c1 * cf.k * u_kj;

                    U[n * k + i] = left;
                    U[n * k + j] = right;
                }

                for (size_t k = 0; k < n; k++) {
                    double v_ki = V[n * k + i];
                    double v_kj = V[n * k + j];

                    double left = cf.c2 * v_ki - cf.s2 * v_kj;
                    double right = cf.s2 * v_ki + cf.c2 * v_kj;

                    V[n * k + i] = left;
                    V[n * k + j] = right;
                }
            }
        }

        matrix_frobenius(Bmat, &norm, &off_norm);
        iter += 1;
    }

    return iter;
}
