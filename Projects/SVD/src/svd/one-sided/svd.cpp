#include "svd.hpp"
#include <cassert>
#include <math.h>
#include <omp.h>
#include <cstdio>
#include "../../utils/types.hpp"
#include "../../utils/util.hpp"
#include "../../utils/matrix.hpp"

/**
* ������������� ����� ����� (Hestenes Jacobi),
  �������� ���������� ��������������� �� ������� �������
 * @param matrix_t Amat ������������� ������� A
 * @param vector_t svec ������ ����������� �����
 * @param matrix_t Umat ������� ����� ����������� ��������
 * @param matrix_t Vmat ������� ������ ����������� ��������
 * @return std::size_t sweeps ����� ��������� ������� �����
 **/
std::size_t coloshjac(struct matrix_t Amat, struct vector_t svec, struct matrix_t Umat, struct matrix_t Vmat, std::size_t ThreadsNum, double* Time, struct string_t errors) {
    const std::size_t m = Amat.rows; //������ ������� A
    const std::size_t n = Amat.cols; //������� ������� A
    const std::size_t n_singular_vals = svec.len; //����� ������� ����������� ��������
    const double tol = 10e-15;  //�������� ������� ����������
	std::size_t max_sweeps = 40;
    double t1, t2;              //����� �������
    std::size_t sweeps = 0;  //����� ���������� ����� ���������
    double norm = 0.0;      //����� ���������� ������� A
    double off_norm = 0.0;  //����� ���������� ������ �������������� ��������� ������� A
    bool converged = true; //������� ���������� �����
    
    matrix_copy(Umat, Amat); //������������� ������� ������� U ��� �������� ������� A
    matrix_identity(Vmat);
    matrix_frobenius(Amat, &norm, &off_norm);
    norm *= tol;

    double* A = Amat.ptr;
    double* s = svec.ptr;
    double* U = Umat.ptr;
    double* V = Vmat.ptr;

	if(ThreadsNum > 1)
		omp_set_num_threads(ThreadsNum);

    t1 = omp_get_wtime();
    do
    {
        converged = true;
        // ������� ���-������������ ��������� ������� U
    #pragma omp parallel for shared(converged, norm, U, V) firstprivate(n, m) schedule(dynamic) if(ThreadsNum > 1)
        for (std::size_t i = 0; i < n - 1; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                double dot_ii = 0, dot_jj = 0, dot_ij = 0;
                //���������� ����� ��������� ��������� ��������, � ������� �����������
                //Uii, Uij, Ujj
                for (std::size_t k = 0; k < m; ++k) {
                    dot_ii += U[n * k + i] * U[n * k + i];
                    dot_ij += U[n * k + i] * U[n * k + j];
                    dot_jj += U[n * k + j] * U[n * k + j];
                }

                if (abs(dot_ij) > norm)
                    converged = false;

                double cosine, sine;
                //���������� cos, sin ������� ��������
                sym_jacobi_coeffs(dot_ii, dot_ij, dot_jj, &cosine, &sine);

                //���������� ��������� ������ ����� ����������� �������� �� ��������
                for (std::size_t k = 0; k < m; ++k) {
                    double left = cosine * U[n * k + i] - sine * U[n * k + j];
                    double right = sine * U[n * k + i] + cosine * U[n * k + j];
                    U[n * k + i] = left;
                    U[n * k + j] = right;
                }
                //���������� ��������� ������ ������ ����������� �������� �� ��������
                for (std::size_t k = 0; k < n; ++k) {
                    double left = cosine * V[n * k + i] - sine * V[n * k + j];
                    double right = sine * V[n * k + i] + cosine * V[n * k + j];
                    V[n * k + i] = left;
                    V[n * k + j] = right;
                }
            }
        }

        if (sweeps > max_sweeps)
            converged = true;
		else
			sweeps++;
    } 
    while (!converged);

    if (sweeps > max_sweeps)
        return 0;

    #pragma omp parallel for shared(s, U, n_singular_vals) firstprivate(m, n) schedule(dynamic) if(ThreadsNum > 1)
    for (std::size_t i = 0; i < n; ++i) {
        double sigma = 0.0;
        //���������� ����������� �����
        for (std::size_t k = 0; k < m; ++k) {
            sigma += U[n * k + i] * U[n * k + i];
        }
        sigma = sqrt(sigma);

        if (i < n_singular_vals) {
            s[i] = sigma;
        }
        //������������� ����� ����������� ��������
        for (std::size_t k = 0; k < m; ++k) {
            U[n * k + i] /= sigma;
        }
    }

    matrix_t matrices[2] = { Umat, Vmat };
    //���������� ����� � �������� �� �������� ��������
    reorder_decomposition(svec, matrices, 2, greater);

    t2 = omp_get_wtime();
    *Time = t2 - t1;
    return sweeps;
}

/**
 * ������� ������������� ����� ����� (Block Jacobi Relaxasion),
 * ����� ���������� � ������������ �� ���������� ���������� ������� (Round Robin)
 * @param matrix_t Amat ������������� ������� A
 * @param vector_t svec ������ ����������� �����
 * @param matrix_t Umat ������� ����� ����������� ��������
 * @param matrix_t Vmat ������� ������ ����������� ��������
 * @return std::size_t sweeps ����� ��������� ������� �����
 **/
std::size_t rrbjrs(struct matrix_t Amat, struct matrix_t Bmat, struct vector_t svec, struct matrix_t Umat, struct matrix_t Vmat, std::size_t ThreadsNum, double* Time, struct string_t errors) {
    const std::size_t m = Amat.rows; //������ ������� A
    const std::size_t n = Amat.cols; //������� ������� A
    const std::size_t n_singular_vals = svec.len; //����� ������� ����������� ��������
    const double tol = 10e-15;  //�������� ������� ����������
	std::size_t max_sweeps = 40; //������������ ����� ���������� ��������
    double t1, t2;              //����� �������
    std::size_t sweeps = 0;  //����� ���������� ����� ���������
    double norm = 0.0;      //����� ���������� ������� A
    double off_norm = 0.0;  //����� ���������� ������ �������������� ��������� ������� A
    bool converged = true; //������� ���������� �����
	//double lambda = 0.0; //����������� ���������� (������ � ���

    std::vector<std::size_t> up(ThreadsNum); //������ ������ ����� ������� ����� ��� ������ ������
    std::vector<std::size_t> dn(ThreadsNum); //������ ������ ����� ������� ����� ��� ������ ������
    std::vector<index_t> SOB(2 * ThreadsNum); //������ ������ ���� �������� ������ � ����� ���� ������ (i - ������ ������ �����, j - �����) ��� ������ ������

    //�������� ������� �� ����� (�� ��������)
    bool result = rrbjrs_column_limits(Amat, ThreadsNum, &SOB[0]);
	if (!result) {
		*errors.len = sprintf(errors.ptr, "matrix must be correctly divided into blocks");
		return 0;
	}

    matrix_copy(Bmat, Amat); //������������� ������� ������� U ��� �������� ������� A
    matrix_identity(Vmat); //������� V ��� ��������� �������
    matrix_frobenius(Amat, &norm, &off_norm); //���������� ���� �������, ���������� �� tol - �������� ����� ����������
    norm *= tol;
    off_norm *= tol;

    //������������� ������� ��� �������� �������� 
    //��������� (up[]) � ������� (dn[]) �������
    for (std::size_t i = 0; i < ThreadsNum; i++) {
        up[i] = (2 * i);
        dn[i] = (2 * i) + 1;
    }

    double* B = Bmat.ptr;
    double* s = svec.ptr;
    double* U = Umat.ptr;
    double* V = Vmat.ptr;

	if(ThreadsNum > 1)
	    omp_set_num_threads(ThreadsNum);

    t1 = omp_get_wtime();

    //���� ���� converged true
    do
    {
        converged = true;
        //���� ��������������� ������ �������
    #pragma omp parallel for shared(converged, U, V, B, SOB, m, n, norm) firstprivate(ThreadsNum) schedule(static,2) if(ThreadsNum > 1)
        for (std::size_t rr_pair = 0; rr_pair < (2 * ThreadsNum); rr_pair++) {
            std::size_t ind = 0;
			std::vector<double> cos(m * n);
			std::vector<double> sin(m * n);
            for (std::size_t i = SOB[rr_pair].i; i <= (SOB[rr_pair].j - 1); i++) {
                for (std::size_t j = (i + 1); j <= (SOB[rr_pair].j); j++) {
                    double dot_ii = 0, dot_jj = 0, dot_ij = 0;
                    //���������� ����� ��������� ��������� ��������, � ������� �����������
                    //Uii, Uij, Ujj
                    for (std::size_t k = 0; k < m; k++) {
                        dot_ii += B[n * k + i] * B[n * k + i];
                        dot_ij += B[n * k + i] * B[n * k + j];
                        dot_jj += B[n * k + j] * B[n * k + j];
                    }

                    if (abs(dot_ij) > norm)
                        converged = false;

                    double cosine, sine;
                    //���������� cos, sin ������� ��������
                    jrs_jacobi_coeffs(dot_ii, dot_ij, dot_jj, &cosine, &sine);
                    cos[ind] = cosine;
                    sin[ind] = sine;
                    ind++;
                }
            }
            ind = 0;
            for (std::size_t i = SOB[rr_pair].i; i <= (SOB[rr_pair].j - 1); i++) {
                for (std::size_t j = (i + 1); j <= (SOB[rr_pair].j); j++) {
                    //���������� ��������� ������ ����� ����������� �������� �� ��������
                    for (std::size_t k = 0; k < m; k++) {
                        double left = cos[ind] * B[n * k + i] - sin[ind] * B[n * k + j];
                        double right = sin[ind] * B[n * k + i] + cos[ind] * B[n * k + j];
                        B[n * k + i] = left;
                        B[n * k + j] = right;
                    }
                    //���������� ��������� ������ ������ ����������� �������� �� ��������
                    for (std::size_t k = 0; k < n; k++) {
                        double left = cos[ind] * V[n * k + i] - sin[ind] * V[n * k + j];
                        double right = sin[ind] * V[n * k + i] + cos[ind] * V[n * k + j];
                        V[n * k + i] = left;
                        V[n * k + j] = right;
                    }
                    ind++;
                }
            }
        }

        //�������� ���� ������� ������������ ����������
        //�� ������ ��������, � ������� ��������� ������������� ������ ����� �������� (Round Robin)
        //������ ����� �������� ���� ���� ������ (�� 2 ����� �� �����)
        for (std::size_t iteration = 0; iteration < ((2 * ThreadsNum) - 1); iteration++) {
    #pragma omp parallel for shared(converged, U, V, B, SOB, up, dn, m, n, norm) firstprivate(ThreadsNum) schedule(static,1) if(ThreadsNum > 1)
            for (std::size_t rr_pair = 0; rr_pair < ThreadsNum; rr_pair++) {
                std::size_t ind = 0;
				std::vector<double> cos(m * n);
				std::vector<double> sin(m * n);
                for (std::size_t i = SOB[up[rr_pair]].i; i <= SOB[up[rr_pair]].j; i++) {
                    for (std::size_t j = SOB[dn[rr_pair]].i; j <= SOB[dn[rr_pair]].j; j++) {
                        double dot_ii = 0, dot_jj = 0, dot_ij = 0;
                        //���������� ����� ��������� ��������� �������, � ������� �����������
                        //Uii, Uij, Ujj
                        for (std::size_t k = 0; k < m; k++) {
                            dot_ii += B[n * k + i] * B[n * k + i];
                            dot_ij += B[n * k + i] * B[n * k + j];
                            dot_jj += B[n * k + j] * B[n * k + j];
                        }

                        if (abs(dot_ij) > norm)
                            converged = false;

                        double cosine, sine;
                        //���������� cos, sin ������� ��������
                        jrs_jacobi_coeffs(dot_ii, dot_ij, dot_jj, &cosine, &sine);
                        cos[ind] = cosine;
                        sin[ind] = sine;
                        ind++;
                    }
                }

                ind = 0;
                for (std::size_t i = SOB[up[rr_pair]].i; i <= SOB[up[rr_pair]].j; i++) {
                    for (std::size_t j = SOB[dn[rr_pair]].i; j <= SOB[dn[rr_pair]].j; j++) {
                        //���������� ��������� ������ ����� ����������� �������� �� ��������
                        for (std::size_t k = 0; k < m; k++) {
                            double left = cos[ind] * B[n * k + i] - sin[ind] * B[n * k + j];
                            double right = sin[ind] * B[n * k + i] + cos[ind] * B[n * k + j];
                            B[n * k + i] = left;
                            B[n * k + j] = right;
                        }
                        //���������� ��������� ������ ������ ����������� �������� �� ��������
                        for (std::size_t k = 0; k < n; k++) {
                            double left = cos[ind] * V[n * k + i] - sin[ind] * V[n * k + j];
                            double right = sin[ind] * V[n * k + i] + cos[ind] * V[n * k + j];
                            V[n * k + i] = left;
                            V[n * k + j] = right;
                        }
                        ind++;
                    }
                }
            }
            //��������� ������������� ������ ����� ��������(Round Robin) ��������� ����� ���� ������ ��� ������� ������
            round_robin(&up[0], &dn[0], ThreadsNum); 
        }
        if (sweeps > max_sweeps)
            converged = true;
		else
			sweeps++; //����� ���������� ��������� �����
    } while (!converged);

	if (sweeps > max_sweeps) {
		*errors.len = sprintf(errors.ptr, "algorithm did not converge after %lu sweeps", sweeps);
		return 0;
	}

    #pragma omp parallel for shared(s, B, n) schedule(guided) if(ThreadsNum > 1)
    for (std::size_t i = 0; i < n; i++) {
        double sigma = 0.0;
        //���������� ����������� �����
        for (std::size_t k = 0; k < m; k++) {
            sigma += B[n * k + i] * B[n * k + i];
        }
        sigma = sqrt(sigma);
        s[i] = sigma;

        //������������� ����� ����������� ��������
        for (std::size_t k = 0; k < m; k++) {
            B[n * k + i] /= sigma;
        }
    }

	t2 = omp_get_wtime();
	*Time = t2 - t1;

    matrix_t matrices[2] = { Umat, Vmat };
    //���������� ����� � �������� �� �������� ��������
    //reorder_decomposition(svec, matrices, 2, greater);

    return sweeps;
}

