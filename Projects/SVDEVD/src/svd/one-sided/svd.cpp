#include "svd.hpp"
#include <assert.h>
#include <math.h>
#include <omp.h>
#include "../../utils/types.hpp"
#include "../../utils/util.hpp"
#include "../../utils/matrix.hpp"

/**
 * @param matrix_t Amat ������������ ���������� (��� �������������) ������� A
 * @param vector_t svec ������ ����������� �����
 * @param matrix_t Umat ������� ����� ����������� ��������
 * @param matrix_t Vmat ������� ������ ����������� ��������
 * @param size_t n_iter ����� ����������� ��� ���������� ���������
 * @return size_t sweeps ����� ��������� ������� �����
 **/
size_t sohjac(struct matrix_t Amat, struct vector_t svec, struct matrix_t Umat, struct matrix_t Vmat) {
    const size_t m = Amat.rows; //������ ������� A
    const size_t n = Amat.cols; //������� ������� A
    const size_t n_singular_vals = svec.len; //����� ������� ����������� ��������
    const double tol = 10e-15;  //�������� ������� ����������
    size_t sweeps = 0;  //����� ���������� ����� ���������
    double norm = 0.0;      //����� ���������� ������� A
    double off_norm = 0.0;  //����� ���������� ������ �������������� ��������� ������� A
    bool converged = true; //������� ���������� �����
    
    matrix_copy(Umat, Amat); //������������� ������� ������� U ��� �������� ������� A
    matrix_identity(Vmat);
    matrix_frobenius(Amat, &norm, &off_norm);
    norm *= tol;

    assert(m > 0);
    assert(n > 0);
    assert(n_singular_vals > 0);
    assert(((m < n) ? m : n) == n_singular_vals);
    assert(m == Umat.rows);
    assert(n == Umat.cols);
    assert(n == Vmat.rows);
    assert(n == Vmat.cols);

    double* A = Amat.ptr;
    double* s = svec.ptr;
    double* U = Umat.ptr;
    double* V = Vmat.ptr;

    //���� ��������� ����������� �� ���������� ����� ��������
    //for (size_t iter = 0; iter < n_iter; ++iter) {
    do
    {
        converged = true;
        // ������� ���-������������ ��������� ������� U
        for (size_t i = 0; i < n - 1; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                double dot_ii = 0, dot_jj = 0, dot_ij = 0;
                //���������� ����� ��������� ��������� �������, � ������� �����������
                //Uii, Uij, Ujj
                for (size_t k = 0; k < m; ++k) {
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
                for (size_t k = 0; k < m; ++k) {
                    double left = cosine * U[n * k + i] - sine * U[n * k + j];
                    double right = sine * U[n * k + i] + cosine * U[n * k + j];
                    U[n * k + i] = left;
                    U[n * k + j] = right;
                }
                //���������� ��������� ������ ������ ����������� �������� �� ��������
                for (size_t k = 0; k < n; ++k) {
                    double left = cosine * V[n * k + i] - sine * V[n * k + j];
                    double right = sine * V[n * k + i] + cosine * V[n * k + j];
                    V[n * k + i] = left;
                    V[n * k + j] = right;
                }
            }
        }
        sweeps++;
    } 
    while (!converged);

    for (size_t i = 0; i < n; ++i) {
        double sigma = 0.0;
        //���������� ����������� �����
        for (size_t k = 0; k < m; ++k) {
            sigma += U[n * k + i] * U[n * k + i];
        }
        sigma = sqrt(sigma);

        if (i < n_singular_vals) {
            s[i] = sigma;
        }
        //������������� ����� ����������� ��������
        for (size_t k = 0; k < m; ++k) {
            U[n * k + i] /= sigma;
        }
    }

    matrix_t matrices[2] = { Umat, Vmat };
    //���������� ����� � �������� �� �������� ��������
    reorder_decomposition(svec, matrices, 2, greater);

    return sweeps;
}

/**
 * @param matrix_t Amat ������������ ���������� (��� �������������) ������� A
 * @param vector_t svec ������ ����������� �����
 * @param matrix_t Umat ������� ����� ����������� ��������
 * @param matrix_t Vmat ������� ������ ����������� ��������
 * @param size_t n_iter ����� ����������� ��� ���������� ���������
 * @return size_t sweeps ����� ��������� ������� �����
 **/
size_t pbjrs(struct matrix_t Amat, struct vector_t svec, struct matrix_t Umat, struct matrix_t Vmat, size_t ThreadsNum, double* Time) {
    const size_t m = Amat.rows; //������ ������� A
    const size_t n = Amat.cols; //������� ������� A
    const size_t n_singular_vals = svec.len; //����� ������� ����������� ��������
    const double tol = 10e-15;  //�������� ������� ����������
    size_t sweeps = 0;  //����� ���������� ����� ���������
    double norm = 0.0;      //����� ���������� ������� A
    double off_norm = 0.0;  //����� ���������� ������ �������������� ��������� ������� A
    bool converged = true; //������� ���������� �����

    //size_t* up = (size_t*)malloc(ThreadsNum *sizeof(size_t)); //������ ������� ��� �������� �������� 
    std::vector<size_t> up(ThreadsNum);
    //size_t* dn = (size_t*)malloc(ThreadsNum * sizeof(size_t)); //������ ������ ��� �������� ��������
    std::vector<size_t> dn(ThreadsNum);

    //struct index_t* SOB = (index_t*)malloc((2 * ThreadsNum) * sizeof(struct index_t)); //������ �������� ������ ������
    std::vector<index_t> SOB(2 * ThreadsNum);

    //double* cos = (double*)malloc((2 * ThreadsNum) * sizeof(double)); //������ ���������� �������� cos
    std::vector<double> cos(2 * ThreadsNum);
    //double* sin = (double*)malloc((2 * ThreadsNum) * sizeof(double)); //������ ���������� �������� sin
    std::vector<double> sin(2 * ThreadsNum);

    bool result = column_limits(Amat, ThreadsNum, &SOB[0]);
    if (!result)
        return 0;

    matrix_copy(Umat, Amat); //������������� ������� ������� U ��� �������� ������� A
    matrix_identity(Vmat);
    matrix_frobenius(Amat, &norm, &off_norm);
    norm *= tol;
    off_norm *= tol;

    //������������� ������� ��� �������� �������� 
    // 
    //��������� (up[]) � ������� (dn[]) �������
    for (size_t i = 0; i < ThreadsNum; i++) {
        up[i] = (2 * i);
        dn[i] = (2 * i) + 1;
    }

    assert(m > 0);
    assert(n > 0);
    assert(n_singular_vals > 0);
    assert(((m < n) ? m : n) == n_singular_vals);
    assert(m == Umat.rows);
    assert(n == Umat.cols);
    assert(n == Vmat.rows);
    assert(n == Vmat.cols);

    double* A = Amat.ptr;
    double* s = svec.ptr;
    double* U = Umat.ptr;
    double* V = Vmat.ptr;

    omp_set_num_threads(ThreadsNum);

    do
    {
    //for (size_t sweep = 0; sweep < 10; sweep++) {
        converged = true;
        #pragma omp parallel for shared(converged, U, V, SOB) private(s, ThreadsNum) schedule(static, 2)
        for (size_t s = 0; s < (2 * ThreadsNum); s++) {
            size_t ind = 0;
            for (size_t i = SOB[s].i; i <= (SOB[s].j - 1); i++) {
                for (size_t j = (i + 1); j <= (SOB[s].j); j++) {
                    double dot_ii = 0, dot_jj = 0, dot_ij = 0;
                    //���������� ����� ��������� ��������� �������, � ������� �����������
                    //Uii, Uij, Ujj
                    for (size_t k = 0; k < m; k++) {
                        dot_ii += U[n * k + i] * U[n * k + i];
                        dot_ij += U[n * k + i] * U[n * k + j];
                        dot_jj += U[n * k + j] * U[n * k + j];
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
            for (size_t i = SOB[s].i; i <= (SOB[s].j - 1); i++) {
                for (size_t j = (i + 1); j <= (SOB[s].j); j++) {
                    //���������� ��������� ������ ����� ����������� �������� �� ��������
                    for (size_t k = 0; k < m; k++) {
                        double left = cos[ind] * U[n * k + i] - sin[ind] * U[n * k + j];
                        double right = sin[ind] * U[n * k + i] + cos[ind] * U[n * k + j];
                        U[n * k + i] = left;
                        U[n * k + j] = right;
                    }
                    //���������� ��������� ������ ������ ����������� �������� �� ��������
                    for (size_t k = 0; k < n; k++) {
                        double left = cos[ind] * V[n * k + i] - sin[ind] * V[n * k + j];
                        double right = sin[ind] * V[n * k + i] + cos[ind] * V[n * k + j];
                        V[n * k + i] = left;
                        V[n * k + j] = right;
                    }
                    ind++;
                }
            }
        }

        for (size_t iteration = 0; iteration < ((2 * ThreadsNum) - 1); iteration++) {
            #pragma omp parallel for shared(converged, U, V, SOB, up, dn) private(s, ThreadsNum)
            for (size_t s = 0; s < ThreadsNum; s++) {
                size_t ind = 0;
                for (size_t i = SOB[up[s]].i; i <= SOB[up[s]].j; i++) {
                    for (size_t j = SOB[dn[s]].i; j <= SOB[dn[s]].j; j++) {
                        double dot_ii = 0, dot_jj = 0, dot_ij = 0;
                        //���������� ����� ��������� ��������� �������, � ������� �����������
                        //Uii, Uij, Ujj
                        for (size_t k = 0; k < m; k++) {
                            dot_ii += U[n * k + i] * U[n * k + i];
                            dot_ij += U[n * k + i] * U[n * k + j];
                            dot_jj += U[n * k + j] * U[n * k + j];
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
                for (size_t i = SOB[up[s]].i; i <= SOB[up[s]].j; i++) {
                    for (size_t j = SOB[dn[s]].i; j <= SOB[dn[s]].j; j++) {
                        //���������� ��������� ������ ����� ����������� �������� �� ��������
                        for (size_t k = 0; k < m; k++) {
                            double left = cos[ind] * U[n * k + i] - sin[ind] * U[n * k + j];
                            double right = sin[ind] * U[n * k + i] + cos[ind] * U[n * k + j];
                            U[n * k + i] = left;
                            U[n * k + j] = right;
                        }
                        //���������� ��������� ������ ������ ����������� �������� �� ��������
                        for (size_t k = 0; k < n; k++) {
                            double left = cos[ind] * V[n * k + i] - sin[ind] * V[n * k + j];
                            double right = sin[ind] * V[n * k + i] + cos[ind] * V[n * k + j];
                            V[n * k + i] = left;
                            V[n * k + j] = right;
                        }
                        ind++;
                    }
                }
            }
            round_robin(&up[0], &dn[0], ThreadsNum);
        }
        sweeps++;
    //}
    } while (!converged);

#pragma omp parallel for shared(s) private(i, n) schedule(static)
    for (size_t i = 0; i < n; i++) {
        double sigma = 0.0;
        //���������� ����������� �����
        for (size_t k = 0; k < m; k++) {
            sigma += U[n * k + i] * U[n * k + i];
        }
        sigma = sqrt(sigma);
        s[i] = sigma;

        //if (i < n_singular_vals) {
        //    s[i] = sigma;
        //}
        //������������� ����� ����������� ��������
        for (size_t k = 0; k < m; k++) {
            U[n * k + i] /= sigma;
        }
    }
    matrix_t matrices[2] = { Umat, Vmat };
    //���������� ����� � �������� �� �������� ��������
    reorder_decomposition(svec, matrices, 2, greater);

    return sweeps;
}

size_t sbjrs(struct matrix_t Amat, struct vector_t svec, struct matrix_t Umat, struct matrix_t Vmat, size_t ThreadsNum, double* Time) {
    const size_t m = Amat.rows; //������ ������� A
    const size_t n = Amat.cols; //������� ������� A
    const size_t n_singular_vals = svec.len; //����� ������� ����������� ��������
    const double tol = 10e-15;  //�������� ������� ����������
    size_t sweeps = 0;  //����� ���������� ����� ���������
    double norm = 0.0;      //����� ���������� ������� A
    double off_norm = 0.0;  //����� ���������� ������ �������������� ��������� ������� A
    bool converged = true; //������� ���������� �����

    size_t* up = (size_t*)malloc(ThreadsNum * sizeof(size_t)); //������ ������� ��� �������� �������� 
    size_t* dn = (size_t*)malloc(ThreadsNum * sizeof(size_t)); //������ ������ ��� �������� ��������

    struct index_t* SOB = (index_t*)malloc((2 * ThreadsNum) * sizeof(struct index_t)); //������ �������� ������ ������
    double* cos = (double*)malloc((2 * ThreadsNum) * sizeof(double)); //������ ���������� �������� cos
    double* sin = (double*)malloc((2 * ThreadsNum) * sizeof(double)); //������ ���������� �������� sin

    bool result = column_limits(Amat, ThreadsNum, SOB);
    if (!result)
        return 0;

    matrix_copy(Umat, Amat); //������������� ������� ������� U ��� �������� ������� A
    matrix_identity(Vmat);
    matrix_frobenius(Amat, &norm, &off_norm);
    norm *= tol;
    off_norm *= tol;

    //������������� ������� ��� �������� �������� 
    // 
    //��������� (up[]) � ������� (dn[]) �������
    for (size_t i = 0; i < ThreadsNum; i++) {
        up[i] = (2 * i);
        dn[i] = (2 * i) + 1;
    }

    assert(m > 0);
    assert(n > 0);
    assert(n_singular_vals > 0);
    assert(((m < n) ? m : n) == n_singular_vals);
    assert(m == Umat.rows);
    assert(n == Umat.cols);
    assert(n == Vmat.rows);
    assert(n == Vmat.cols);

    double* A = Amat.ptr;
    double* s = svec.ptr;
    double* U = Umat.ptr;
    double* V = Vmat.ptr;

    omp_set_num_threads(ThreadsNum);

    do
    {
        //for (size_t sweep = 0; sweep < 10; sweep++) {
        converged = true;
        for (size_t s = 0; s < (2 * ThreadsNum); s++) {
            size_t ind = 0;
            for (size_t i = SOB[s].i; i <= (SOB[s].j - 1); i++) {
                for (size_t j = (i + 1); j <= (SOB[s].j); j++) {
                    double dot_ii = 0, dot_jj = 0, dot_ij = 0;
                    //���������� ����� ��������� ��������� �������, � ������� �����������
                    //Uii, Uij, Ujj
                    for (size_t k = 0; k < m; k++) {
                        dot_ii += U[n * k + i] * U[n * k + i];
                        dot_ij += U[n * k + i] * U[n * k + j];
                        dot_jj += U[n * k + j] * U[n * k + j];
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
            for (size_t i = SOB[s].i; i <= (SOB[s].j - 1); i++) {
                for (size_t j = (i + 1); j <= (SOB[s].j); j++) {
                    //���������� ��������� ������ ����� ����������� �������� �� ��������
                    for (size_t k = 0; k < m; k++) {
                        double left = cos[ind] * U[n * k + i] - sin[ind] * U[n * k + j];
                        double right = sin[ind] * U[n * k + i] + cos[ind] * U[n * k + j];
                        U[n * k + i] = left;
                        U[n * k + j] = right;
                    }
                    //���������� ��������� ������ ������ ����������� �������� �� ��������
                    for (size_t k = 0; k < n; k++) {
                        double left = cos[ind] * V[n * k + i] - sin[ind] * V[n * k + j];
                        double right = sin[ind] * V[n * k + i] + cos[ind] * V[n * k + j];
                        V[n * k + i] = left;
                        V[n * k + j] = right;
                    }
                    ind++;
                }
            }
        }

        for (size_t iteration = 0; iteration < ((2 * ThreadsNum) - 1); iteration++) {
            for (size_t s = 0; s < ThreadsNum; s++) {
                size_t ind = 0;
                for (size_t i = SOB[up[s]].i; i <= SOB[up[s]].j; i++) {
                    for (size_t j = SOB[dn[s]].i; j <= SOB[dn[s]].j; j++) {
                        double dot_ii = 0, dot_jj = 0, dot_ij = 0;
                        //���������� ����� ��������� ��������� �������, � ������� �����������
                        //Uii, Uij, Ujj
                        for (size_t k = 0; k < m; k++) {
                            dot_ii += U[n * k + i] * U[n * k + i];
                            dot_ij += U[n * k + i] * U[n * k + j];
                            dot_jj += U[n * k + j] * U[n * k + j];
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
                for (size_t i = SOB[up[s]].i; i <= SOB[up[s]].j; i++) {
                    for (size_t j = SOB[dn[s]].i; j <= SOB[dn[s]].j; j++) {
                        //���������� ��������� ������ ����� ����������� �������� �� ��������
                        for (size_t k = 0; k < m; k++) {
                            double left = cos[ind] * U[n * k + i] - sin[ind] * U[n * k + j];
                            double right = sin[ind] * U[n * k + i] + cos[ind] * U[n * k + j];
                            U[n * k + i] = left;
                            U[n * k + j] = right;
                        }
                        //���������� ��������� ������ ������ ����������� �������� �� ��������
                        for (size_t k = 0; k < n; k++) {
                            double left = cos[ind] * V[n * k + i] - sin[ind] * V[n * k + j];
                            double right = sin[ind] * V[n * k + i] + cos[ind] * V[n * k + j];
                            V[n * k + i] = left;
                            V[n * k + j] = right;
                        }
                        ind++;
                    }
                }
            }
            round_robin(&up[0], &dn[0], ThreadsNum);
        }
        sweeps++;
        //}
    } while (!converged);

    for (size_t i = 0; i < n; i++) {
        double sigma = 0.0;
        //���������� ����������� �����
        for (size_t k = 0; k < m; k++) {
            sigma += U[n * k + i] * U[n * k + i];
        }
        sigma = sqrt(sigma);
        s[i] = sigma;

        //if (i < n_singular_vals) {
        //    s[i] = sigma;
        //}
        //������������� ����� ����������� ��������
        for (size_t k = 0; k < m; k++) {
            U[n * k + i] /= sigma;
        }
    }
    matrix_t matrices[2] = { Umat, Vmat };
    //���������� ����� � �������� �� �������� ��������
    reorder_decomposition(svec, matrices, 2, greater);

    return sweeps;
}

