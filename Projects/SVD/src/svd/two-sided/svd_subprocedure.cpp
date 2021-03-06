#include "svd_subprocedure.hpp"
#include <math.h>
#include <immintrin.h>
#include "../../utils/matrix.hpp"
#include "nsvd.hpp"

std::size_t svd_subprocedure(struct matrix_t Bmat, struct matrix_t Umat, struct matrix_t Vmat) {
    std::size_t iter = 0;  //????? ?????????? ????? ?????????
	size_t max_sweeps = 40;
    std::size_t n = Bmat.rows; //?????? ???????
    const double tol = 1e-15; //???????? ??????? ??????????
    double* B = Bmat.ptr;
    double* U = Umat.ptr;
    double* V = Vmat.ptr;
    double norm = 0.0;      //????? ?????????? ??????? B
    double off_norm = 0.0;  //????? ?????????? ?????? ?????????????? ????????? ??????? B

    //??????? U ? V ???????????????? ??? ????????? ???????
    matrix_identity(Umat);
    matrix_identity(Vmat);
    //?????????? ????? ??????????, ? ????? ?????????????? ?????????
    matrix_frobenius(Bmat, &norm, &off_norm);

    while (sqrt(off_norm) > tol * sqrt(norm)) {
        for (std::size_t i = 0; i < n - 1; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
#ifdef COLWISE
				double bii = B[n * i + i];
				double bij = B[n * j + i];
				double bji = B[n * i + j];
				double bjj = B[n * j + j];

				//????????? ???????????? cos sin ??? ???????? bij
				//??????? NSVD - https://maths-people.anu.edu.au/~brent/pd/rpb080i.pdf (???. 12)
				struct svd_2x2_params cf = nsvd(bii, bij, bji, bjj);

				//?????????? ???????? J^T*B*J, ?.?. ?????????? ????? ? ????????
				//J - ??????? ??????? ???????? (??????? ???????) ? cos sin ?? ?????? i j
				//?????? ???? ????????? ?????? i j

				for (std::size_t k = 0; k < n; k++) {
					double b_ik = B[n * k + i];
					double b_jk = B[n * k + j];

					double left = cf.c1 * b_ik - cf.s1 * b_jk;
					double right = cf.s1 * cf.k * b_ik + cf.c1 * cf.k * b_jk;

					B[n * k + i] = left;
					B[n * k + j] = right;
				}

				//?????? ???? ????????? ??????? i j
				for (std::size_t k = 0; k < n; k++) {
					double b_ki = B[n * i + k];
					double b_kj = B[n * j + k];

					double left = cf.c2 * b_ki - cf.s2 * b_kj;
					double right = cf.s2 * b_ki + cf.c2 * b_kj;

					B[n * i + k] = left;
					B[n * j + k] = right;
				}

				//????????? ??? ????? ????????? ??????? i j ? ?????? U V 
				for (std::size_t k = 0; k < n; k++) {
					double u_ki = U[n * i + k];
					double u_kj = U[n * j + k];

					double left = cf.c1 * u_ki - cf.s1 * u_kj;
					double right = cf.s1 * cf.k * u_ki + cf.c1 * cf.k * u_kj;

					U[n * i + k] = left;
					U[n * j + k] = right;
				}

				for (std::size_t k = 0; k < n; k++) {
					double v_ki = V[n * i + k];
					double v_kj = V[n * j + k];

					double left = cf.c2 * v_ki - cf.s2 * v_kj;
					double right = cf.s2 * v_ki + cf.c2 * v_kj;

					V[n * i + k] = left;
					V[n * j + k] = right;
				}
#else
                double bii = B[n * i + i];
                double bij = B[n * i + j];
                double bji = B[n * j + i];
                double bjj = B[n * j + j];

				//????????? ???????????? cos sin ??? ???????? bij
                //??????? NSVD - https://maths-people.anu.edu.au/~brent/pd/rpb080i.pdf (???. 12)
                struct svd_2x2_params cf = nsvd(bii, bij, bji, bjj);

                //?????????? ???????? J^T*B*J, ?.?. ?????????? ????? ? ????????
                //J - ??????? ??????? ???????? (??????? ???????) ? cos sin ?? ?????? i j
                //?????? ???? ????????? ?????? i j

                for (std::size_t k = 0; k < n; k++) {
                    double b_ik = B[n * i + k];
                    double b_jk = B[n * j + k];

                    double left = cf.c1 * b_ik - cf.s1 * b_jk;
                    double right = cf.s1 * cf.k * b_ik + cf.c1 * cf.k * b_jk;

                    B[n * i + k] = left;
                    B[n * j + k] = right;
                }

                //?????? ???? ????????? ??????? i j
                for (std::size_t k = 0; k < n; k++) {
                    double b_ki = B[n * k + i];
                    double b_kj = B[n * k + j];

                    double left = cf.c2 * b_ki - cf.s2 * b_kj;
                    double right = cf.s2 * b_ki + cf.c2 * b_kj;

                    B[n * k + i] = left;
                    B[n * k + j] = right;
                }

                //????????? ??? ????? ????????? ??????? i j ? ?????? U V 
                for (std::size_t k = 0; k < n; k++) {
                    double u_ki = U[n * k + i];
                    double u_kj = U[n * k + j];

                    double left = cf.c1 * u_ki - cf.s1 * u_kj;
                    double right = cf.s1 * cf.k * u_ki + cf.c1 * cf.k * u_kj;

                    U[n * k + i] = left;
                    U[n * k + j] = right;
                }

                for (std::size_t k = 0; k < n; k++) {
                    double v_ki = V[n * k + i];
                    double v_kj = V[n * k + j];

                    double left = cf.c2 * v_ki - cf.s2 * v_kj;
                    double right = cf.s2 * v_ki + cf.c2 * v_kj;

                    V[n * k + i] = left;
                    V[n * k + j] = right;
                }
#endif
            }        
		}

        matrix_frobenius(Bmat, &norm, &off_norm);
		if (iter > max_sweeps)
			return 0;
		else
			iter++;
    }

    return iter;
}

size_t svd_subprocedure_vectorized(struct matrix_t Bmat, struct matrix_t Umat, struct matrix_t Vmat) {
	size_t iter = 0;
	size_t max_sweeps = 40;
	size_t n = Bmat.rows;
	const double tol = 1e-15;
	double* B = Bmat.ptr;
	double* U = Umat.ptr;
	double* V = Vmat.ptr;
	double norm = 0.0;
	double off_norm = 0.0;

	matrix_identity(Umat);
	matrix_identity(Vmat);
	matrix_frobenius(Bmat, &norm, &off_norm);

	while (off_norm > tol * tol * norm) {
		for (size_t i = 0; i < n - 1; ++i) {
			for (size_t j = i + 1; j < n; ++j) {
#ifdef COLWISE
				const double bii = B[n * i + i];
				const double bij = B[n * j + i];
				const double bji = B[n * i + j];
				const double bjj = B[n * j + j];
				const struct svd_2x2_params cf = nsvd(bii, bij, bji, bjj);
				const double s1k = cf.s1 * cf.k;
				const double c1k = cf.c1 * cf.k;

				size_t k = 0;
				__m256d vi_0, vi_1, vj_0, vj_1;
				__m256d left_0, left_1, right_0, right_1;
				__m256d v_c = _mm256_set1_pd(cf.c1);
				__m256d v_s = _mm256_set1_pd(cf.s1);
				__m256d v_s1k = _mm256_set1_pd(s1k);
				__m256d v_c1k = _mm256_set1_pd(c1k);
				for (; k + 7 < n; k += 8) {
					vi_0 = _mm256_set_pd(B[n * (k + 0) + i], B[n * (k + 1) + i], B[n * (k + 2) + i], B[n * (k + 3) + i]);
					vi_1 = _mm256_set_pd(B[n * (k + 4) + i], B[n * (k + 5) + i], B[n * (k + 6) + i], B[n * (k + 7) + i]);
					vj_0 = _mm256_set_pd(B[n * (k + 0) + j], B[n * (k + 1) + j], B[n * (k + 2) + j], B[n * (k + 3) + j]);
					vj_1 = _mm256_set_pd(B[n * (k + 4) + j], B[n * (k + 5) + j], B[n * (k + 6) + j], B[n * (k + 7) + j]);

					left_0 = _mm256_mul_pd(v_s, vj_0);
					left_1 = _mm256_mul_pd(v_s, vj_1);
					left_0 = _mm256_fmsub_pd(v_c, vi_0, left_0);
					left_1 = _mm256_fmsub_pd(v_c, vi_1, left_1);

					right_0 = _mm256_mul_pd(v_c1k, vj_0);
					right_1 = _mm256_mul_pd(v_c1k, vj_1);
					right_0 = _mm256_fmadd_pd(v_s1k, vi_0, right_0);
					right_1 = _mm256_fmadd_pd(v_s1k, vi_1, right_1);

					double* left_0_ptr = (double*)&left_0;
					double* left_1_ptr = (double*)&left_1;
					double* right_0_ptr = (double*)&right_0;
					double* right_1_ptr = (double*)&right_1;
					B[n * (k + 0) + i] = left_0_ptr[3];
					B[n * (k + 1) + i] = left_0_ptr[2];
					B[n * (k + 2) + i] = left_0_ptr[1];
					B[n * (k + 3) + i] = left_0_ptr[0];
					B[n * (k + 4) + i] = left_1_ptr[3];
					B[n * (k + 5) + i] = left_1_ptr[2];
					B[n * (k + 6) + i] = left_1_ptr[1];
					B[n * (k + 7) + i] = left_1_ptr[0];
					B[n * (k + 0) + j] = right_0_ptr[3];
					B[n * (k + 1) + j] = right_0_ptr[2];
					B[n * (k + 2) + j] = right_0_ptr[1];
					B[n * (k + 3) + j] = right_0_ptr[0];
					B[n * (k + 4) + j] = right_1_ptr[3];
					B[n * (k + 5) + j] = right_1_ptr[2];
					B[n * (k + 6) + j] = right_1_ptr[1];
					B[n * (k + 7) + j] = right_1_ptr[0];
				}
				for (; k < n; ++k) {
					double b_ik = B[n * k + i];
					double b_jk = B[n * k + j];
					double left = cf.c1 * b_ik - cf.s1 * b_jk;
					double right = s1k * b_ik + c1k * b_jk;
					B[n * k + i] = left;
					B[n * k + j] = right;
				}

				k = 0;
				v_c = _mm256_set1_pd(cf.c2);
				v_s = _mm256_set1_pd(cf.s2);
				for (; k + 7 < n; k += 8) {
					vi_0 = _mm256_loadu_pd(B + n * i + (k + 0));
					vi_1 = _mm256_loadu_pd(B + n * i + (k + 4));
					vj_0 = _mm256_loadu_pd(B + n * j + (k + 0));
					vj_1 = _mm256_loadu_pd(B + n * j + (k + 4));

					left_0 = _mm256_mul_pd(v_s, vj_0);
					left_1 = _mm256_mul_pd(v_s, vj_1);
					left_0 = _mm256_fmsub_pd(v_c, vi_0, left_0);
					left_1 = _mm256_fmsub_pd(v_c, vi_1, left_1);

					right_0 = _mm256_mul_pd(v_c, vj_0);
					right_1 = _mm256_mul_pd(v_c, vj_1);
					right_0 = _mm256_fmadd_pd(v_s, vi_0, right_0);
					right_1 = _mm256_fmadd_pd(v_s, vi_1, right_1);

					_mm256_storeu_pd(B + n * i + k + 0, left_0);
					_mm256_storeu_pd(B + n * i + k + 4, left_1);
					_mm256_storeu_pd(B + n * j + k + 0, right_0);
					_mm256_storeu_pd(B + n * j + k + 4, right_1);
				}
				for (; k < n; ++k) {
					double b_ki = B[n * i + k];
					double b_kj = B[n * j + k];
					double left = cf.c2 * b_ki - cf.s2 * b_kj;
					double right = cf.s2 * b_ki + cf.c2 * b_kj;
					B[n * i + k] = left;
					B[n * j + k] = right;
				}

				k = 0;
				v_c = _mm256_set1_pd(cf.c1);
				v_s = _mm256_set1_pd(cf.s1);
				for (; k + 7 < n; k += 8) {
					vi_0 = _mm256_loadu_pd(U + n * i + (k + 0));
					vi_1 = _mm256_loadu_pd(U + n * i + (k + 4));
					vj_0 = _mm256_loadu_pd(U + n * j + (k + 0));
					vj_1 = _mm256_loadu_pd(U + n * j + (k + 4));

					left_0 = _mm256_mul_pd(v_s, vj_0);
					left_1 = _mm256_mul_pd(v_s, vj_1);
					left_0 = _mm256_fmsub_pd(v_c, vi_0, left_0);
					left_1 = _mm256_fmsub_pd(v_c, vi_1, left_1);

					right_0 = _mm256_mul_pd(v_c1k, vj_0);
					right_1 = _mm256_mul_pd(v_c1k, vj_1);
					right_0 = _mm256_fmadd_pd(v_s1k, vi_0, right_0);
					right_1 = _mm256_fmadd_pd(v_s1k, vi_1, right_1);

					_mm256_storeu_pd(U + n * i + k + 0, left_0);
					_mm256_storeu_pd(U + n * i + k + 4, left_1);
					_mm256_storeu_pd(U + n * j + k + 0, right_0);
					_mm256_storeu_pd(U + n * j + k + 4, right_1);
				}
				for (; k < n; ++k) {
					double u_ki = U[n * i + k];
					double u_kj = U[n * j + k];
					double left = cf.c1 * u_ki - cf.s1 * u_kj;
					double right = s1k * u_ki + c1k * u_kj;
					U[n * i + k] = left;
					U[n * j + k] = right;
				}

				k = 0;
				v_c = _mm256_set1_pd(cf.c2);
				v_s = _mm256_set1_pd(cf.s2);
				for (; k + 7 < n; k += 8) {
					vi_0 = _mm256_loadu_pd(V + n * i + (k + 0));
					vi_1 = _mm256_loadu_pd(V + n * i + (k + 4));
					vj_0 = _mm256_loadu_pd(V + n * j + (k + 0));
					vj_1 = _mm256_loadu_pd(V + n * j + (k + 4));

					left_0 = _mm256_mul_pd(v_s, vj_0);
					left_1 = _mm256_mul_pd(v_s, vj_1);
					left_0 = _mm256_fmsub_pd(v_c, vi_0, left_0);
					left_1 = _mm256_fmsub_pd(v_c, vi_1, left_1);

					right_0 = _mm256_mul_pd(v_c, vj_0);
					right_1 = _mm256_mul_pd(v_c, vj_1);
					right_0 = _mm256_fmadd_pd(v_s, vi_0, right_0);
					right_1 = _mm256_fmadd_pd(v_s, vi_1, right_1);

					_mm256_storeu_pd(V + n * i + k + 0, left_0);
					_mm256_storeu_pd(V + n * i + k + 4, left_1);
					_mm256_storeu_pd(V + n * j + k + 0, right_0);
					_mm256_storeu_pd(V + n * j + k + 4, right_1);
				}
				for (; k < n; ++k) {
					double v_ki = V[n * i + k];
					double v_kj = V[n * j + k];
					double left = cf.c2 * v_ki - cf.s2 * v_kj;
					double right = cf.s2 * v_ki + cf.c2 * v_kj;
					V[n * i + k] = left;
					V[n * j + k] = right;
				}
#else
				const double bii = B[n * i + i];
				const double bij = B[n * i + j];
				const double bji = B[n * j + i];
				const double bjj = B[n * j + j];
				const struct svd_2x2_params cf = nsvd(bii, bij, bji, bjj);
				const double s1k = cf.s1 * cf.k;
				const double c1k = cf.c1 * cf.k;

				size_t k = 0;
				__m256d vi_0, vi_1, vj_0, vj_1;
				__m256d left_0, left_1, right_0, right_1;
				__m256d v_c = _mm256_set1_pd(cf.c1);
				__m256d v_s = _mm256_set1_pd(cf.s1);
				__m256d v_s1k = _mm256_set1_pd(s1k);
				__m256d v_c1k = _mm256_set1_pd(c1k);
				for (; k + 7 < n; k += 8) {
					vi_0 = _mm256_loadu_pd(B + n * i + (k + 0));
					vi_1 = _mm256_loadu_pd(B + n * i + (k + 4));
					vj_0 = _mm256_loadu_pd(B + n * j + (k + 0));
					vj_1 = _mm256_loadu_pd(B + n * j + (k + 4));

					left_0 = _mm256_mul_pd(v_s, vj_0);
					left_1 = _mm256_mul_pd(v_s, vj_1);
					left_0 = _mm256_fmsub_pd(v_c, vi_0, left_0);
					left_1 = _mm256_fmsub_pd(v_c, vi_1, left_1);

					right_0 = _mm256_mul_pd(v_c1k, vj_0);
					right_1 = _mm256_mul_pd(v_c1k, vj_1);
					right_0 = _mm256_fmadd_pd(v_s1k, vi_0, right_0);
					right_1 = _mm256_fmadd_pd(v_s1k, vi_1, right_1);

					_mm256_storeu_pd(B + n * i + k + 0, left_0);
					_mm256_storeu_pd(B + n * i + k + 4, left_1);
					_mm256_storeu_pd(B + n * j + k + 0, right_0);
					_mm256_storeu_pd(B + n * j + k + 4, right_1);
				}
				for (; k < n; ++k) {
					double b_ik = B[n * i + k];
					double b_jk = B[n * j + k];
					double left = cf.c1 * b_ik - cf.s1 * b_jk;
					double right = s1k * b_ik + c1k * b_jk;
					B[n * i + k] = left;
					B[n * j + k] = right;
				}

				k = 0;
				v_c = _mm256_set1_pd(cf.c2);
				v_s = _mm256_set1_pd(cf.s2);
				for (; k + 7 < n; k += 8) {
					vi_0 = _mm256_set_pd(B[n * (k + 0) + i], B[n * (k + 1) + i], B[n * (k + 2) + i], B[n * (k + 3) + i]);
					vi_1 = _mm256_set_pd(B[n * (k + 4) + i], B[n * (k + 5) + i], B[n * (k + 6) + i], B[n * (k + 7) + i]);
					vj_0 = _mm256_set_pd(B[n * (k + 0) + j], B[n * (k + 1) + j], B[n * (k + 2) + j], B[n * (k + 3) + j]);
					vj_1 = _mm256_set_pd(B[n * (k + 4) + j], B[n * (k + 5) + j], B[n * (k + 6) + j], B[n * (k + 7) + j]);

					left_0 = _mm256_mul_pd(v_s, vj_0);
					left_1 = _mm256_mul_pd(v_s, vj_1);
					left_0 = _mm256_fmsub_pd(v_c, vi_0, left_0);
					left_1 = _mm256_fmsub_pd(v_c, vi_1, left_1);

					right_0 = _mm256_mul_pd(v_c, vj_0);
					right_1 = _mm256_mul_pd(v_c, vj_1);
					right_0 = _mm256_fmadd_pd(v_s, vi_0, right_0);
					right_1 = _mm256_fmadd_pd(v_s, vi_1, right_1);

					double* left_0_ptr = (double*)&left_0;
					double* left_1_ptr = (double*)&left_1;
					double* right_0_ptr = (double*)&right_0;
					double* right_1_ptr = (double*)&right_1;
					B[n * (k + 0) + i] = left_0_ptr[3];
					B[n * (k + 1) + i] = left_0_ptr[2];
					B[n * (k + 2) + i] = left_0_ptr[1];
					B[n * (k + 3) + i] = left_0_ptr[0];
					B[n * (k + 4) + i] = left_1_ptr[3];
					B[n * (k + 5) + i] = left_1_ptr[2];
					B[n * (k + 6) + i] = left_1_ptr[1];
					B[n * (k + 7) + i] = left_1_ptr[0];
					B[n * (k + 0) + j] = right_0_ptr[3];
					B[n * (k + 1) + j] = right_0_ptr[2];
					B[n * (k + 2) + j] = right_0_ptr[1];
					B[n * (k + 3) + j] = right_0_ptr[0];
					B[n * (k + 4) + j] = right_1_ptr[3];
					B[n * (k + 5) + j] = right_1_ptr[2];
					B[n * (k + 6) + j] = right_1_ptr[1];
					B[n * (k + 7) + j] = right_1_ptr[0];
				}
				for (; k < n; ++k) {
					double b_ki = B[n * k + i];
					double b_kj = B[n * k + j];
					double left = cf.c2 * b_ki - cf.s2 * b_kj;
					double right = cf.s2 * b_ki + cf.c2 * b_kj;
					B[n * k + i] = left;
					B[n * k + j] = right;
				}

				k = 0;
				v_c = _mm256_set1_pd(cf.c1);
				v_s = _mm256_set1_pd(cf.s1);
				for (; k + 7 < n; k += 8) {
					vi_0 = _mm256_set_pd(U[n * (k + 0) + i], U[n * (k + 1) + i], U[n * (k + 2) + i], U[n * (k + 3) + i]);
					vi_1 = _mm256_set_pd(U[n * (k + 4) + i], U[n * (k + 5) + i], U[n * (k + 6) + i], U[n * (k + 7) + i]);
					vj_0 = _mm256_set_pd(U[n * (k + 0) + j], U[n * (k + 1) + j], U[n * (k + 2) + j], U[n * (k + 3) + j]);
					vj_1 = _mm256_set_pd(U[n * (k + 4) + j], U[n * (k + 5) + j], U[n * (k + 6) + j], U[n * (k + 7) + j]);

					left_0 = _mm256_mul_pd(v_s, vj_0);
					left_1 = _mm256_mul_pd(v_s, vj_1);
					left_0 = _mm256_fmsub_pd(v_c, vi_0, left_0);
					left_1 = _mm256_fmsub_pd(v_c, vi_1, left_1);

					right_0 = _mm256_mul_pd(v_c1k, vj_0);
					right_1 = _mm256_mul_pd(v_c1k, vj_1);
					right_0 = _mm256_fmadd_pd(v_s1k, vi_0, right_0);
					right_1 = _mm256_fmadd_pd(v_s1k, vi_1, right_1);

					double* left_0_ptr = (double*)&left_0;
					double* left_1_ptr = (double*)&left_1;
					double* right_0_ptr = (double*)&right_0;
					double* right_1_ptr = (double*)&right_1;
					U[n * (k + 0) + i] = left_0_ptr[3];
					U[n * (k + 1) + i] = left_0_ptr[2];
					U[n * (k + 2) + i] = left_0_ptr[1];
					U[n * (k + 3) + i] = left_0_ptr[0];
					U[n * (k + 4) + i] = left_1_ptr[3];
					U[n * (k + 5) + i] = left_1_ptr[2];
					U[n * (k + 6) + i] = left_1_ptr[1];
					U[n * (k + 7) + i] = left_1_ptr[0];
					U[n * (k + 0) + j] = right_0_ptr[3];
					U[n * (k + 1) + j] = right_0_ptr[2];
					U[n * (k + 2) + j] = right_0_ptr[1];
					U[n * (k + 3) + j] = right_0_ptr[0];
					U[n * (k + 4) + j] = right_1_ptr[3];
					U[n * (k + 5) + j] = right_1_ptr[2];
					U[n * (k + 6) + j] = right_1_ptr[1];
					U[n * (k + 7) + j] = right_1_ptr[0];
				}
				for (; k < n; ++k) {
					double u_ki = U[n * k + i];
					double u_kj = U[n * k + j];
					double left = cf.c1 * u_ki - cf.s1 * u_kj;
					double right = s1k * u_ki + c1k * u_kj;
					U[n * k + i] = left;
					U[n * k + j] = right;
				}

				k = 0;
				v_c = _mm256_set1_pd(cf.c2);
				v_s = _mm256_set1_pd(cf.s2);
				for (; k + 7 < n; k += 8) {
					vi_0 = _mm256_set_pd(V[n * (k + 0) + i], V[n * (k + 1) + i], V[n * (k + 2) + i], V[n * (k + 3) + i]);
					vi_1 = _mm256_set_pd(V[n * (k + 4) + i], V[n * (k + 5) + i], V[n * (k + 6) + i], V[n * (k + 7) + i]);
					vj_0 = _mm256_set_pd(V[n * (k + 0) + j], V[n * (k + 1) + j], V[n * (k + 2) + j], V[n * (k + 3) + j]);
					vj_1 = _mm256_set_pd(V[n * (k + 4) + j], V[n * (k + 5) + j], V[n * (k + 6) + j], V[n * (k + 7) + j]);

					left_0 = _mm256_mul_pd(v_s, vj_0);
					left_1 = _mm256_mul_pd(v_s, vj_1);
					left_0 = _mm256_fmsub_pd(v_c, vi_0, left_0);
					left_1 = _mm256_fmsub_pd(v_c, vi_1, left_1);

					right_0 = _mm256_mul_pd(v_c, vj_0);
					right_1 = _mm256_mul_pd(v_c, vj_1);
					right_0 = _mm256_fmadd_pd(v_s, vi_0, right_0);
					right_1 = _mm256_fmadd_pd(v_s, vi_1, right_1);

					double* left_0_ptr = (double*)&left_0;
					double* left_1_ptr = (double*)&left_1;
					double* right_0_ptr = (double*)&right_0;
					double* right_1_ptr = (double*)&right_1;
					V[n * (k + 0) + i] = left_0_ptr[3];
					V[n * (k + 1) + i] = left_0_ptr[2];
					V[n * (k + 2) + i] = left_0_ptr[1];
					V[n * (k + 3) + i] = left_0_ptr[0];
					V[n * (k + 4) + i] = left_1_ptr[3];
					V[n * (k + 5) + i] = left_1_ptr[2];
					V[n * (k + 6) + i] = left_1_ptr[1];
					V[n * (k + 7) + i] = left_1_ptr[0];
					V[n * (k + 0) + j] = right_0_ptr[3];
					V[n * (k + 1) + j] = right_0_ptr[2];
					V[n * (k + 2) + j] = right_0_ptr[1];
					V[n * (k + 3) + j] = right_0_ptr[0];
					V[n * (k + 4) + j] = right_1_ptr[3];
					V[n * (k + 5) + j] = right_1_ptr[2];
					V[n * (k + 6) + j] = right_1_ptr[1];
					V[n * (k + 7) + j] = right_1_ptr[0];
				}
				for (; k < n; ++k) {
					double v_ki = V[n * k + i];
					double v_kj = V[n * k + j];
					double left = cf.c2 * v_ki - cf.s2 * v_kj;
					double right = cf.s2 * v_ki + cf.c2 * v_kj;
					V[n * k + i] = left;
					V[n * k + j] = right;
				}
#endif
			}
		}

		matrix_off_frobenius(Bmat, &off_norm);
		if (iter > max_sweeps)
			return 0;
		else
			iter++;
	}

	return iter;
}