#include "svd.hpp"
#include <assert.h>
#include <math.h>
#include "../../utils/types.hpp"
#include "../../utils/util.hpp"
#include "../../utils/matrix.hpp"

/**
 * @param matrix_t Amat симметрична€ квадратна€ (или пр€моугольна€) матрица A
 * @param vector_t svec вектор сингул€рных чисел
 * @param matrix_t Umat матрица левых сингул€рных векторов
 * @param matrix_t Vmat матрица правых сингул€рных векторов
 * @param size_t n_iter число необходимых дл€ сходимости разверток
 * @return size_t sweeps число разверток методом якоби
 **/
size_t svd(struct matrix_t Amat, struct vector_t svec, struct matrix_t Umat, struct matrix_t Vmat, size_t n_iter) {
    const size_t m = Amat.rows; //строки матрицы A
    const size_t n = Amat.cols; //столбцы матрицы A
    const size_t n_singular_vals = svec.len; //ƒлина вектора сингул€рных значений
    const double tol = 10e-15;  //точность предела сходимости
    size_t sweeps = 0;  //число повторений цикла развертки
    double norm = 0.0;      //норма ‘робениуса матрицы A
    double off_norm = 0.0;  //норма ‘робениуса только недиагональных элементов матрицы A
    
    matrix_copy(Umat, Amat); //инициализаци€ рабочей матрицы U как исходна€ матрица A
    matrix_identity(Vmat);
    matrix_frobenius(Amat, &norm, &off_norm);

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

    //÷икл развертки повтор€етс€ до достижени€ числа итераций
    for (size_t iter = 0; iter < n_iter; ++iter) {
        // ѕеребор над-диагональных элементов матрицы U
        for (size_t i = 0; i < n - 1; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                double dot_ii = 0, dot_jj = 0, dot_ij = 0;
                //¬ычисление суммм квадратов элементов стобцов, в которых расположены
                //Uii, Uij, Ujj
                for (size_t k = 0; k < m; ++k) {
                    dot_ii += U[n * k + i] * U[n * k + i];
                    dot_ij += U[n * k + i] * U[n * k + j];
                    dot_jj += U[n * k + j] * U[n * k + j];
                }

                double cosine, sine;
                //¬ычисление cos, sin матрицы вращени€
                sym_jacobi_coeffs(dot_ii, dot_ij, dot_jj, &cosine, &sine);

                //ќбновление элементов матриц левых сингул€рных векторов
                for (size_t k = 0; k < m; ++k) {
                    double left = cosine * U[n * k + i] - sine * U[n * k + j];
                    double right = sine * U[n * k + i] + cosine * U[n * k + j];
                    U[n * k + i] = left;
                    U[n * k + j] = right;
                }
                //ќбновление элементов матриц правых сингул€рных векторов
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
    //while (converged);

    for (size_t i = 0; i < n; ++i) {
        double sigma = 0.0;
        //¬ычисление сингул€рных чисел
        for (size_t k = 0; k < m; ++k) {
            sigma += U[n * k + i] * U[n * k + i];
        }
        sigma = sqrt(sigma);

        if (i < n_singular_vals) {
            s[i] = sigma;
        }
        // орректировка левых сингул€рных векторов
        for (size_t k = 0; k < m; ++k) {
            U[n * k + i] /= sigma;
        }
    }

    matrix_t matrices[2] = { Umat, Vmat };
    //—ортировка чисел и векторов по убыванию значений
    reorder_decomposition(svec, matrices, 2, greater);

    return sweeps;
}