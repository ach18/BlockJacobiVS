#include "svd.hpp"
#include <cassert>
#include <math.h>
#include <omp.h>
#include <cstdio>
#include <immintrin.h>
#include "../../utils/types.hpp"
#include "../../utils/util.hpp"
#include "../../utils/matrix.hpp"

/**
* Односторонний метод Якоби (Hestenes Jacobi),
  элементы выбираются последовательно по столбцу матрицы
 * @param matrix_t Amat прямоугольная матрица A
 * @param vector_t svec вектор сингулярных чисел
 * @param matrix_t Umat матрица левых сингулярных векторов
 * @param matrix_t Vmat матрица правых сингулярных векторов
 * @return std::size_t sweeps число разверток методом Якоби
 **/
std::size_t coloshjac(struct matrix_t Amat, struct vector_t svec, struct matrix_t Umat, struct matrix_t Vmat, std::size_t ThreadsNum, double* Time, struct string_t errors) {
    const std::size_t m = Amat.rows; //строки матрицы A
    const std::size_t n = Amat.cols; //столбцы матрицы A
    const std::size_t n_singular_vals = svec.len; //Длина вектора сингулярных значений
    const double tol = 10e-15;  //точность предела сходимости
	std::size_t max_sweeps = 40;
    double t1, t2;              //замер времени
    std::size_t sweeps = 0;  //число повторений цикла развертки
    double norm = 0.0;      //норма Фробениуса матрицы A
    double off_norm = 0.0;  //норма Фробениуса только недиагональных элементов матрицы A
    bool converged = true; //Признак сходимости цикла
    
    matrix_copy(Umat, Amat); //инициализация рабочей матрицы U как исходная матрица A
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
        // Перебор над-диагональных элементов матрицы U
    #pragma omp parallel for shared(converged, norm, U, V) firstprivate(n, m) schedule(dynamic) if(ThreadsNum > 1)
        for (std::size_t i = 0; i < n - 1; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                double dot_ii = 0, dot_jj = 0, dot_ij = 0;
                //Вычисление суммм квадратов элементов столбцов, в которых расположены
                //Uii, Uij, Ujj
                for (std::size_t k = 0; k < m; ++k) {
                    dot_ii += U[n * k + i] * U[n * k + i];
                    dot_ij += U[n * k + i] * U[n * k + j];
                    dot_jj += U[n * k + j] * U[n * k + j];
                }

                if (abs(dot_ij) > norm)
                    converged = false;

                double cosine, sine;
                //Вычисление cos, sin матрицы вращения
                sym_jacobi_coeffs(dot_ii, dot_ij, dot_jj, &cosine, &sine);

                //Обновление элементов матриц левых сингулярных векторов по столбцам
                for (std::size_t k = 0; k < m; ++k) {
                    double left = cosine * U[n * k + i] - sine * U[n * k + j];
                    double right = sine * U[n * k + i] + cosine * U[n * k + j];
                    U[n * k + i] = left;
                    U[n * k + j] = right;
                }
                //Обновление элементов матриц правых сингулярных векторов по столбцам
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
        //Вычисление сингулярных чисел
        for (std::size_t k = 0; k < m; ++k) {
            sigma += U[n * k + i] * U[n * k + i];
        }
        sigma = sqrt(sigma);

        if (i < n_singular_vals) {
            s[i] = sigma;
        }
        //Корректировка левых сингулярных векторов
        for (std::size_t k = 0; k < m; ++k) {
            U[n * k + i] /= sigma;
        }
    }

    matrix_t matrices[2] = { Umat, Vmat };
    //Сортировка чисел и векторов по убыванию значений
    reorder_decomposition(svec, matrices, 2, greater);

    t2 = omp_get_wtime();
    *Time = t2 - t1;
    return sweeps;
}

/**
 * Блочный односторонний метод Якоби (Block Jacobi Relaxasion),
 * Блоки выбираются в соответствии со стратегией шахматного турнира (Round Robin)
 * @param matrix_t Amat прямоугольная матрица A
 * @param vector_t svec вектор сингулярных чисел
 * @param matrix_t Umat матрица левых сингулярных векторов
 * @param matrix_t Vmat матрица правых сингулярных векторов
 * @return std::size_t sweeps число разверток методом Якоби
 **/
std::size_t rrbjrs(struct matrix_t Amat, struct matrix_t Bmat, struct vector_t svec, struct matrix_t Umat, struct matrix_t Vmat, std::size_t ThreadsNum, double* Time, struct string_t errors) {
    const std::size_t m = Amat.rows; //строки матрицы A
    const std::size_t n = Amat.cols; //столбцы матрицы A
    const double tol = 10e-15;  //точность предела сходимости
	std::size_t max_sweeps = 30; //максимальное число повторения итераций
    double t1, t2;              //замер времени
    std::size_t sweeps = 0;  //число повторений цикла развертки
    double norm = 0.0;      //норма Фробениуса матрицы A
    double off_norm = 0.0;  //норма Фробениуса только недиагональных элементов матрицы A
    bool converged = true; //Признак сходимости цикла
    std::vector<double> cos(m * n);
    std::vector<double> sin(m * n);
    std::vector<std::size_t> up(ThreadsNum); //массив хранит номер первого блока для одного потока
    std::vector<std::size_t> dn(ThreadsNum); //массив хранит номер второго блока для одного потока
    std::vector<index_t> SOB(2 * ThreadsNum); //массив хранит пары индексов начала и конца двух блоков (i - индекс начала блока, j - конец) для одного потока

    //разметка матрицы на блоки (по столбцам)
    bool result = rrbjrs_column_limits(Amat, ThreadsNum, &SOB[0]);
	if (!result) {
		*errors.len = sprintf(errors.ptr, "matrix must be correctly divided into blocks");
		return 0;
	}

    matrix_copy(Bmat, Amat); //инициализация рабочей матрицы U как исходная матрица A
    matrix_identity(Vmat); //матрица V как единичная матрица

    //инициализация массива пар индексов вращений 
    //нечетными (up[]) и четными (dn[]) числами
    for (std::size_t i = 0; i < ThreadsNum; i++) {
        up[i] = (2 * i);
        dn[i] = (2 * i) + 1;
    }

    double* B = Bmat.ptr;
    double* s = svec.ptr;
    double* U = Umat.ptr;
    double* V = Vmat.ptr;

    for (std::size_t i = 0; i < n; i++) {
        double ortonorm = 0.0;
        for (std::size_t k = 0; k < m; k++) {
#ifdef COLWISE
            ortonorm += B[n * i + k] * B[n * i + k];
#else
            ortonorm += B[n * k + i] * B[n * k + i];
#endif // COLWISE
        }
        norm += ortonorm;
    }
    norm *= tol;

	if(ThreadsNum > 1)
	    omp_set_num_threads(ThreadsNum);

    t1 = omp_get_wtime();

    //цикл пока converged true
    do
    {
        converged = true;
#ifdef COLWISE
        //цикл ортогонализации блоков матрицы
    #pragma omp parallel for shared(converged, V, B, SOB, m, n, norm) private(ThreadsNum) schedule(static,2) if(ThreadsNum > 1)
        for (std::size_t rr_pair = 0; rr_pair < (2 * ThreadsNum); rr_pair++) {
            std::size_t ind = 0;
            for (std::size_t i = SOB[rr_pair].i; i <= (SOB[rr_pair].j - 1); i++) {
                for (std::size_t j = (i + 1); j <= (SOB[rr_pair].j); j++) {
                    double dot_ii = 0, dot_jj = 0, dot_ij = 0;
                    //Вычисление суммм квадратов элементов столбцов, в которых расположены
                    //Uii, Uij, Ujj
                    for (std::size_t k = 0; k < m; k++) {
                        dot_ii += B[n * i + k] * B[n * i + k];
                        dot_ij += B[n * i + k] * B[n * j + k];
                        dot_jj += B[n * j + k] * B[n * j + k];
                    }

                    if (abs(dot_ij) > norm)
                        converged = false;

                    double cosine, sine;
                    //Вычисление cos, sin матрицы вращения
                    jrs_jacobi_coeffs(dot_ii, dot_ij, dot_jj, &cosine, &sine);
                    cos[ind] = cosine;
                    sin[ind] = sine;
                    ind++;
                }
            }
            ind = 0;
            for (std::size_t i = SOB[rr_pair].i; i <= (SOB[rr_pair].j - 1); i++) {
                for (std::size_t j = (i + 1); j <= (SOB[rr_pair].j); j++) {
                    //Обновление элементов матриц левых сингулярных векторов по столбцам
                    for (std::size_t k = 0; k < m; k++) {
                        double left = cos[ind] * B[n * i + k] - sin[ind] * B[n * j + k];
                        double right = sin[ind] * B[n * i + k] + cos[ind] * B[n * j + k];
                        B[n * i + k] = left;
                        B[n * j + k] = right;
                    }
                    //Обновление элементов матриц правых сингулярных векторов по столбцам
                    for (std::size_t k = 0; k < n; k++) {
                        double left = cos[ind] * V[n * i + k] - sin[ind] * V[n * j + k];
                        double right = sin[ind] * V[n * i + k] + cos[ind] * V[n * j + k];
                        V[n * i + k] = left;
                        V[n * j + k] = right;
                    }
                    ind++;
                }
            }
        }

        //основной цикл решения сингулярного разложения
        //на каждой итерации, с помощью стратегии распределения блоков между потоками (Round Robin)
        //каждый поток получает свои пары блоков (по 2 блока на поток)
        for (std::size_t iteration = 0; iteration < ((2 * ThreadsNum) - 1); iteration++) {
#pragma omp parallel for shared(converged, V, B, SOB, up, dn, m, n, norm) private(ThreadsNum) schedule(static,1) if(ThreadsNum > 1)
            for (std::size_t rr_pair = 0; rr_pair < ThreadsNum; rr_pair++) {
                std::size_t ind = 0;
                for (std::size_t i = SOB[up[rr_pair]].i; i <= SOB[up[rr_pair]].j; i++) {
                    for (std::size_t j = SOB[dn[rr_pair]].i; j <= SOB[dn[rr_pair]].j; j++) {
                        double dot_ii = 0, dot_jj = 0, dot_ij = 0;
                        //Вычисление суммм квадратов элементов стобцов, в которых расположены
                        //Uii, Uij, Ujj
                        for (std::size_t k = 0; k < m; k++) {
                            dot_ii += B[n * i + k] * B[n * i + k];
                            dot_ij += B[n * i + k] * B[n * j + k];
                            dot_jj += B[n * j + k] * B[n * j + k];
                        }

                        if (abs(dot_ij) > norm)
                            converged = false;

                        double cosine, sine;
                        //Вычисление cos, sin матрицы вращения
                        jrs_jacobi_coeffs(dot_ii, dot_ij, dot_jj, &cosine, &sine);
                        cos[ind] = cosine;
                        sin[ind] = sine;
                        ind++;
                    }
                }

                ind = 0;
                for (std::size_t i = SOB[up[rr_pair]].i; i <= SOB[up[rr_pair]].j; i++) {
                    for (std::size_t j = SOB[dn[rr_pair]].i; j <= SOB[dn[rr_pair]].j; j++) {
                        //Обновление элементов матриц левых сингулярных векторов по столбцам
                        for (std::size_t k = 0; k < m; k++) {
                            double left = cos[ind] * B[n * i + k] - sin[ind] * B[n * j + k];
                            double right = sin[ind] * B[n * i + k] + cos[ind] * B[n * j + k];
                            B[n * i + k] = left;
                            B[n * j + k] = right;
                        }
                        //Обновление элементов матриц правых сингулярных векторов по столбцам
                        for (std::size_t k = 0; k < n; k++) {
                            double left = cos[ind] * V[n * i + k] - sin[ind] * V[n * j + k];
                            double right = sin[ind] * V[n * i + k] + cos[ind] * V[n * j + k];
                            V[n * i + k] = left;
                            V[n * j + k] = right;
                        }
                        ind++;
                    }
                }
            }
            //стратегия распределения блоков между потоками(Round Robin) формирует новые пары блоков для каждого потока
            round_robin(&up[0], &dn[0], ThreadsNum);
        }
#else
        //цикл ортогонализации блоков матрицы
    #pragma omp parallel for shared(converged, U, V, B, SOB, m, n, norm) private(ThreadsNum) schedule(static,2) if(ThreadsNum > 1)
        for (std::size_t rr_pair = 0; rr_pair < (2 * ThreadsNum); rr_pair++) {
            std::size_t ind = 0;
            for (std::size_t i = SOB[rr_pair].i; i <= (SOB[rr_pair].j - 1); i++) {
                for (std::size_t j = (i + 1); j <= (SOB[rr_pair].j); j++) {
                    double dot_ii = 0, dot_jj = 0, dot_ij = 0;
                    //Вычисление суммм квадратов элементов столбцов, в которых расположены
                    //Uii, Uij, Ujj
                    for (std::size_t k = 0; k < m; k++) {
                        dot_ii += B[n * k + i] * B[n * k + i];
                        dot_ij += B[n * k + i] * B[n * k + j];
                        dot_jj += B[n * k + j] * B[n * k + j];
                    }

                    if (abs(dot_ij) > norm)
                        converged = false;

                    double cosine, sine;
                    //Вычисление cos, sin матрицы вращения
                    jrs_jacobi_coeffs(dot_ii, dot_ij, dot_jj, &cosine, &sine);
                    cos[ind] = cosine;
                    sin[ind] = sine;
                    ind++;
                }
            }
            ind = 0;
            for (std::size_t i = SOB[rr_pair].i; i <= (SOB[rr_pair].j - 1); i++) {
                for (std::size_t j = (i + 1); j <= (SOB[rr_pair].j); j++) {
                    //Обновление элементов матриц левых сингулярных векторов по столбцам
                    for (std::size_t k = 0; k < m; k++) {
                        double left = cos[ind] * B[n * k + i] - sin[ind] * B[n * k + j];
                        double right = sin[ind] * B[n * k + i] + cos[ind] * B[n * k + j];
                        B[n * k + i] = left;
                        B[n * k + j] = right;
                    }
                    //Обновление элементов матриц правых сингулярных векторов по столбцам
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

        //основной цикл решения сингулярного разложения
        //на каждой итерации, с помощью стратегии распределения блоков между потоками (Round Robin)
        //каждый поток получает свои пары блоков (по 2 блока на поток)
        for (std::size_t iteration = 0; iteration < ((2 * ThreadsNum) - 1); iteration++) {
    #pragma omp parallel for shared(converged, U, V, B, SOB, up, dn, m, n, norm) private(ThreadsNum) schedule(static,1) if(ThreadsNum > 1)
            for (std::size_t rr_pair = 0; rr_pair < ThreadsNum; rr_pair++) {
                std::size_t ind = 0;
                for (std::size_t i = SOB[up[rr_pair]].i; i <= SOB[up[rr_pair]].j; i++) {
                    for (std::size_t j = SOB[dn[rr_pair]].i; j <= SOB[dn[rr_pair]].j; j++) {
                        double dot_ii = 0, dot_jj = 0, dot_ij = 0;
                        //Вычисление суммм квадратов элементов стобцов, в которых расположены
                        //Uii, Uij, Ujj
                        for (std::size_t k = 0; k < m; k++) {
                            dot_ii += B[n * k + i] * B[n * k + i];
                            dot_ij += B[n * k + i] * B[n * k + j];
                            dot_jj += B[n * k + j] * B[n * k + j];
                        }

                        if (abs(dot_ij) > norm)
                            converged = false;

                        double cosine, sine;
                        //Вычисление cos, sin матрицы вращения
                        jrs_jacobi_coeffs(dot_ii, dot_ij, dot_jj, &cosine, &sine);
                        cos[ind] = cosine;
                        sin[ind] = sine;
                        ind++;
                    }
                }

                ind = 0;
                for (std::size_t i = SOB[up[rr_pair]].i; i <= SOB[up[rr_pair]].j; i++) {
                    for (std::size_t j = SOB[dn[rr_pair]].i; j <= SOB[dn[rr_pair]].j; j++) {
                        //Обновление элементов матриц левых сингулярных векторов по столбцам
                        for (std::size_t k = 0; k < m; k++) {
                            double left = cos[ind] * B[n * k + i] - sin[ind] * B[n * k + j];
                            double right = sin[ind] * B[n * k + i] + cos[ind] * B[n * k + j];
                            B[n * k + i] = left;
                            B[n * k + j] = right;
                        }
                        //Обновление элементов матриц правых сингулярных векторов по столбцам
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
            //стратегия распределения блоков между потоками(Round Robin) формирует новые пары блоков для каждого потока
            round_robin(&up[0], &dn[0], ThreadsNum); 
        }
#endif
        if (sweeps > max_sweeps)
            converged = true;
		else
			sweeps++; //число повторений основного цикла
    } while (!converged);

	if (sweeps > max_sweeps) {
		*errors.len = sprintf(errors.ptr, "algorithm did not converge after %lu sweeps", sweeps);
		return 0;
	}

#ifdef COLWISE
#pragma omp parallel for shared(s, B, n) schedule(guided) if(ThreadsNum > 1)
    for (std::size_t i = 0; i < n; i++) {
        double sigma = 0.0;
        //Вычисление сингулярных чисел
        for (std::size_t k = 0; k < m; k++) {
            sigma += B[n * i + k] * B[n * i + k];
        }
        sigma = sqrt(sigma);
        s[i] = sigma;

        //Корректировка левых сингулярных векторов
        for (std::size_t k = 0; k < m; k++) {
            B[n * i + k] /= sigma;
        }
    }
#else
#pragma omp parallel for shared(s, B, n) schedule(guided) if(ThreadsNum > 1)
    for (std::size_t i = 0; i < n; i++) {
        double sigma = 0.0;
        //Вычисление сингулярных чисел
        for (std::size_t k = 0; k < m; k++) {
            sigma += B[n * k + i] * B[n * k + i];
        }
        sigma = sqrt(sigma);
        s[i] = sigma;

        //Корректировка левых сингулярных векторов
        for (std::size_t k = 0; k < m; k++) {
            B[n * k + i] /= sigma;
        }
    }
#endif

    matrix_t matrices[2] = { Umat, Vmat };
    //Сортировка чисел и векторов по убыванию значений
    reorder_decomposition(svec, matrices, 2, greater);

    t2 = omp_get_wtime();
    *Time = t2 - t1;

    return sweeps;
}


std::size_t rrbjrs_vectorization(struct matrix_t Amat, struct matrix_t Bmat_aligned, struct vector_t svec_aligned, struct matrix_t Umat_aligned, struct matrix_t Vmat_aligned, std::size_t ThreadsNum, double* Time, struct string_t errors)
{
    const std::size_t m = Amat.rows; //строки матрицы A
    const std::size_t n = Amat.cols; //столбцы матрицы A
    const std::size_t n_singular_vals = svec_aligned.len; //Длина вектора сингулярных значений
    const double tol = 10e-15;  //точность предела сходимости
    std::size_t max_sweeps = 30; //максимальное число повторения итераций
    double t1, t2;              //замер времени
    std::size_t sweeps = 0;  //число повторений цикла развертки
    double norm = 0.0;      //норма Фробениуса матрицы A
    double off_norm = 0.0;  //норма Фробениуса только недиагональных элементов матрицы A
    bool converged = true; //Признак сходимости цикла

    std::vector<std::size_t> up(ThreadsNum); //массив хранит номер первого блока для одного потока
    std::vector<std::size_t> dn(ThreadsNum); //массив хранит номер второго блока для одного потока
    std::vector<index_t> SOB(2 * ThreadsNum); //массив хранит пары индексов начала и конца двух блоков (i - индекс начала блока, j - конец) для одного потока

    //разметка матрицы на блоки (по столбцам)
    bool result = rrbjrs_column_limits(Amat, ThreadsNum, &SOB[0]);
    if (!result) {
        *errors.len = sprintf(errors.ptr, "matrix must be correctly divided into blocks");
        return 0;
    }
    matrix_copy(Bmat_aligned, Amat); //инициализация рабочей матрицы U как исходная матрица A
    matrix_identity(Vmat_aligned); //матрица V как единичная матрица

    //инициализация массива пар индексов вращений 
    //нечетными (up[]) и четными (dn[]) числами
    for (std::size_t i = 0; i < ThreadsNum; i++) {
        up[i] = (2 * i);
        dn[i] = (2 * i) + 1;
    }

    double* B = Bmat_aligned.ptr;
    double* s = svec_aligned.ptr;
    double* U = Umat_aligned.ptr;
    double* V = Vmat_aligned.ptr;

    for (std::size_t i = 0; i < n; i++) {
        double ortonorm = 0.0;
        for (std::size_t k = 0; k < m; k++) {
#ifdef COLWISE
            ortonorm += B[n * i + k] * B[n * i + k];
#else
            ortonorm += B[n * k + i] * B[n * k + i];
#endif // COLWISE
        }
        norm += ortonorm;
    }
    norm *= tol;

    if (ThreadsNum > 1)
        omp_set_num_threads(ThreadsNum);

    t1 = omp_get_wtime();

    //цикл пока converged true
    do
    {
        converged = true;
#ifdef COLWISE
        //цикл ортогонализации блоков матрицы
#pragma omp parallel for shared(converged, U, V, B, SOB, m, n, norm) private(ThreadsNum) schedule(static,2) if(ThreadsNum > 1)
        for (std::size_t rr_pair = 0; rr_pair < (2 * ThreadsNum); rr_pair++) {
            std::size_t ind = 0;
            std::vector<double> cos(m * n);
            std::vector<double> sin(m * n);
            for (std::size_t i = SOB[rr_pair].i; i <= (SOB[rr_pair].j - 1); i++) {
                for (std::size_t j = (i + 1); j <= (SOB[rr_pair].j); j++) {
                    double dot_ii = 0, dot_jj = 0, dot_ij = 0;
                    //Вычисление суммм квадратов элементов столбцов, в которых расположены
                    //Uii, Uij, Ujj
                    for (std::size_t k = 0; k < m; k++) {
                        dot_ii += B[n * i + k] * B[n * i + k];
                        dot_ij += B[n * i + k] * B[n * j + k];
                        dot_jj += B[n * j + k] * B[n * j + k];
                    }

                    if (abs(dot_ij) > norm)
                        converged = false;

                    double cosine, sine;
                    //Вычисление cos, sin матрицы вращения
                    jrs_jacobi_coeffs(dot_ii, dot_ij, dot_jj, &cosine, &sine);
                    cos[ind] = cosine;
                    sin[ind] = sine;
                    ind++;
                }
            }
            ind = 0;
            for (std::size_t i = SOB[rr_pair].i; i <= (SOB[rr_pair].j - 1); i++) {
                for (std::size_t j = (i + 1); j <= (SOB[rr_pair].j); j++) {
                    //Обновление элементов матриц левых сингулярных векторов по столбцам
                    size_t k = 0;
                    __m256d vi0, vj0, left0, right0;
                    __m256d v_c = _mm256_set1_pd(cos[ind]); 
                    __m256d v_s = _mm256_set1_pd(sin[ind]);

                    for (; k + 3 < m; k += 4) {
                        vi0 = _mm256_load_pd(B + n * i + k);
                        vj0 = _mm256_load_pd(B + n * j + k);

                        left0 = _mm256_mul_pd(v_s, vj0);
                        left0 = _mm256_fmsub_pd(v_c, vi0, left0);

                        right0 = _mm256_mul_pd(v_c, vj0);
                        right0 = _mm256_fmadd_pd(v_s, vi0, right0);

                        _mm256_store_pd(B + n * i + k, left0);
                        _mm256_store_pd(B + n * j + k, right0);
                    }
                    for (; k < m; ++k) 
                    {
                        double left = cos[ind] * B[n * i + k] - sin[ind] * B[n * j + k];
                        double right = sin[ind] * B[n * i + k] + cos[ind] * B[n * j + k];
                        B[n * i + k] = left;
                        B[n * j + k] = right;
                    }
                    //Обновление элементов матриц правых сингулярных векторов по столбцам
                    k = 0;
                    for (; k + 3 < n; k += 4) {
                        vi0 = _mm256_load_pd(V + n * i + k);
                        vj0 = _mm256_load_pd(V + n * j + k);

                        left0 = _mm256_mul_pd(v_s, vj0);
                        left0 = _mm256_fmsub_pd(v_c, vi0, left0);

                        right0 = _mm256_mul_pd(v_c, vj0);
                        right0 = _mm256_fmadd_pd(v_s, vi0, right0);

                        _mm256_store_pd(V + n * i + k, left0);
                        _mm256_store_pd(V + n * j + k, right0);
                    }
                    for (; k < m; ++k) 
                    {
                        double left = cos[ind] * V[n * i + k] - sin[ind] * V[n * j + k];
                        double right = sin[ind] * V[n * i + k] + cos[ind] * V[n * j + k];
                        V[n * i + k] = left;
                        V[n * j + k] = right;
                    }
                    ind++;
                }
            }
        }

        //основной цикл решения сингулярного разложения
        //на каждой итерации, с помощью стратегии распределения блоков между потоками (Round Robin)
        //каждый поток получает свои пары блоков (по 2 блока на поток)
        for (std::size_t iteration = 0; iteration < ((2 * ThreadsNum) - 1); iteration++) {
#pragma omp parallel for shared(converged, U, V, B, SOB, up, dn, m, n, norm) private(ThreadsNum) schedule(static,1) if(ThreadsNum > 1)
            for (std::size_t rr_pair = 0; rr_pair < ThreadsNum; rr_pair++) {
                std::size_t ind = 0;
                std::vector<double> cos(m * n);
                std::vector<double> sin(m * n);
                for (std::size_t i = SOB[up[rr_pair]].i; i <= SOB[up[rr_pair]].j; i++) {
                    for (std::size_t j = SOB[dn[rr_pair]].i; j <= SOB[dn[rr_pair]].j; j++) {
                        double dot_ii = 0, dot_jj = 0, dot_ij = 0;
                        //Вычисление суммм квадратов элементов стобцов, в которых расположены
                        //Uii, Uij, Ujj
                        for (std::size_t k = 0; k < m; k++) {
                            dot_ii += B[n * i + k] * B[n * i + k];
                            dot_ij += B[n * i + k] * B[n * j + k];
                            dot_jj += B[n * j + k] * B[n * j + k];
                        }

                        if (abs(dot_ij) > norm)
                            converged = false;

                        double cosine, sine;
                        //Вычисление cos, sin матрицы вращения
                        jrs_jacobi_coeffs(dot_ii, dot_ij, dot_jj, &cosine, &sine);
                        cos[ind] = cosine;
                        sin[ind] = sine;
                        ind++;
                    }
                }

                ind = 0;
                for (std::size_t i = SOB[up[rr_pair]].i; i <= SOB[up[rr_pair]].j; i++) {
                    for (std::size_t j = SOB[dn[rr_pair]].i; j <= SOB[dn[rr_pair]].j; j++) {
                        //Обновление элементов матриц левых сингулярных векторов по столбцам
                        //Обновление элементов матриц левых сингулярных векторов по столбцам
                        size_t k = 0;
                        __m256d vi0, vj0, left0, right0;
                        __m256d v_c = _mm256_set1_pd(cos[ind]);
                        __m256d v_s = _mm256_set1_pd(sin[ind]);
                        
                        for (; k + 3 < m; k += 4) {
                            vi0 = _mm256_load_pd(B + n * i + k);
                            vj0 = _mm256_load_pd(B + n * j + k);

                            left0 = _mm256_mul_pd(v_s, vj0);
                            left0 = _mm256_fmsub_pd(v_c, vi0, left0);

                            right0 = _mm256_mul_pd(v_c, vj0);
                            right0 = _mm256_fmadd_pd(v_s, vi0, right0);

                            _mm256_store_pd(B + n * i + k, left0);
                            _mm256_store_pd(B + n * j + k, right0);
                        }
                        for (; k < m; ++k)
                        {
                            double left = cos[ind] * B[n * i + k] - sin[ind] * B[n * j + k];
                            double right = sin[ind] * B[n * i + k] + cos[ind] * B[n * j + k];
                            B[n * i + k] = left;
                            B[n * j + k] = right;
                        }
                        //Обновление элементов матриц правых сингулярных векторов по столбцам
                        k = 0;
                        for (; k + 3 < n; k += 4) {
                            vi0 = _mm256_load_pd(V + n * i + k);
                            vj0 = _mm256_load_pd(V + n * j + k);

                            left0 = _mm256_mul_pd(v_s, vj0);
                            left0 = _mm256_fmsub_pd(v_c, vi0, left0);

                            right0 = _mm256_mul_pd(v_c, vj0);
                            right0 = _mm256_fmadd_pd(v_s, vi0, right0);

                            _mm256_store_pd(V + n * i + k, left0);
                            _mm256_store_pd(V + n * j + k, right0);
                        }
                        for (; k < m; ++k)
                        {
                            double left = cos[ind] * V[n * i + k] - sin[ind] * V[n * j + k];
                            double right = sin[ind] * V[n * i + k] + cos[ind] * V[n * j + k];
                            V[n * i + k] = left;
                            V[n * j + k] = right;
                        }
                        ind++;
                    }
                }
            }
            //стратегия распределения блоков между потоками(Round Robin) формирует новые пары блоков для каждого потока
            round_robin(&up[0], &dn[0], ThreadsNum);
        }
#else
        //цикл ортогонализации блоков матрицы
#pragma omp parallel for shared(converged, U, V, B, SOB, m, n, norm) private(ThreadsNum) schedule(static,2) if(ThreadsNum > 1)
        for (std::size_t rr_pair = 0; rr_pair < (2 * ThreadsNum); rr_pair++) {
            std::size_t ind = 0;
            std::vector<double> cos(m * n);
            std::vector<double> sin(m * n);
            for (std::size_t i = SOB[rr_pair].i; i <= (SOB[rr_pair].j - 1); i++) {
                for (std::size_t j = (i + 1); j <= (SOB[rr_pair].j); j++) {
                    double dot_ii = 0, dot_jj = 0, dot_ij = 0;
                    //Вычисление суммм квадратов элементов столбцов, в которых расположены
                    //Uii, Uij, Ujj
                    for (std::size_t k = 0; k < m; k++) {
                        dot_ii += B[n * k + i] * B[n * k + i];
                        dot_ij += B[n * k + i] * B[n * k + j];
                        dot_jj += B[n * k + j] * B[n * k + j];
                    }

                    if (abs(dot_ij) > norm)
                        converged = false;

                    double cosine, sine;
                    //Вычисление cos, sin матрицы вращения
                    jrs_jacobi_coeffs(dot_ii, dot_ij, dot_jj, &cosine, &sine);
                    cos[ind] = cosine;
                    sin[ind] = sine;
                    ind++;
                }
            }
            ind = 0;
            for (std::size_t i = SOB[rr_pair].i; i <= (SOB[rr_pair].j - 1); i++) {
                for (std::size_t j = (i + 1); j <= (SOB[rr_pair].j); j++) {
                    //Обновление элементов матриц левых сингулярных векторов по столбцам
                    size_t k = 0;
                    __m256d vi0, vj0, left0, right0;
                    __m256d v_c = _mm256_set1_pd(cos[ind]);
                    __m256d v_s = _mm256_set1_pd(sin[ind]);

                    for (; k + 3 < m; k += 4) {
                        vi0 = _mm256_set_pd(B[n * (k + 0) + i], B[n * (k + 1) + i], B[n * (k + 2) + i], B[n * (k + 3) + i]);
                        vj0 = _mm256_set_pd(B[n * (k + 0) + j], B[n * (k + 1) + j], B[n * (k + 2) + j], B[n * (k + 3) + j]);

                        left0 = _mm256_mul_pd(v_s, vj0);
                        left0 = _mm256_fmsub_pd(v_c, vi0, left0);

                        right0 = _mm256_mul_pd(v_c, vj0);
                        right0 = _mm256_fmadd_pd(v_s, vi0, right0);

                        double* left_0_ptr = (double*)&left0;
                        double* right_0_ptr = (double*)&right0;
                        B[n * (k + 0) + i] = left_0_ptr[3];
                        B[n * (k + 1) + i] = left_0_ptr[2];
                        B[n * (k + 2) + i] = left_0_ptr[1];
                        B[n * (k + 3) + i] = left_0_ptr[0];
                        B[n * (k + 0) + j] = right_0_ptr[3];
                        B[n * (k + 1) + j] = right_0_ptr[2];
                        B[n * (k + 2) + j] = right_0_ptr[1];
                        B[n * (k + 3) + j] = right_0_ptr[0];
                    }
                    for (; k < m; ++k) {
                        double left = cos[ind] * B[n * k + i] - sin[ind] * B[n * k + j];
                        double right = sin[ind] * B[n * k + i] + cos[ind] * B[n * k + j];
                        B[n * k + i] = left;
                        B[n * k + j] = right;
                    }
                    //Обновление элементов матриц правых сингулярных векторов по столбцам
                    for (std::size_t k = 0; k < n; k++) {
                        vi0 = _mm256_set_pd(V[n * (k + 0) + i], V[n * (k + 1) + i], V[n * (k + 2) + i], V[n * (k + 3) + i]);
                        vj0 = _mm256_set_pd(V[n * (k + 0) + j], V[n * (k + 1) + j], V[n * (k + 2) + j], V[n * (k + 3) + j]);

                        left0 = _mm256_mul_pd(v_s, vj0);
                        left0 = _mm256_fmsub_pd(v_c, vi0, left0);

                        right0 = _mm256_mul_pd(v_c, vj0);
                        right0 = _mm256_fmadd_pd(v_s, vi0, right0);

                        double* left_0_ptr = (double*)&left0;
                        double* right_0_ptr = (double*)&right0;
                        V[n * (k + 0) + i] = left_0_ptr[3];
                        V[n * (k + 1) + i] = left_0_ptr[2];
                        V[n * (k + 2) + i] = left_0_ptr[1];
                        V[n * (k + 3) + i] = left_0_ptr[0];
                        V[n * (k + 0) + j] = right_0_ptr[3];
                        V[n * (k + 1) + j] = right_0_ptr[2];
                        V[n * (k + 2) + j] = right_0_ptr[1];
                        V[n * (k + 3) + j] = right_0_ptr[0];
                    }
                    for (; k < n; ++k) {
                        double left = cos[ind] * V[n * k + i] - sin[ind] * V[n * k + j];
                        double right = sin[ind] * V[n * k + i] + cos[ind] * V[n * k + j];
                        V[n * k + i] = left;
                        V[n * k + j] = right;
                    }
                    ind++;
                }
            }
        }

        //основной цикл решения сингулярного разложения
        //на каждой итерации, с помощью стратегии распределения блоков между потоками (Round Robin)
        //каждый поток получает свои пары блоков (по 2 блока на поток)
        for (std::size_t iteration = 0; iteration < ((2 * ThreadsNum) - 1); iteration++) {
#pragma omp parallel for shared(converged, U, V, B, SOB, up, dn, m, n, norm) private(ThreadsNum) schedule(static,1) if(ThreadsNum > 1)
            for (std::size_t rr_pair = 0; rr_pair < ThreadsNum; rr_pair++) {
                std::size_t ind = 0;
                std::vector<double> cos(m * n);
                std::vector<double> sin(m * n);
                for (std::size_t i = SOB[up[rr_pair]].i; i <= SOB[up[rr_pair]].j; i++) {
                    for (std::size_t j = SOB[dn[rr_pair]].i; j <= SOB[dn[rr_pair]].j; j++) {
                        double dot_ii = 0, dot_jj = 0, dot_ij = 0;
                        //Вычисление суммм квадратов элементов стобцов, в которых расположены
                        //Uii, Uij, Ujj
                        for (std::size_t k = 0; k < m; k++) {
                            dot_ii += B[n * k + i] * B[n * k + i];
                            dot_ij += B[n * k + i] * B[n * k + j];
                            dot_jj += B[n * k + j] * B[n * k + j];
                        }

                        if (abs(dot_ij) > norm)
                            converged = false;

                        double cosine, sine;
                        //Вычисление cos, sin матрицы вращения
                        jrs_jacobi_coeffs(dot_ii, dot_ij, dot_jj, &cosine, &sine);
                        cos[ind] = cosine;
                        sin[ind] = sine;
                        ind++;
                    }
                }

                ind = 0;
                for (std::size_t i = SOB[up[rr_pair]].i; i <= SOB[up[rr_pair]].j; i++) {
                    for (std::size_t j = SOB[dn[rr_pair]].i; j <= SOB[dn[rr_pair]].j; j++) {
                        //Обновление элементов матриц левых сингулярных векторов по столбцам
                        size_t k = 0;
                        __m256d vi0, vj0, left0, right0;
                        __m256d v_c = _mm256_set1_pd(cos[ind]);
                        __m256d v_s = _mm256_set1_pd(sin[ind]);

                        for (; k + 3 < m; k += 4) {
                            vi0 = _mm256_set_pd(B[n * (k + 0) + i], B[n * (k + 1) + i], B[n * (k + 2) + i], B[n * (k + 3) + i]);
                            vj0 = _mm256_set_pd(B[n * (k + 0) + j], B[n * (k + 1) + j], B[n * (k + 2) + j], B[n * (k + 3) + j]);

                            left0 = _mm256_mul_pd(v_s, vj0);
                            left0 = _mm256_fmsub_pd(v_c, vi0, left0);

                            right0 = _mm256_mul_pd(v_c, vj0);
                            right0 = _mm256_fmadd_pd(v_s, vi0, right0);

                            double* left_0_ptr = (double*)&left0;
                            double* right_0_ptr = (double*)&right0;
                            B[n * (k + 0) + i] = left_0_ptr[3];
                            B[n * (k + 1) + i] = left_0_ptr[2];
                            B[n * (k + 2) + i] = left_0_ptr[1];
                            B[n * (k + 3) + i] = left_0_ptr[0];
                            B[n * (k + 0) + j] = right_0_ptr[3];
                            B[n * (k + 1) + j] = right_0_ptr[2];
                            B[n * (k + 2) + j] = right_0_ptr[1];
                            B[n * (k + 3) + j] = right_0_ptr[0];
                        }
                        for (; k < m; ++k) {
                            double left = cos[ind] * B[n * k + i] - sin[ind] * B[n * k + j];
                            double right = sin[ind] * B[n * k + i] + cos[ind] * B[n * k + j];
                            B[n * k + i] = left;
                            B[n * k + j] = right;
                        }
                        //Обновление элементов матриц правых сингулярных векторов по столбцам
                        for (std::size_t k = 0; k < n; k++) {
                            vi0 = _mm256_set_pd(V[n * (k + 0) + i], V[n * (k + 1) + i], V[n * (k + 2) + i], V[n * (k + 3) + i]);
                            vj0 = _mm256_set_pd(V[n * (k + 0) + j], V[n * (k + 1) + j], V[n * (k + 2) + j], V[n * (k + 3) + j]);

                            left0 = _mm256_mul_pd(v_s, vj0);
                            left0 = _mm256_fmsub_pd(v_c, vi0, left0);

                            right0 = _mm256_mul_pd(v_c, vj0);
                            right0 = _mm256_fmadd_pd(v_s, vi0, right0);

                            double* left_0_ptr = (double*)&left0;
                            double* right_0_ptr = (double*)&right0;
                            V[n * (k + 0) + i] = left_0_ptr[3];
                            V[n * (k + 1) + i] = left_0_ptr[2];
                            V[n * (k + 2) + i] = left_0_ptr[1];
                            V[n * (k + 3) + i] = left_0_ptr[0];
                            V[n * (k + 0) + j] = right_0_ptr[3];
                            V[n * (k + 1) + j] = right_0_ptr[2];
                            V[n * (k + 2) + j] = right_0_ptr[1];
                            V[n * (k + 3) + j] = right_0_ptr[0];
                        }
                        for (; k < n; ++k) {
                            double left = cos[ind] * V[n * k + i] - sin[ind] * V[n * k + j];
                            double right = sin[ind] * V[n * k + i] + cos[ind] * V[n * k + j];
                            V[n * k + i] = left;
                            V[n * k + j] = right;
                        }
                        ind++;
                    }
                }
            }
            //стратегия распределения блоков между потоками(Round Robin) формирует новые пары блоков для каждого потока
            round_robin(&up[0], &dn[0], ThreadsNum);
        }
#endif
        if (sweeps > max_sweeps)
            converged = true;
        else
            sweeps++; //число повторений основного цикла
    } while (!converged);

    if (sweeps > max_sweeps) {
        *errors.len = sprintf(errors.ptr, "algorithm did not converge after %lu sweeps", sweeps);
        return 0;
    }

#ifdef COLWISE
#pragma omp parallel for shared(s, B, n) schedule(guided) if(ThreadsNum > 1)
    for (std::size_t i = 0; i < n; i++) {
        double sigma = 0.0;
        //Вычисление сингулярных чисел
        for (std::size_t k = 0; k < m; k++) {
            sigma += B[n * i + k] * B[n * i + k];
        }
        sigma = sqrt(sigma);
        s[i] = sigma;

        //Корректировка левых сингулярных векторов
        for (std::size_t k = 0; k < m; k++) {
            B[n * i + k] /= sigma;
        }
    }
#else
#pragma omp parallel for shared(s, B, n) schedule(guided) if(ThreadsNum > 1)
    for (std::size_t i = 0; i < n; i++) {
        double sigma = 0.0;
        //Вычисление сингулярных чисел
        for (std::size_t k = 0; k < m; k++) {
            sigma += B[n * k + i] * B[n * k + i];
        }
        sigma = sqrt(sigma);
        s[i] = sigma;

        //Корректировка левых сингулярных векторов
        for (std::size_t k = 0; k < m; k++) {
            B[n * k + i] /= sigma;
        }
    }
#endif

    matrix_t matrices[2] = { Umat_aligned, Vmat_aligned };
    //Сортировка чисел и векторов по убыванию значений
    reorder_decomposition(svec_aligned, matrices, 2, greater);
    
    t2 = omp_get_wtime();
    *Time = t2 - t1;
    
    return sweeps;
}
