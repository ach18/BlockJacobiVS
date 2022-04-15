#include "svd.hpp"
#include <cassert>
#include <math.h>
#include <omp.h>
#include <cstdio>
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
    const std::size_t n_singular_vals = svec.len; //Длина вектора сингулярных значений
    const double tol = 10e-15;  //точность предела сходимости
	std::size_t max_sweeps = 40; //максимальное число повторения итераций
    double t1, t2;              //замер времени
    std::size_t sweeps = 0;  //число повторений цикла развертки
    double norm = 0.0;      //норма Фробениуса матрицы A
    double off_norm = 0.0;  //норма Фробениуса только недиагональных элементов матрицы A
    bool converged = true; //Признак сходимости цикла
	//double lambda = 0.0; //Коэффициент релаксации (связан с тау

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
    matrix_frobenius(Amat, &norm, &off_norm); //вычисление норм матрицы, домножение на tol - машинное число сходимости
    norm *= tol;
    off_norm *= tol;

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

	if(ThreadsNum > 1)
	    omp_set_num_threads(ThreadsNum);

    t1 = omp_get_wtime();

    //цикл пока converged true
    do
    {
        converged = true;
        //цикл ортогонализации блоков матрицы
    #pragma omp parallel for shared(converged, U, V, B, SOB, m, n, norm) firstprivate(ThreadsNum) schedule(static,2) if(ThreadsNum > 1)
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
    #pragma omp parallel for shared(converged, U, V, B, SOB, up, dn, m, n, norm) firstprivate(ThreadsNum) schedule(static,1) if(ThreadsNum > 1)
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
        if (sweeps > max_sweeps)
            converged = true;
		else
			sweeps++; //число повторений основного цикла
    } while (!converged);

	if (sweeps > max_sweeps) {
		*errors.len = sprintf(errors.ptr, "algorithm did not converge after %lu sweeps", sweeps);
		return 0;
	}

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

	t2 = omp_get_wtime();
	*Time = t2 - t1;

    matrix_t matrices[2] = { Umat, Vmat };
    //Сортировка чисел и векторов по убыванию значений
    //reorder_decomposition(svec, matrices, 2, greater);

    return sweeps;
}

