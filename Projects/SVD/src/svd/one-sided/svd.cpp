#include "svd.hpp"
#include <assert.h>
#include <math.h>
#include <omp.h>
#include "../../utils/types.hpp"
#include "../../utils/util.hpp"
#include "../../utils/matrix.hpp"

/**
 * @param matrix_t Amat симметричная квадратная (или прямоугольная) матрица A
 * @param vector_t svec вектор сингулярных чисел
 * @param matrix_t Umat матрица левых сингулярных векторов
 * @param matrix_t Vmat матрица правых сингулярных векторов
 * @param size_t n_iter число необходимых для сходимости разверток
 * @return size_t sweeps число разверток методом Якоби
 **/
size_t coloshjac(struct matrix_t Amat, struct vector_t svec, struct matrix_t Umat, struct matrix_t Vmat, size_t ThreadsNum, double* Time) {
    const size_t m = Amat.rows; //строки матрицы A
    const size_t n = Amat.cols; //столбцы матрицы A
    const size_t n_singular_vals = svec.len; //Длина вектора сингулярных значений
    const double tol = 10e-15;  //точность предела сходимости
    double t1, t2;              //замер времени
    size_t sweeps = 0;  //число повторений цикла развертки
    double norm = 0.0;      //норма Фробениуса матрицы A
    double off_norm = 0.0;  //норма Фробениуса только недиагональных элементов матрицы A
    bool converged = true; //Признак сходимости цикла
    
    matrix_copy(Umat, Amat); //инициализация рабочей матрицы U как исходная матрица A
    matrix_identity(Vmat);
    matrix_frobenius(Amat, &norm, &off_norm);
    norm *= tol;

    //assert(m > 0);
    //assert(n > 0);
    //assert(n_singular_vals > 0);
    //assert(((m < n) ? m : n) == n_singular_vals);
    //assert(m == Umat.rows);
    //assert(n == Umat.cols);
    //assert(m == Vmat.rows);
    //assert(n == Vmat.cols);

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
#if ThreadsNum > 1
    #pragma omp parallel for shared(converged, norm, U, V) private(i, n) schedule(static)
#endif
        for (size_t i = 0; i < n - 1; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                double dot_ii = 0, dot_jj = 0, dot_ij = 0;
                //Вычисление суммм квадратов элементов столбцов, в которых расположены
                //Uii, Uij, Ujj
                for (size_t k = 0; k < m; ++k) {
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
                for (size_t k = 0; k < m; ++k) {
                    double left = cosine * U[n * k + i] - sine * U[n * k + j];
                    double right = sine * U[n * k + i] + cosine * U[n * k + j];
                    U[n * k + i] = left;
                    U[n * k + j] = right;
                }
                //Обновление элементов матриц правых сингулярных векторов по столбцам
                for (size_t k = 0; k < n; ++k) {
                    double left = cosine * V[n * k + i] - sine * V[n * k + j];
                    double right = sine * V[n * k + i] + cosine * V[n * k + j];
                    V[n * k + i] = left;
                    V[n * k + j] = right;
                }
            }
        }
        sweeps++;

        if (sweeps > 30)
            converged = true;
    } 
    while (!converged);

    if (sweeps > 30)
        return 0;

#if ThreadsNum > 1
    #pragma omp parallel for shared(s, U) private(i, n) schedule(static)
#endif
    for (size_t i = 0; i < n; ++i) {
        double sigma = 0.0;
        //Вычисление сингулярных чисел
        for (size_t k = 0; k < m; ++k) {
            sigma += U[n * k + i] * U[n * k + i];
        }
        sigma = sqrt(sigma);

        if (i < n_singular_vals) {
            s[i] = sigma;
        }
        //Корректировка левых сингулярных векторов
        for (size_t k = 0; k < m; ++k) {
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
 * @param matrix_t Amat симметричная квадратная (или прямоугольная) матрица A
 * @param vector_t svec вектор сингулярных чисел
 * @param matrix_t Umat матрица левых сингулярных векторов
 * @param matrix_t Vmat матрица правых сингулярных векторов
 * @param size_t n_iter число необходимых для сходимости разверток
 * @return size_t sweeps число разверток методом Якоби
 **/
size_t rrbjrs(struct matrix_t Amat, struct vector_t svec, struct matrix_t Umat, struct matrix_t Vmat, size_t ThreadsNum, double* Time) {
    const size_t m = Amat.rows; //строки матрицы A
    const size_t n = Amat.cols; //столбцы матрицы A
    const size_t n_singular_vals = svec.len; //Длина вектора сингулярных значений
    const double tol = 10e-15;  //точность предела сходимости
    double t1, t2;              //замер времени
    size_t sweeps = 0;  //число повторений цикла развертки
    double norm = 0.0;      //норма Фробениуса матрицы A
    double off_norm = 0.0;  //норма Фробениуса только недиагональных элементов матрицы A
    bool converged = true; //Признак сходимости цикла

    std::vector<size_t> up(ThreadsNum); //массив хранит номер первого блока для одного потока
    std::vector<size_t> dn(ThreadsNum); //массив хранит номер второго блока для одного потока
    std::vector<index_t> SOB(2 * ThreadsNum); //массив хранит пары индексов начала и конца блоков (i - индекс начала блока, j - конец)
    std::vector<double> cos(m * n);
    std::vector<double> sin (m * n);

    //разметка матрицы на блоки (по столбцам)
    bool result = column_limits(Amat, ThreadsNum, &SOB[0]);
    if (!result)
        return 0;

    matrix_copy(Umat, Amat); //инициализация рабочей матрицы U как исходная матрица A
    matrix_identity(Vmat); //матрица V как единичная матрица
    matrix_frobenius(Amat, &norm, &off_norm); //вычисление норм матрицы, домножение на tol - машинное число сходимости
    norm *= tol;
    off_norm *= tol;

    //инициализация массива пар индексов вращений 
    //нечетными (up[]) и четными (dn[]) числами
    for (size_t i = 0; i < ThreadsNum; i++) {
        up[i] = (2 * i);
        dn[i] = (2 * i) + 1;
    }

    //assert(m > 0);
    //assert(n > 0);
    //assert(n_singular_vals > 0);
    //assert(((m < n) ? m : n) == n_singular_vals);
    //assert(m == Umat.rows);
    //assert(n == Umat.cols);
    //assert(m == Vmat.rows);
    //assert(n == Vmat.cols);

    double* A = Amat.ptr;
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
#if ThreadsNum > 1
    #pragma omp parallel for shared(converged, U, V, SOB) private(s, ThreadsNum) schedule(static, 2)
#endif
        for (size_t s = 0; s < (2 * ThreadsNum); s++) {
            size_t ind = 0;
            for (size_t i = SOB[s].i; i <= (SOB[s].j - 1); i++) {
                for (size_t j = (i + 1); j <= (SOB[s].j); j++) {
                    double dot_ii = 0, dot_jj = 0, dot_ij = 0;
                    //Вычисление суммм квадратов элементов стобцов, в которых расположены
                    //Uii, Uij, Ujj
                    for (size_t k = 0; k < m; k++) {
                        dot_ii += U[n * k + i] * U[n * k + i];
                        dot_ij += U[n * k + i] * U[n * k + j];
                        dot_jj += U[n * k + j] * U[n * k + j];
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
            for (size_t i = SOB[s].i; i <= (SOB[s].j - 1); i++) {
                for (size_t j = (i + 1); j <= (SOB[s].j); j++) {
                    //Обновление элементов матриц левых сингулярных векторов по столбцам
                    for (size_t k = 0; k < m; k++) {
                        double left = cos[ind] * U[n * k + i] - sin[ind] * U[n * k + j];
                        double right = sin[ind] * U[n * k + i] + cos[ind] * U[n * k + j];
                        U[n * k + i] = left;
                        U[n * k + j] = right;
                    }
                    //Обновление элементов матриц правых сингулярных векторов по столбцам
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

        //основной цикл решения сингулярного разложения
        //на каждой итерации, с помощью стратегии распределения блоков между потоками (Round Robin)
        //каждый поток получает свои пары блоков (по 2 блока на поток)
        for (size_t iteration = 0; iteration < ((2 * ThreadsNum) - 1); iteration++) {
#if ThreadsNum > 1
    #pragma omp parallel for shared(converged, U, V, SOB, up, dn) private(s, ThreadsNum)
#endif
            for (size_t s = 0; s < ThreadsNum; s++) {
                size_t ind = 0;
                for (size_t i = SOB[up[s]].i; i <= SOB[up[s]].j; i++) {
                    for (size_t j = SOB[dn[s]].i; j <= SOB[dn[s]].j; j++) {
                        double dot_ii = 0, dot_jj = 0, dot_ij = 0;
                        //Вычисление суммм квадратов элементов стобцов, в которых расположены
                        //Uii, Uij, Ujj
                        for (size_t k = 0; k < m; k++) {
                            dot_ii += U[n * k + i] * U[n * k + i];
                            dot_ij += U[n * k + i] * U[n * k + j];
                            dot_jj += U[n * k + j] * U[n * k + j];
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
                for (size_t i = SOB[up[s]].i; i <= SOB[up[s]].j; i++) {
                    for (size_t j = SOB[dn[s]].i; j <= SOB[dn[s]].j; j++) {
                        //Обновление элементов матриц левых сингулярных векторов по столбцам
                        for (size_t k = 0; k < m; k++) {
                            double left = cos[ind] * U[n * k + i] - sin[ind] * U[n * k + j];
                            double right = sin[ind] * U[n * k + i] + cos[ind] * U[n * k + j];
                            U[n * k + i] = left;
                            U[n * k + j] = right;
                        }
                        //Обновление элементов матриц правых сингулярных векторов по столбцам
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
            //стратегия распределения блоков между потоками(Round Robin) формирует новые пары блоков для каждого потока
            round_robin(&up[0], &dn[0], ThreadsNum); 
        }
        sweeps++; //число повторений основного цикла

        if (sweeps > 30)
            converged = true;
    } while (!converged);

    if (sweeps > 30)
        return 0;

#if ThreadsNum > 1
    #pragma omp parallel for shared(s, U) private(i, n) schedule(static)
#endif
    for (size_t i = 0; i < n; i++) {
        double sigma = 0.0;
        //Вычисление сингулярных чисел
        for (size_t k = 0; k < m; k++) {
            sigma += U[n * k + i] * U[n * k + i];
        }
        sigma = sqrt(sigma);
        s[i] = sigma;

        //Корректировка левых сингулярных векторов
        for (size_t k = 0; k < m; k++) {
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

