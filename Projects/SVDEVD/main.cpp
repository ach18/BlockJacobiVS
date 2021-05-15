#include <algorithm>
#include <omp.h>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include <vector>
#include "src/utils/types.hpp"
#include "src/utils/util.hpp"
#include "src/svd/two-sided/svd.hpp"
#include "src/svd/one-sided/svd.hpp"
#include "alglib/linalg.h"

void svd_test(matrix_t Data_matr, matrix_t B_mat, matrix_t U_mat, matrix_t V_mat, vector_t S_vect, size_t n, size_t block_size, size_t threads, double time);
void colbnsvd_test(matrix_t Data_matr, matrix_t B_mat, matrix_t U_mat, matrix_t V_mat, size_t block_size, size_t threads, double* time);

int main(int argc, char* argv[])
{
    /*std::vector<index_t> sizes = { {100, 100}, {200, 100}, {200, 200}, {300, 100}, {300, 200}, {300, 300}, {500, 100}, {500, 200}, {500, 300}, {500, 500},
        {1000, 100}, {1000, 200}, {1000, 300}, {1000, 500}, {1000, 1000}, {1100, 100}, {1100, 200}, {1100, 300}, {1100, 500}, {1100, 1000}, {1100, 1100},
        {1200, 100}, {1200, 200}, {1200, 300}, {1200, 500}, {1200, 1000}, {1200, 1100}, {1200, 1200}, {1300, 100}, {1300, 200}, {1300, 300}, {1300, 500},
        {1300, 1000}, {1300, 1100}, {1300, 1200}, {1300, 1300}, {1500, 100}, {1500, 200}, {1500, 300}, {1500, 500}, {1500, 1000}, {1500, 1100}, {1500, 1200},
        {1500, 1300}, {1500, 1500}, {2000, 100}, {2000, 200}, {2000, 300}, {2000, 500}, {2000, 1000}, {2000, 1100}, {2000, 1200}, {2000, 1300}, {2000, 1500},
        {2000, 2000}, {2100, 100}, {2100, 200}, {2100, 300}, {2100, 500}, {2100, 1000}, {2100, 1100}, {2100, 1200}, {2100, 1300}, {2100, 1500}, {2100, 2000},
        {2100, 2100}, {2200, 100}, {2200, 200}, {2200, 300}, {2200, 500}, {2200, 1000}, {2200, 1100}, {2200, 1200}, {2200, 1300}, {2200, 1500}, {2200, 2000},
        {2200, 2100}, {2200, 2200}, {2300, 100}, {2300, 200}, {2300, 300}, {2300, 500}, {2300, 1000}, {2300, 1100}, {2300, 1200}, {2300, 1300}, {2300, 1500},
        {2300, 2000}, {2300, 2100}, {2300, 2200}, {2300, 2300}, {2500, 100}, {2500, 200}, {2500, 300}, {2500, 500}, {2500, 1000}, {2500, 1100}, {2500, 1200},
        {2500, 1300}, {2500, 1500}, {2500, 2000}, {2500, 2100}, {2500, 2200}, {2500, 2300}, {2500, 2500}, {3000, 100}, {3000, 200}, {3000, 300}, {3000, 500},
        {3000, 1000}, {3000, 1100}, {3000, 1200}, {3000, 1300}, {3000, 1500}, {3000, 2000}, {3000, 2100}, {3000, 2200}, {3000, 2300}, {3000, 2500},
        {3000, 3000}, {3100, 100}, {3100, 200}, {3100, 300}, {3100, 500}, {3100, 1000}, {3100, 1100}, {3100, 1200}, {3100, 1300}, {3100, 1500}, {3100, 2000},
        {3100, 2100}, {3100, 2200}, {3100, 2300}, {3100, 2500}, {3100, 3000}, {3100, 3100}, {3200, 100}, {3200, 200}, {3200, 300}, {3200, 500}, {3200, 1000},
        {3200, 1100}, {3200, 1200}, {3200, 1300}, {3200, 1500}, {3200, 2000}, {3200, 2100}, {3200, 2200}, {3200, 2300}, {3200, 2500}, {3200, 3000},
        {3200, 3100}, {3200, 3200}, {3300, 100}, {3300, 200}, {3300, 300}, {3300, 500}, {3300, 1000}, {3300, 1100}, {3300, 1200}, {3300, 1300}, {3300, 1500},
        {3300, 2000}, {3300, 2100}, {3300, 2200}, {3300, 2300}, {3300, 2500}, {3300, 3000}, {3300, 3100}, {3300, 3200}, {3300, 3300}, {3500, 100},
        {3500, 200}, {3500, 300}, {3500, 500}, {3500, 1000}, {3500, 1200}, {3500, 1300}, {3500, 1500}, {3500, 2000}, {3500, 2100}, {3500, 2200}, {3500, 2300},
        {3500, 2500}, {3500, 3000}, {3500, 3100}, {3500, 3200}, {3500, 3300}, {3500, 3500}, {4000, 100}, {4000, 200}, {4000, 300}, {4000, 500}, {4000, 1000},
        {4000, 1100}, {4000, 1200}, {4000, 1300}, {4000, 1500}, {4000, 2000}, {4000, 2100}, {4000, 2200}, {4000, 2300}, {4000, 2500}, {4000, 3000},
        {4000, 3100}, {4000, 3200}, {4000, 3300}, {4000, 3500}, {4000, 4000}, {4100, 100}, {4100, 200}, {4100, 300}, {4100, 500}, {4100, 500}, {4100, 1000},
        {4100, 1100}, {4100, 1200}, {4100, 1300}, {4100, 1500}, {4100, 2000}, {4100, 2100}, {4100, 2200}, {4100, 2300}, {4100, 2500}, {4100, 3000},
        {4100, 3100}, {4100, 3200}, {4100, 3300}, {4100, 3500}, {4100, 4000}, {4100, 4100}, {4200, 100}, {4200, 200}, {4200, 500}, {4200, 1000},
        {4200, 1100}, {4200, 1200}, {4200, 1300}, {4200, 1500}, {4200, 2000}, {4200, 2100}, {4200, 2200}, {4200, 2300}, {4200, 2500}, {4200, 3000},
        {4200, 3100}, {4200, 3200}, {4200, 3300}, {4200, 3500}, {4200, 4000}, {4200, 4100}, {4200, 4200}, {4300, 100}, {4300, 200}, {4300, 300}, {4300, 500},
        {4300, 1000}, {4300, 1100}, {4300, 1200}, {4300, 1300}, {4300, 1500}, {4300, 2000}, {4300, 2100}, {4300, 2200}, {4300, 2300}, {4300, 2500},
        {4300, 3000}, {4300, 3100}, {4300, 3200}, {4300, 3300}, {4300, 3500}, {4300, 4000}, {4300, 4100}, {4300, 4200}, {4300, 4300}, {4500, 100},
        {4500, 200}, {4500, 300}, {4500, 500}, {4500, 1000}, {4500, 1100}, {4500, 1200}, {4500, 1300}, {4500, 1500}, {4500, 2000}, {4500, 2100},
        {4500, 2200}, {4500, 2300}, {4500, 2500}, {4500, 3000}, {4500, 3100}, {4500, 3200}, {4500, 3300}, {4500, 3500}, {4500, 4000}, {4500, 4100},
        {4500, 4200}, {4500, 4300}, {4500, 4500}, {5000, 100}, {5000, 200}, {5000, 300}, {5000, 500}, {5000, 1000}, {5000, 1200}, {5000, 1300}, {5000, 1500},
        {5000, 2000}, {5000, 2100}, {5000, 2200}, {5000, 2300}, {5000, 2500}, {5000, 3000}, {5000, 3100}, {5000, 3200}, {5000, 3300}, {5000, 3500},
        {5000, 4000}, {5000, 4100}, {5000, 4200}, {5000, 4300}, {5000, 4500}, {5000, 5000}, {5100, 100}, {5100, 200}, {5100, 300}, {5100, 500}, {5100, 1000},
        {5100, 1100}, {5100, 1200}, {5100, 1300}, {5100, 1500}, {5100, 2000}, {5100, 2100}, {5100, 2200}, {5100, 2300}, {5100, 2500}, {5100, 3000},
        {5100, 3100}, {5100, 3200}, {5100, 3300}, {5100, 3500}, {5100, 4000}, {5100, 4100}, {5100, 4200}, {5100, 4300}, {5100, 4500}, {5100, 5000},
        {5100, 5100}, {5200, 100}, {5200, 200}, {5200, 300}, {5200, 500}, {5200, 1000}, {5200, 1100}, {5200, 1200}, {5200, 1300}, {5200, 1500},
        {5200, 2000}, {5200, 2100}, {5200, 2200}, {5200, 2300}, {5200, 2500}, {5200, 3000}, {5200, 3100}, {5200, 3200}, {5200, 3300}, {5200, 3500},
        {5200, 4000}, {5200, 4100}, {5200, 4200}, {5200, 4300}, {5200, 4500}, {5200, 5000}, {5200, 5100}, {5200, 5200}, {5300, 100}, {5300, 200}, {5300, 300},
        {5300, 500}, {5300, 1000}, {5300, 1100}, {5300, 1200}, {5300, 1300}, {5300, 1500}, {5300, 2000}, {5300, 2100}, {5300, 2200}, {5300, 2300},
        {5300, 2500}, {5300, 3000}, {5300, 3100}, {5300, 3200}, {5300, 3300}, {5300, 3500}, {5300, 4000}, {5300, 100}, {5300, 200}, {5300, 300}, {5300, 500},
        {5300, 1000}, {5300, 1100}, {5300, 1200}, {5300, 1300}, {5300, 1500}, {5300, 2000}, {5300, 2100}, {5300, 2200}, {5300, 2300}, {5300, 2500},
        {5300, 3000}, {5300, 3100}, {5300, 3200}, {5300, 3300}, {5300, 3500}, {5300, 4000}, {5300, 4100}, {5300, 4200}, {5300, 4300}, {5300, 4500},
        {5300, 5000}, {5300, 5100}, {5300, 5200}, {5300, 5300}, {5500, 100}, {5500, 200}, {5500, 300}, {5500, 500}, {5500, 1000}, {5500, 1100}, {5500, 1200},
        {5500, 1300}, {5500, 1500}, {5500, 2000}, {5500, 2100}, {5500, 2200}, {5500, 2300}, {5500, 2500}, {5500, 3000}, {5500, 3100}, {5500, 3200},
        {5500, 3300}, {5500, 3500}, {5500, 4000}, {5500, 4100}, {5500, 4200}, {5500, 4300}, {5500, 4500}, {5500, 5000}, {5500, 5100}, {5500, 5200},
        {5500, 5300}, {5500, 5500}, {6000, 100}, {6000, 200}, {6000, 300}, {6000, 500}, {6000, 1000}, {6000, 1200}, {6000, 1300},
        {6000, 1500}, {6000, 2000}, {6000, 2100}, {6000, 2200}, {6000, 2300}, {6000, 2500}, {6000, 3000}, {6000, 3100}, {6000, 3200}, {6000, 3500},
        {6000, 4000}, {6000, 4100}, {6000, 4200}, {6000, 4300}, {6000, 4500}, {6000, 5000}, {6000, 5100}, {6000, 5200}, {6000, 5300}, {6000, 5500},
        {6000, 6000} };*/
    std::vector<index_t> sizes = { {100, 100},  {200, 100}, {200, 200}, {300, 100}, {300, 200}, {300, 300}, {500, 100}, {500, 200}, {500, 500},
        {1000, 100}, {1000, 200}, {1000, 500}, {1000, 1000}, {1100, 300}, {1100, 500}, {1100, 1000}, {1100, 1100}, {1200, 300}, {1200, 500}, 
        {1200, 1000}, {1200, 1100}, {1200, 1200}, {1500, 1100}, {1500, 1200}, {1500, 1300}, {1500, 1500}, {2000, 1300}, {2000, 1500}, {2000, 2000}, 
        {2100, 1000}, {2500, 2100}, {2500, 2200}, {2500, 2500}, {3000, 2200}, {3200, 3000}, {3200, 3200}, {3300, 3000}, {3300, 3100}, {3300, 3300}, 
        {3500, 3500}, {4000, 3000}, {4000, 3100}, {4000, 4000}, {4500, 1000}, {4500, 3200}, {4500, 4500}, {5000, 3000}, {5000, 4500}, {5000, 5000}, 
        {5100, 3500}, {5500, 2000}, {6000, 1300}, {6000, 3100}, {6000, 5500}, {6000, 6000} };

    size_t n; //Размер столбцов матрицы A(mxn)
    size_t m; //Размер строк матрицы A(mxn)
    size_t block_size; //Размер блока матрицы (минимум 8)
    size_t max_threads = omp_get_max_threads();

    double time = 0.0;
    char in_path[100];
    char errors[200];
    char info[200];

    std::vector<compute_params> rrbjrs_times(sizes.size() * max_threads);
    std::vector<compute_params> coloshjac_times(sizes.size() * max_threads);

    for (size_t i = 0; i < sizes.size(); i++) {
        m = sizes[i].i;
        n = sizes[i].j;

        //Объявление структур и типов данных
        std::vector<double> A(m * n);
        std::vector<double> B(m * n);
        std::vector<double> S(n);
        std::vector<double> U(m * n, 0);
        std::vector<double> V(m * n, 0);
        //Data_matr - исходная матрица A
        //B_mat, S_vect - сингулярные числа матрицы
        //U_mat - левые сингулярные векторы
        //V_mat - правые сингулярные векторы
        matrix_t Data_matr = { &A[0], m, n };
        matrix_t B_mat = { &B[0], m, n };
        vector_t S_vect = { &S[0], n };
        matrix_t U_mat = { &U[0], m, n };
        matrix_t V_mat = { &V[0], m, n };

        sprintf_s(in_path, "./LocalData/in/%d_%d.in", m, n);
        try
        {
            //matrix_from_file(Data_matr, in_path);
            random_matrix(Data_matr);
        }
        catch (const std::exception&)
        {
            sprintf_s(errors, "[ERROR READ MATRIX] matrix from file ./LocalData/in/%d_%d.in", m, n);
            std::cout << errors << std::endl;
            continue;
        }

        for (size_t threads = 1; threads <= max_threads; threads++) {
            //rrbjrs - Блочный односторонний Якоби со стратегией выбора элементов Round Robin
            try
            {

                size_t rrbjrs_iters = rrbjrs(Data_matr, S_vect, U_mat, V_mat, threads, &time);
                if (rrbjrs_iters <= 0) {
                    sprintf_s(errors, "[WARNING] Alg 'rrbjrs' not computed: matrix %d %d, %d threads", m, n, threads);
                    std::cout << errors << std::endl;
                }
                else
                {
                    sprintf_s(info, "Compute alg 'rrbjrs': matrix %d %d, %d threads", m, n, threads);
                    std::cout << info << std::endl;
                    rrbjrs_times.push_back({ Data_matr.rows, Data_matr.cols, threads, rrbjrs_iters, time });
                }

            }
            catch (const std::exception&)
            {
                sprintf_s(errors, "[ERROR COMPUTATION] in 'rrbjrs': matrix %d %d, %d threads", m, n, threads);
                std::cout << errors << std::endl;
            }

            //coloshjac - Односторонний метод Якоби, элементы выбираются по столбцам последовательно
            try
            {
                size_t coloshjac_iters = coloshjac(Data_matr, S_vect, U_mat, V_mat, threads, &time);
                if (coloshjac_iters <= 0) {
                    sprintf_s(errors, "[WARNING] Alg 'coloshjac' not computed: matrix %d %d, %d threads", m, n, threads);
                    std::cout << errors << std::endl;
                }
                else
                {
                    sprintf_s(info, "Compute alg 'coloshjac': matrix %d %d, %d threads", m, n, threads);
                    std::cout << info << std::endl;
                    coloshjac_times.push_back({ Data_matr.rows, Data_matr.cols, threads, coloshjac_iters, time });
                }

            }
            catch (const std::exception&)
            {
                sprintf_s(errors, "[ERROR COMPUTATION] in 'coloshjac': matrix %d %d, %d threads", m, n, threads);
                std::cout << errors << std::endl;
            }
        }

    }

    //результаты методов записываются в файл
    // число строк, столбцов
    // число потоков
    // число итераций
    // время
    compute_params_to_file(rrbjrs_times, "./LocalData/out/tests/rrbjrs_times.to");
    compute_params_to_file(coloshjac_times, "./LocalData/out/tests/coloshjac_times.to");
    return 0;
}

void colbnsvd_test(matrix_t Data_matr, matrix_t B_mat, matrix_t U_mat, matrix_t V_mat, size_t block_size, size_t threads, double* time) {
    std::cout << "Singular decomposition of double square matrix by block two-sided NSVD Jacobi" << std::endl;
    //Блочное двустороннее сингулярное (SVD) разложение методом NSVD квадратной вещественной матрицы A
    //NSVD алгоритм описан в работе https://maths-people.anu.edu.au/~brent/pd/rpb080i.pdf (стр. 12)
    //Сейчас используется циклический перебор наддиагональных блоков матрицы
    size_t iterations = colbnsvd(Data_matr, B_mat, U_mat, V_mat, block_size, threads, time);

    matrix_to_file(B_mat, "./LocalData/out/SVD/NSVD/square/A.to");
    matrix_to_file(U_mat, "./LocalData/out/SVD/NSVD/square/V.to");
    matrix_to_file(V_mat, "./LocalData/out/SVD/NSVD/square/U.to");
}

void svd_test(matrix_t Data_matr, matrix_t B_mat, matrix_t U_mat, matrix_t V_mat, vector_t S_vect, size_t n, size_t block_size, size_t threads, double time)
{
    //Инициализация матрицы A
    matrix_from_file(Data_matr, "./LocalData/in/square.in");
    assert(Data_matr.rows == Data_matr.cols);
    assert(Data_matr.rows == B_mat.rows && Data_matr.cols == B_mat.cols);
    assert(Data_matr.rows == U_mat.rows && Data_matr.cols == U_mat.cols);
    assert(Data_matr.rows == V_mat.rows && Data_matr.cols == V_mat.cols);

    std::cout << "Singular decomposition of double square matrix by block one-sided BJRS Jacobi" << std::endl;
    //Блочное одностороннее сингулярное (SVD) разложение методом BJRS квадратной вещественной матрицы A
    //BJRS алгоритм описан в работе file:///C:/Users/reado/Downloads/A_Block_JRS_Algorithm_for_Highly_Parallel_Computat.pdf (стр. 10)
    size_t bjrs_iters = rrbjrs(Data_matr, S_vect, U_mat, V_mat, threads, &time);

    //Запись в файлы полученных данных
    //А - сингулярные числа матрицы
    //U - левые сингулярные векторы
    //V - правые сингулярные векторы
    vector_to_file(S_vect, "./LocalData/out/SVD/BJRS/square/A.to");
    matrix_to_file(U_mat, "./LocalData/out/SVD/BJRS/square/V.to");
    matrix_to_file(V_mat, "./LocalData/out/SVD/BJRS/square/U.to");

    std::cout << "Singular decomposition of double square matrix by block two-sided NSVD Jacobi" << std::endl;
    //Блочное двустороннее сингулярное (SVD) разложение методом NSVD квадратной вещественной матрицы A
    //NSVD алгоритм описан в работе https://maths-people.anu.edu.au/~brent/pd/rpb080i.pdf (стр. 12)
    //Сейчас используется циклический перебор наддиагональных блоков матрицы
    size_t iterations = colbnsvd(Data_matr, B_mat, U_mat, V_mat, block_size, threads, &time);

    //Запись в файлы полученных данных
    //А - сингулярные числа матрицы
    //U - левые сингулярные векторы
    //V - правые сингулярные векторы
    matrix_to_file(B_mat, "./LocalData/out/SVD/NSVD/square/A.to");
    matrix_to_file(U_mat, "./LocalData/out/SVD/NSVD/square/V.to");
    matrix_to_file(V_mat, "./LocalData/out/SVD/NSVD/square/U.to");

    std::cout << "Singular decomposition of double square matrix by Alglib sequence rmatrixsvd function" << std::endl;
    // Инициализация структур данных библиотеки Alglib
    alglib::real_2d_array A_alglib, U_alglib, V_alglib;
    alglib::real_1d_array W_alglib;
    A_alglib.attach_to_ptr(n, n, Data_matr.ptr);
    W_alglib.setlength(n);
    U_alglib.setlength(n, n);
    V_alglib.setlength(n, n);

    //Сингулярное (SVD) разложение функцией rmatrixsvd квадратной (или прямоугольной) вещественной матрицы A
    //Функция rmatrixsvd часть библиотеки Alglib
    //https://www.alglib.net/matrixops/general/svd.php
    bool rmatrixsvd_result = alglib::rmatrixsvd(A_alglib, n, n, 2, 2, 0, W_alglib, U_alglib, V_alglib);

    //Запись в файлы полученных данных
    U_mat.ptr = &U_alglib[0][0];
    V_mat.ptr = &V_alglib[0][0];
    S_vect = { &W_alglib[0], n };
    matrix_to_file(U_mat, "./LocalData/out/SVD/Alglib/V.to");
    matrix_to_file(V_mat, "./LocalData/out/SVD/Alglib/U.to");
    vector_to_file(S_vect, "./LocalData/out/SVD/Alglib/A.to");


    std::cout << "Singular decomposition of double square matrix by one-sided Hestenes Jacobi" << std::endl;
    //Одностороннее сингулярное (SVD) разложение по столбцам методом Hestenes one-sided Jacobi квадратной (или прямоугольной) 
    // вещественной матрицы A
    //Алгоритм описан в работе https://www.cs.utexas.edu/users/inderjit/public_papers/HLA_SVD.pdf (Algorithm 6, стр. 11)
    //Сейчас используется циклический перебор наддиагональных блоков матрицы
    iterations = coloshjac(Data_matr, S_vect, U_mat, V_mat, threads, &time);

    matrix_to_file(U_mat, "./LocalData/out/SVD/OHJac/square/V.to");
    matrix_to_file(V_mat, "./LocalData/out/SVD/OHJac/square/U.to");
    vector_to_file(S_vect, "./LocalData/out/SVD/OHJac/square/A.to");
}