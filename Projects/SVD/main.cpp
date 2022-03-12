#include <algorithm>
#include <omp.h>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include <vector>
#include "src/utils/types.hpp"
#include "src/utils/util.hpp"
#include "src/svd/one-sided/svd.hpp"

void svd_test(matrix_t Data_matr, matrix_t B_mat, matrix_t U_mat, matrix_t V_mat, vector_t S_vect, size_t n, size_t block_size, size_t threads, double time);

int main(int argc, char* argv[])
{
    std::vector<index_t> sizes = { {100, 100},  {200, 100}, {200, 200}, {300, 100}, {300, 200}, {300, 300}, {500, 100}, {500, 200}, {500, 300}, {500, 500} };

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

void svd_test(matrix_t Data_matr, matrix_t B_mat, matrix_t U_mat, matrix_t V_mat, vector_t S_vect, size_t n, size_t block_size, size_t threads, double time)
{
	size_t iterations;
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