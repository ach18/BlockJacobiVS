#include <algorithm>
#include <cstdio>
#include <cassert>
#include <iostream>
#include <omp.h>
#include "src/utils/types.hpp"
#include "src/utils/util.hpp"
#include "src/utils/matrix.hpp"
#include "src/svd/one-sided/svd.hpp"
#include "src/svd/two-sided/svd.hpp"

int main(int argc, char* argv[])
{
	std::vector<index_t> sizes = { {100, 100}, {200, 100}, {200, 200}, {300, 100}, {300, 200}, {300, 300}, {500, 100}, {500, 200}, {500, 300}, {500, 500} };
	std::size_t n; //Размер столбцов матрицы A(mxn)
    std::size_t m; //Размер строк матрицы A(mxn)
    std::size_t block_size; //Размер блока матрицы (минимум 10)
	std::size_t max_threads = omp_get_num_procs();

    double time = 0.0;
    char in_path[100];
    char errors[500];
	char alg_errors[200];
    char info[200];

    std::vector<compute_params> rrbjrs_times(sizes.size() * max_threads);
    std::vector<compute_params> coloshjac_times(sizes.size() * max_threads);
	std::vector<compute_params> rrbnsvd_times(sizes.size() * max_threads);

	std::cout << "Singular Value Decomposition" << std::endl;
    for (std::size_t i = 0; i < sizes.size(); i++) {
        m = sizes[i].i;
        n = sizes[i].j;
        //Объявление структур и типов данных
        std::vector<double> A(m * n);
        std::vector<double> B(m * n);
        std::vector<double> S(n);
        std::vector<double> U(m * m, 0);
        std::vector<double> V(n * n, 0);
        //Data_matr - исходная матрица A
        //B_mat, S_vect - сингулярные числа матрицы
        //U_mat - левые сингулярные векторы
        //V_mat - правые сингулярные векторы
        matrix_t Data_matr = { &A[0], m, n };
        matrix_t B_mat = { &B[0], m, n };
        vector_t S_vect = { &S[0], n };
        matrix_t U_mat = { &U[0], m, m };
        matrix_t V_mat = { &V[0], n, n };

		//Alg_Errors_str - структура со строкой ошибок выполнения алгоритма
		//Alg_Errors_len - длина строки ошибок алгоритма
		std::size_t Alg_Errors_len = 0;
		string_t Alg_Errors_str = { alg_errors, &Alg_Errors_len };

        try
        {
            random_matrix(Data_matr);
        }
        catch (const std::exception&)
        {
            sprintf(errors, "[ERROR] Can't create %lu_%lu matrix", m, n);
            std::cout << errors << std::endl;
            continue;
        }
#ifdef RRBJRS_TEST
		std::cout << "Alg RRBJRS - 1D Blocked Jacobi with Round Robin pivoting" << std::endl;
		for (std::size_t threads = 1; threads <= max_threads; threads++) {
			//rrbjrs - Блочный односторонний Якоби со стратегией выбора элементов Round Robin
			try
			{
				*(Alg_Errors_str.len) = 0;
				std::size_t rrbjrs_iters = rrbjrs(Data_matr, B_mat, S_vect, U_mat, V_mat, threads, &time, Alg_Errors_str);
				if (*(Alg_Errors_str.len) > 0) {
					sprintf(errors, "[WARNING] Alg 'rrbjrs' not computed: matrix %lu %lu, %lu threads. [%s]", m, n, threads, Alg_Errors_str.ptr);
					std::cout << errors << std::endl;
				}
				else
				{
					sprintf(info, "[COMPUTED] Alg 'rrbjrs': matrix %lu %lu, %lu threads", m, n, threads);
					std::cout << info << std::endl;
					rrbjrs_times.push_back({ Data_matr.rows, Data_matr.cols, threads, rrbjrs_iters, time });
				}

			}
			catch (const std::exception&)
			{
				sprintf(errors, "[ERROR] In 'rrbjrs': matrix %lu %lu, %lu threads", m, n, threads);
				std::cout << errors << std::endl;
			}
		}
#endif
#ifdef RRBNSVD_TEST
		std::cout << "Alg RRBNSVD - 2D Blocked Jacobi with Round Robin pivoting" << std::endl;
		for (std::size_t threads = 1; threads <= max_threads; threads++) {
			//rrbnsvd - Блочный двусторонний Якоби со стратегией выбора элементов Round Robin
			try
			{
				*(Alg_Errors_str.len) = 0;
				std::size_t rrbnsvd_iters = rrbnsvd(Data_matr, B_mat, U_mat, V_mat, threads, &time, Alg_Errors_str);
				if (*(Alg_Errors_str.len) > 0) {
					sprintf(errors, "[WARNING] Alg 'rrbnsvd' not computed: matrix %lu %lu, %lu threads. [%s]", m, n, threads, Alg_Errors_str.ptr);
					std::cout << errors << std::endl;
				}
				else
				{
					sprintf(info, "[COMPUTED] Alg 'rrbnsvd': matrix %lu %lu, %lu threads", m, n, threads);
					std::cout << info << std::endl;
					rrbnsvd_times.push_back({ Data_matr.rows, Data_matr.cols, threads, rrbnsvd_iters, time });
				}

			}
			catch (const std::exception&)
			{
				sprintf(errors, "[ERROR] In 'rrbnsvd': matrix %lu %lu, %lu threads", m, n, threads);
				std::cout << errors << std::endl;
			}
		}
#endif

    }

    //результаты методов записываются в файл
    // число строк, столбцов
    // число потоков
    // число итераций
    // время
    compute_params_to_file(rrbjrs_times, "./TimeTests/rrbjrs_times.to");
    compute_params_to_file(coloshjac_times, "./TimeTests/coloshjac_times.to");
	compute_params_to_file(rrbnsvd_times, "./TimeTests/rrbnsvd_times.to");
    return 0;
}

