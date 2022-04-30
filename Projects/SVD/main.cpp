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
	std::vector<index_t> sizes = { {800, 800} };
	std::size_t n; //Размер столбцов матрицы A(mxn)
    std::size_t m; //Размер строк матрицы A(mxn)
    std::size_t block_size; //Размер блока матрицы (минимум 10)
	std::size_t max_threads = omp_get_max_threads();
	std::size_t start_thread = max_threads; // 1 ... max_threads
	bool vectorization = false;

    double time = 0.0;
    char in_path[100];
    char errors[500];
	char alg_errors[200];
    char info[200];

	std::vector<compute_params> coloshjac_times(sizes.size() * max_threads);
    std::vector<compute_params> prrbjrs_times(sizes.size() * max_threads);
	std::vector<compute_params> rrbnsvd_times(sizes.size() * max_threads);
	std::vector<compute_params> prrbnsvd_times(sizes.size() * max_threads);
	std::vector<compute_params> rrbnsvd_avx_times(sizes.size() * max_threads);
	std::vector<compute_params> prrbnsvd_avx_times(sizes.size() * max_threads);

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
#ifdef PRRBJRS
		std::cout << "\nParallel RRBJRS - 1D Blocked Jacobi with Round Robin pivoting" << std::endl;
		for (std::size_t threads = start_thread; threads <= max_threads; threads++) {
			//rrbjrs - Блочный односторонний Якоби со стратегией выбора элементов Round Robin
			try
			{
				*(Alg_Errors_str.len) = 0;
				std::size_t rrbjrs_iters = rrbjrs(Data_matr, B_mat, S_vect, U_mat, V_mat, threads, &time, Alg_Errors_str);
				if (*(Alg_Errors_str.len) > 0) {
					sprintf(errors, "[WARNING] not computed: %lu %lu, %lu threads. [%s]", m, n, threads, Alg_Errors_str.ptr);
					std::cout << errors << std::endl;
				}
				else
				{
					sprintf(info, "[COMPUTED] %lu %lu, %lu threads", m, n, threads);
					std::cout << info << std::endl;
					prrbjrs_times.push_back({ Data_matr.rows, Data_matr.cols, threads, rrbjrs_iters, time });
				}

			}
			catch (const std::exception&)
			{
				sprintf(errors, "[ERROR] %lu %lu, %lu threads", m, n, threads);
				std::cout << errors << std::endl;
			}
		}
#endif

#ifdef RRBNSVD
		std::cout << "\nRRBNSVD - 2D Blocked Jacobi with Round Robin pivoting" << std::endl;
		for (std::size_t rr_pairs = start_thread; rr_pairs <= max_threads; rr_pairs++) {
			//rrbnsvd - Блочный двусторонний Якоби со стратегией выбора элементов Round Robin
			try
			{
				*(Alg_Errors_str.len) = 0;
				std::size_t rrbnsvd_iters;
				rrbnsvd_iters = rrbnsvd_seq(Data_matr, B_mat, U_mat, V_mat, rr_pairs, vectorization, &time, Alg_Errors_str);
				if (*(Alg_Errors_str.len) > 0) {
					sprintf(errors, "[WARNING] not computed: %lu %lu, %lu threads. [%s]", m, n, rr_pairs, Alg_Errors_str.ptr);
					std::cout << errors << std::endl;
				}
				else
				{
					sprintf(info, "[COMPUTED] %lu %lu, %lu threads", m, n, rr_pairs);
					std::cout << info << std::endl;
					rrbnsvd_times.push_back({ Data_matr.rows, Data_matr.cols, rr_pairs, rrbnsvd_iters, time });
				}

			}
			catch (const std::exception&)
			{
				sprintf(errors, "[ERROR] %lu %lu, %lu threads", m, n, rr_pairs);
				std::cout << errors << std::endl;
			}
		}
#endif

#ifdef PRRBNSVD
		std::cout << "\nParallel RRBNSVD - 2D Blocked Jacobi with Round Robin pivoting" << std::endl;
		for (std::size_t threads = start_thread; threads <= max_threads; threads++) {
			//rrbnsvd - Блочный двусторонний Якоби со стратегией выбора элементов Round Robin
			try
			{
				*(Alg_Errors_str.len) = 0;
				std::size_t rrbnsvd_iters;
				rrbnsvd_iters = rrbnsvd_parallel(Data_matr, B_mat, U_mat, V_mat, threads, vectorization, &time, Alg_Errors_str);
				if (*(Alg_Errors_str.len) > 0) {
					sprintf(errors, "[WARNING] not computed: %lu %lu, %lu threads. [%s]", m, n, threads, Alg_Errors_str.ptr);
					std::cout << errors << std::endl;
				}
				else
				{
					sprintf(info, "[COMPUTED] %lu %lu, %lu threads", m, n, threads);
					std::cout << info << std::endl;
					prrbnsvd_times.push_back({ Data_matr.rows, Data_matr.cols, threads, rrbnsvd_iters, time });
				}

			}
			catch (const std::exception&)
			{
				sprintf(errors, "[ERROR] %lu %lu, %lu threads", m, n, threads);
				std::cout << errors << std::endl;
			}
		}
#endif

#ifdef RRBNSVD_AVX
		vectorization = true;
		std::cout << "\nRRBNSVD AVX - 2D Blocked Jacobi with Round Robin pivoting" << std::endl;
		for (std::size_t rr_pairs = start_thread; rr_pairs <= max_threads; rr_pairs++) {
			//rrbnsvd - Блочный двусторонний Якоби со стратегией выбора элементов Round Robin
			try
			{
				*(Alg_Errors_str.len) = 0;
				std::size_t rrbnsvd_iters;
				rrbnsvd_iters = rrbnsvd_seq(Data_matr, B_mat, U_mat, V_mat, rr_pairs, vectorization, &time, Alg_Errors_str);
				if (*(Alg_Errors_str.len) > 0) {
					sprintf(errors, "[WARNING] not computed: %lu %lu, %lu threads. [%s]", m, n, rr_pairs, Alg_Errors_str.ptr);
					std::cout << errors << std::endl;
				}
				else
				{
					sprintf(info, "[COMPUTED] %lu %lu, %lu threads", m, n, rr_pairs);
					std::cout << info << std::endl;
					rrbnsvd_avx_times.push_back({ Data_matr.rows, Data_matr.cols, rr_pairs, rrbnsvd_iters, time });
				}

			}
			catch (const std::exception&)
			{
				sprintf(errors, "[ERROR] %lu %lu, %lu threads", m, n, rr_pairs);
				std::cout << errors << std::endl;
			}
		}
		!vectorization;
#endif

#ifdef PRRBNSVD_AVX
		vectorization = true;
		std::cout << "\nParallel RRBNSVD AVX - 2D Blocked Jacobi with Round Robin pivoting" << std::endl;
		for (std::size_t threads = start_thread; threads <= max_threads; threads++) {
			//rrbnsvd - Блочный двусторонний Якоби со стратегией выбора элементов Round Robin
			try
			{
				*(Alg_Errors_str.len) = 0;
				std::size_t rrbnsvd_iters;
				rrbnsvd_iters = rrbnsvd_parallel(Data_matr, B_mat, U_mat, V_mat, threads, vectorization, &time, Alg_Errors_str);
				if (*(Alg_Errors_str.len) > 0) {
					sprintf(errors, "[WARNING] not computed: %lu %lu, %lu threads. [%s]", m, n, threads, Alg_Errors_str.ptr);
					std::cout << errors << std::endl;
				}
				else
				{
					sprintf(info, "[COMPUTED] %lu %lu, %lu threads", m, n, threads);
					std::cout << info << std::endl;
					prrbnsvd_avx_times.push_back({ Data_matr.rows, Data_matr.cols, threads, rrbnsvd_iters, time });
				}

			}
			catch (const std::exception&)
			{
				sprintf(errors, "[ERROR] %lu %lu, %lu threads", m, n, threads);
				std::cout << errors << std::endl;
			}
		}
		!vectorization;
#endif
    }

    //результаты методов записываются в файл
    // число строк, столбцов
    // число потоков
    // число итераций
    // время
#ifdef PRRBJRS 
	compute_params_to_file(prrbjrs_times, "./TimeTests/prrbjrs_times.to");
#endif 
#ifdef RRBNSVD
	compute_params_to_file(rrbnsvd_times, "./TimeTests/rrbnsvd_times.to");
#endif
#ifdef PRRBNSVD
	compute_params_to_file(prrbnsvd_times, "./TimeTests/prrbnsvd_times.to");
#endif
#ifdef RRBNSVD_AVX
	compute_params_to_file(rrbnsvd_avx_times, "./TimeTests/rrbnsvd_avx_times.to");
#endif
#ifdef PRRBNSVD_AVX
	compute_params_to_file(prrbnsvd_avx_times, "./TimeTests/prrbnsvd_avx_times.to");
#endif
    return 0;
}

