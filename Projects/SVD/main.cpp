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
//#pragma GCC target("avx2")
int main(int argc, char* argv[])
{
	std::vector<index_t> sizes = { {512, 512}, {1024, 1024} };
	std::size_t n; //Размер столбцов матрицы A(mxn)
    std::size_t m; //Размер строк матрицы A(mxn)
    std::size_t block_size; //Размер блока матрицы (минимум 10)
	std::size_t max_threads = omp_get_max_threads();
	std::size_t start_thread = max_threads; // 1 ... max_threads
	std::string inf_message = "\nO2 ftree-vectorize, unaligned columns data store";
	bool vectorization = false;

    double time = 0.0;
    char in_path[100];
    char errors[500];
	char alg_errors[200];
    char info[200];

	std::vector<compute_params> coloshjac_times(sizes.size() * (max_threads - start_thread));
    std::vector<compute_params> prrbjrs_times(sizes.size() * (max_threads - start_thread));
	std::vector<compute_params> rrbnsvd_times(sizes.size() * (max_threads - start_thread));
	std::vector<compute_params> prrbnsvd_times(sizes.size() * (max_threads - start_thread));
	std::vector<compute_params> rrbnsvd_avx_times(sizes.size() * (max_threads - start_thread));
	std::vector<compute_params> prrbnsvd_avx_times(sizes.size() * (max_threads - start_thread));

	std::cout << "Singular Value Decomposition" << std::endl;
#ifdef PRRBJRS
#ifdef COLWISE
		std::cout << "\nParallel RRBJRS - 1D Blocked Jacobi with Round Robin pivoting, column wise storage" << std::endl;
#else
		std::cout << "\nParallel RRBJRS - 1D Blocked Jacobi with Round Robin pivoting, row wise storage" << std::endl;
#endif
		for (std::size_t i = 0; i < sizes.size(); i++) {
			m = sizes[i].i;
			n = sizes[i].j;
			//Объявление структур и типов данных
			std::vector<double> A(m * n);
			std::vector<double> B(m * n);
			std::vector<double> S(m * n);
			std::vector<double> U(m * n, 0);
			std::vector<double> V(m * n, 0);
			//Data_matr - исходная матрица A
			//B_mat, S_vect - сингулярные числа матрицы
			//U_mat - левые сингулярные векторы
			//V_mat - правые сингулярные векторы
#ifdef COLWISE
			matrix_t Data_matr = { &A[0], m, n, 'C' };
			matrix_t B_mat = { &B[0], m, n, 'C' };
			vector_t S_vect = { &S[0], n };
			matrix_t U_mat = { &U[0], m, m, 'C' };
			matrix_t V_mat = { &V[0], n, n, 'C' };
#else
			matrix_t Data_matr = { &A[0], m, n, 'R' };
			matrix_t B_mat = { &B[0], m, n, 'R' };
			vector_t S_vect = { &S[0], n };
			matrix_t U_mat = { &U[0], m, m, 'R' };
			matrix_t V_mat = { &V[0], n, n, 'R' };
#endif

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
		}
#endif

#ifdef COLBNSVD
#ifdef COLWISE
		std::cout << "\nCOLBNSVD - 2D Blocked Jacobi, column wise storage" << std::endl;
#else
		std::cout << "\nCOLBNSVD - 2D Blocked Jacobi, row wise storage" << std::endl;
#endif
		for (std::size_t i = 0; i < sizes.size(); i++) {
			m = sizes[i].i;
			n = sizes[i].j;
			//Объявление структур и типов данных
			std::vector<double> A(m * n);
			std::vector<double> B(m * n);
			std::vector<double> S(m * n);
			std::vector<double> U(m * n, 0);
			std::vector<double> V(m * n, 0);
			//Data_matr - исходная матрица A
			//B_mat, S_vect - сингулярные числа матрицы
			//U_mat - левые сингулярные векторы
			//V_mat - правые сингулярные векторы
#ifdef COLWISE
			matrix_t Data_matr = { &A[0], m, n, 'C' };
			matrix_t B_mat = { &B[0], m, n, 'C' };
			vector_t S_vect = { &S[0], n };
			matrix_t U_mat = { &U[0], m, m, 'C' };
			matrix_t V_mat = { &V[0], n, n, 'C' };
#else
			matrix_t Data_matr = { &A[0], m, n, 'R' };
			matrix_t B_mat = { &B[0], m, n, 'R' };
			vector_t S_vect = { &S[0], n };
			matrix_t U_mat = { &U[0], m, m, 'R' };
			matrix_t V_mat = { &V[0], n, n, 'R' };
#endif
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
			for (std::size_t block_size = start_thread; block_size <= max_threads; block_size++) {
				//colbnsvd - Блочный двусторонний Якоби с последовательным перебором по столбцам и заданным размером блока
				try
				{
					*(Alg_Errors_str.len) = 0;
					std::size_t rrbnsvd_iters;
					rrbnsvd_iters = colbnsvd(Data_matr, B_mat, U_mat, V_mat, block_size, vectorization, &time, Alg_Errors_str);
					if (*(Alg_Errors_str.len) > 0) {
						sprintf(errors, "[WARNING] not computed: %lu %lu, %lu block size. [%s]", m, n, block_size, Alg_Errors_str.ptr);
						std::cout << errors << std::endl;
					}
					else
					{
						sprintf(info, "[COMPUTED] %lu %lu, %lu block size", m, n, block_size);
						std::cout << info << std::endl;
						rrbnsvd_times.push_back({ Data_matr.rows, Data_matr.cols, block_size, rrbnsvd_iters, time });
					}

				}
				catch (const std::exception&)
				{
					sprintf(errors, "[ERROR] %lu %lu, %lu block size", m, n, block_size);
					std::cout << errors << std::endl;
				}
			}
		}
#endif

#ifdef PRRBNSVD
 #ifdef COLWISE
		std::cout << "\nParallel RRBNSVD - 2D Blocked Jacobi with Round Robin pivoting, column wise storage" << std::endl;
 #else
		std::cout << "\nParallel RRBNSVD - 2D Blocked Jacobi with Round Robin pivoting, row wise storage" << std::endl;
 #endif
		for (std::size_t i = 0; i < sizes.size(); i++) {
			m = sizes[i].i;
			n = sizes[i].j;
			//Объявление структур и типов данных
			std::vector<double> A(m* n);
			std::vector<double> B(m* n);
			std::vector<double> S(m* n);
			std::vector<double> U(m* n, 0);
			std::vector<double> V(m* n, 0);
			//Data_matr - исходная матрица A
			//B_mat, S_vect - сингулярные числа матрицы
			//U_mat - левые сингулярные векторы
			//V_mat - правые сингулярные векторы
#ifdef COLWISE
			matrix_t Data_matr = { &A[0], m, n, 'C' };
			matrix_t B_mat = { &B[0], m, n, 'C' };
			vector_t S_vect = { &S[0], n };
			matrix_t U_mat = { &U[0], m, m, 'C' };
			matrix_t V_mat = { &V[0], n, n, 'C' };
#else
			matrix_t Data_matr = { &A[0], m, n, 'R' };
			matrix_t B_mat = { &B[0], m, n, 'R' };
			vector_t S_vect = { &S[0], n };
			matrix_t U_mat = { &U[0], m, m, 'R' };
			matrix_t V_mat = { &V[0], n, n, 'R' };
#endif
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
		}
#endif

#ifdef COLBNSVD_AVX
		vectorization = true;
#ifdef COLWISE
		std::cout << "\nCOLBNSVD AVX - 2D Blocked Jacobi, column wise storage" << std::endl;
#else
		std::cout << "\nCOLBNSVD AVX - 2D Blocked Jacobi, row wise storage" << std::endl;
#endif
		for (std::size_t i = 0; i < sizes.size(); i++) {
			m = sizes[i].i;
			n = sizes[i].j;
			//Объявление структур и типов данных
			std::vector<double> A(m* n);
			std::vector<double> B(m* n);
			std::vector<double> S(m* n);
			std::vector<double> U(m* n, 0);
			std::vector<double> V(m* n, 0);
			//Data_matr - исходная матрица A
			//B_mat, S_vect - сингулярные числа матрицы
			//U_mat - левые сингулярные векторы
			//V_mat - правые сингулярные векторы
#ifdef COLWISE
			matrix_t Data_matr = { &A[0], m, n, 'C' };
			matrix_t B_mat = { &B[0], m, n, 'C' };
			vector_t S_vect = { &S[0], n };
			matrix_t U_mat = { &U[0], m, m, 'C' };
			matrix_t V_mat = { &V[0], n, n, 'C' };
#else
			matrix_t Data_matr = { &A[0], m, n, 'R' };
			matrix_t B_mat = { &B[0], m, n, 'R' };
			vector_t S_vect = { &S[0], n };
			matrix_t U_mat = { &U[0], m, m, 'R' };
			matrix_t V_mat = { &V[0], n, n, 'R' };
#endif
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
			for (std::size_t block_size = start_thread; block_size <= max_threads; block_size++) {
				//colbnsvd - Блочный двусторонний Якоби с последовательным перебором по столбцам и заданным размером блока
				try
				{
					*(Alg_Errors_str.len) = 0;
					std::size_t rrbnsvd_iters;
					rrbnsvd_iters = colbnsvd(Data_matr, B_mat, U_mat, V_mat, block_size, vectorization, &time, Alg_Errors_str);
					if (*(Alg_Errors_str.len) > 0) {
						sprintf(errors, "[WARNING] not computed: %lu %lu, %lu block size. [%s]", m, n, block_size, Alg_Errors_str.ptr);
						std::cout << errors << std::endl;
					}
					else
					{
						sprintf(info, "[COMPUTED] %lu %lu, %lu block size", m, n, block_size);
						std::cout << info << std::endl;
						rrbnsvd_times.push_back({ Data_matr.rows, Data_matr.cols, block_size, rrbnsvd_iters, time });
					}

				}
				catch (const std::exception&)
				{
					sprintf(errors, "[ERROR] %lu %lu, %lu block size", m, n, block_size);
					std::cout << errors << std::endl;
				}
			}
		}
		!vectorization;
#endif

#ifdef PRRBNSVD_AVX
		vectorization = true;
#ifdef COLWISE
		std::cout << "\nParallel RRBNSVD AVX - 2D Blocked Jacobi with Round Robin pivoting, column wise storage" << std::endl;
#else
		std::cout << "\nParallel RRBNSVD AVX - 2D Blocked Jacobi with Round Robin pivoting, row wise storage" << std::endl;
#endif
		for (std::size_t i = 0; i < sizes.size(); i++) {
			m = sizes[i].i;
			n = sizes[i].j;
			//Объявление структур и типов данных
			std::vector<double> A(m* n);
			std::vector<double> B(m* n);
			std::vector<double> S(m* n);
			std::vector<double> U(m* n, 0);
			std::vector<double> V(m* n, 0);
			//Data_matr - исходная матрица A
			//B_mat, S_vect - сингулярные числа матрицы
			//U_mat - левые сингулярные векторы
			//V_mat - правые сингулярные векторы
#ifdef COLWISE
			matrix_t Data_matr = { &A[0], m, n, 'C' };
			matrix_t B_mat = { &B[0], m, n, 'C' };
			vector_t S_vect = { &S[0], n };
			matrix_t U_mat = { &U[0], m, m, 'C' };
			matrix_t V_mat = { &V[0], n, n, 'C' };
#else
			matrix_t Data_matr = { &A[0], m, n, 'R' };
			matrix_t B_mat = { &B[0], m, n, 'R' };
			vector_t S_vect = { &S[0], n };
			matrix_t U_mat = { &U[0], m, m, 'R' };
			matrix_t V_mat = { &V[0], n, n, 'R' };
#endif
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
		}
		!vectorization;
#endif

    //результаты методов записываются в файл
    // число строк, столбцов
    // число потоков
    // число итераций
    // время
#ifdef PRRBJRS 
	compute_params_to_file(&inf_message[0], prrbjrs_times, "./TimeTests/prrbjrs_times.to");
#endif 
#ifdef COLBNSVD
	compute_params_to_file(&inf_message[0], rrbnsvd_times, "./TimeTests/colbnsvd_times.to");
#endif
#ifdef PRRBNSVD
	compute_params_to_file(&inf_message[0], prrbnsvd_times, "./TimeTests/prrbnsvd_times.to");
#endif
#ifdef COLBNSVD_AVX
	compute_params_to_file(&inf_message[0], rrbnsvd_avx_times, "./TimeTests/colbnsvd_avx_times.to");
#endif
#ifdef PRRBNSVD_AVX
	compute_params_to_file(&inf_message[0], prrbnsvd_avx_times, "./TimeTests/prrbnsvd_avx_times.to");
#endif
    return 0;
}

