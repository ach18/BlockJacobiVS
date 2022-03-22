#include <algorithm>
#include <cstdio>
#include <cassert>
#include <iostream>
//#include <vector>
#include <omp.h>
#ifdef MKL_TEST
#include <mkl.h>
#endif //
#include "src/utils/types.hpp"
#include "src/utils/util.hpp"
#include "src/utils/matrix.hpp"
#include "src/svd/one-sided/svd.hpp"
#include "src/svd/two-sided/svd.hpp"

int main(int argc, char* argv[])
{
    std::vector<index_t> sizes = { {100, 100},  {200, 100}, {200, 200}, {300, 100}, {300, 200}, {300, 300}, {500, 100}, {500, 200}, {500, 300}, {500, 500} };
    std::size_t n; //Размер столбцов матрицы A(mxn)
    std::size_t m; //Размер строк матрицы A(mxn)
    std::size_t block_size; //Размер блока матрицы (минимум 8)
    std::size_t max_threads = 20;

    double time = 0.0;
    char in_path[100];
    char errors[500];
	char alg_errors[200];
    char info[200];

    std::vector<compute_params> rrbjrs_times(sizes.size() * max_threads);
    std::vector<compute_params> coloshjac_times(sizes.size() * max_threads);
	std::vector<compute_params> mkl_dgesvj_times(sizes.size() * max_threads);
	std::vector<compute_params> rrbnsvd_times(sizes.size() * max_threads);

    for (std::size_t i = 0; i < sizes.size(); i++) {
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
		for (std::size_t threads = 1; threads <= max_threads; threads++) {
			//rrbjrs - Блочный односторонний Якоби со стратегией выбора элементов Round Robin
			try
			{
				*(Alg_Errors_str.len) = 0;
				std::size_t rrbjrs_iters = rrbjrs(Data_matr, S_vect, U_mat, V_mat, threads, &time, Alg_Errors_str);
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
#ifdef MKL_TEST
		for (std::size_t threads = 1; threads <= max_threads; threads++) {
			//dgesvj - MKL односторонний Якоби со стратегией выбора элементов перестановки столбцов (deRijk98)
			//void DGESVJ(const char* joba, const char* jobu, const char* jobv,
			//	const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda,
			//	double* sva, const MKL_INT* mv, double* v, const MKL_INT* ldv,
			//	double* work, const MKL_INT* lwork, MKL_INT* info);
			try
			{
				mkl_set_num_threads(threads);
				double mkl_t1, mkl_t2;	//замер времени

				//MKL параметры
				char joba[] = "G"; //Указывает, что входящая матрица A(mxn) имееет общий вид (m >= n)
				char jobu[] = "U"; //Указывает, что ненулевые левые сингулярные векторы будут вычислены, и сохранены в матрице A
				char jobv[] = "V"; //Указывает, что правые сингулярные векторы будут вычислены, и сохранены в матрице V
				
				MKL_INT lda = Data_matr.rows;       //Ведущая "ось" матрицы A (строки)
				MKL_INT ldv = Data_matr.cols;       //Ведущая "ось" матрицы V (столбцы)
				MKL_INT mv = 0;						//Применяется если jobv = "A". Не используется.
				MKL_INT lwork = Data_matr.rows + Data_matr.cols;
				MKL_INT dgesvj_info = -1;			//Значение 0 соответствует успешному вычислению, -1 ошибка. По умолчанию -1.
				std::vector<double> work(lwork);

				std::vector<double> A_mkl(m * n);
				matrix_t MKL_matr = { &A_mkl[0], m, n }; //Инициализация копии исходной матрицы Data_matr (т.к. функция MKL изменяет её).
				matrix_copy(MKL_matr, Data_matr);

				mkl_t1 = omp_get_wtime();
				dgesvj(joba, jobu, jobv, &MKL_matr.rows, &MKL_matr.cols, MKL_matr.ptr, &lda, S_vect.ptr, &mv, V_mat.ptr, &ldv, &work[0], &lwork, &dgesvj_info);
				mkl_t2 = omp_get_wtime();
				time = mkl_t2 - mkl_t1;

				if (dgesvj_info != 0) {
					sprintf(errors, "[WARNING] Alg MKL 'dgesvj' not computed: matrix %lu %lu, %lu threads", m, n, threads);
					std::cout << errors << std::endl;
				}
				else
				{
					sprintf(info, "[COMPUTED] Alg MKL 'dgesvj': matrix %lu %lu, %lu threads", m, n, threads);
					std::cout << info << std::endl;
					mkl_dgesvj_times.push_back({ Data_matr.rows, Data_matr.cols, threads, 0, time });
				}

			}
			catch (const std::exception&)
			{
				sprintf(errors, "[ERROR] In MKL 'dgesvj': matrix %lu %lu, %lu threads", m, n, threads);
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
	compute_params_to_file(mkl_dgesvj_times, "./TimeTests/mkl_dgesvj_times.to");
    return 0;
}

