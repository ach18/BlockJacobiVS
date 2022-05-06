#include <omp.h>
#include <math.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include "../../utils/block.hpp"
#include "../../utils/matrix.hpp"
#include "nsvd.hpp"
#include "svd.hpp"
#include "svd_subprocedure.hpp"
#include "../../utils/types.hpp"
#include "../../utils/util.hpp"
#include <immintrin.h>

/**
 * @param matrix_t Amat квадратная матрица A
 * @param matrix_t Bmat матрица A после деления на блоки
 * @param matrix_t Umat матрица левых сингулярных векторов
 * @param matrix_t Vmat матрица правых сингулярных векторов
 * @param std::size_t block_size размер блока матрицы Bmat
 * @return std::size_t sweeps число разверток методом Якоби
 **/
std::size_t colbnsvd(struct matrix_t Amat, struct matrix_t Bmat, struct matrix_t Umat, struct matrix_t Vmat,
    std::size_t block_size, std::size_t ThreadsNum, double* Time, struct string_t errors) {

    std::size_t sweeps = 0;  //число повторений цикла развертки
    std::size_t block_iter = 0;
    const double tol = 1e-15;  //точность предела сходимости
    const std::size_t n = Amat.rows; //размер матрицы
    double norm = 0.0;      //норма Фробениуса матрицы B
    double off_norm = 0.0;  //норма Фробениуса только недиагональных элементов матрицы B
	double t1, t2;              //замер времени

    matrix_copy(Bmat, Amat); //инициализация рабочей матрицы B
    //матрицы U и V инициализируются как единичные матрицы
    matrix_identity(Umat);
    matrix_identity(Vmat);
    matrix_frobenius(Bmat, &norm, &off_norm);

    const std::size_t n_blocks = n / block_size; //число блоков вдоль строки/столбца

    //если строка/столбец состоят из двух блоков, лучше вычислить SVD классическим не блочным Якоби
    if (n <= 2 * block_size) {
        std::size_t block_iters = svd_subprocedure(Bmat, Umat, Vmat);
        return block_iters;
    }

    //общее число элементов всех блоков строки/столбца должно быть равно размерности матрицы
	if (n_blocks * block_size != n)
		return 0;

    //выделение памяти для хранения блоков матриц B U V M1 M2
    std::vector<double> Bblock(4 * block_size * block_size);
    std::vector<double> Ublock(4 * block_size * block_size);
    std::vector<double> Vblock(4 * block_size * block_size);
    //M1 M2 хранят промежуточные значения вычислений
    std::vector<double> M1(block_size * block_size);
    std::vector<double> M2(block_size * block_size);

    //хранение блоков в виде структур matrix_t
    matrix_t Bblockmat = { &Bblock[0], 2 * block_size, 2 * block_size };
    matrix_t Ublockmat = { &Ublock[0], 2 * block_size, 2 * block_size };
    matrix_t Vblockmat = { &Vblock[0], 2 * block_size, 2 * block_size };
    matrix_t M1mat = { &M1[0], block_size, block_size };
    matrix_t M2mat = { &M2[0], block_size, block_size };

	//omp_set_num_threads(ThreadsNum);
	t1 = omp_get_wtime();
	bool converged = sqrt(off_norm) > tol * sqrt(norm);

    //основной цикл развретки, он продолжается пока 
    while (converged) {
        //цикл обхода над/поддиагональных блоков
        for (std::size_t i_block = 0; i_block < n_blocks - 1; ++i_block) {
            for (std::size_t j_block = i_block + 1; j_block < n_blocks; ++j_block) {
                //в Bblockmat копируются блоки с индексами ii ij ji jj из матрицы Bmat
                copy_block(Bmat, i_block, i_block, Bblockmat, 0, 0, block_size);
                copy_block(Bmat, i_block, j_block, Bblockmat, 0, 1, block_size);
                copy_block(Bmat, j_block, i_block, Bblockmat, 1, 0, block_size);
                copy_block(Bmat, j_block, j_block, Bblockmat, 1, 1, block_size);

                //вычисление SVD разложения циклическим методом Якоби над блоком Bblockmat
				//Bblockmat используется только в этой процедуре для упрощения обращения к индексам внутри
                block_iter += svd_subprocedure(Bblockmat, Ublockmat, Vblockmat);

                //транспонировать блок матрицы U 
                matrix_transpose(Ublockmat, Ublockmat);

				//выполнение операции U^T*B*V, т.е. обновление строк и столбцов в двух циклах
				//элементами U, V выступают матрицы текущей подпроблемы, B - исходная матрица 
				//при этом матрица B считается диагональной
                //обновление блоков исходной матрицы B по строкам i j с помощью U
                for (std::size_t k_block = 0; k_block < n_blocks; ++k_block) {
                    mult_block(Ublockmat, 0, 0, Bmat, i_block, k_block, M1mat, 0, 0, block_size);
                    mult_block(Ublockmat, 0, 1, Bmat, j_block, k_block, M2mat, 0, 0, block_size);
                    matrix_add(M1mat, M2mat, M2mat);
                    mult_block(Ublockmat, 1, 0, Bmat, i_block, k_block, M1mat, 0, 0, block_size);
                    copy_block(M2mat, 0, 0, Bmat, i_block, k_block, block_size);
                    mult_block(Ublockmat, 1, 1, Bmat, j_block, k_block, M2mat, 0, 0, block_size);
                    matrix_add(M1mat, M2mat, M2mat);
                    copy_block(M2mat, 0, 0, Bmat, j_block, k_block, block_size);
                }

                //обновление блоков исходной матрицы B по столбцам i j с помощью V
                for (std::size_t k_block = 0; k_block < n_blocks; ++k_block) {
                    mult_block(Bmat, k_block, i_block, Vblockmat, 0, 0, M1mat, 0, 0, block_size);
                    mult_block(Bmat, k_block, j_block, Vblockmat, 1, 0, M2mat, 0, 0, block_size);
                    matrix_add(M1mat, M2mat, M2mat);
                    mult_block(Bmat, k_block, i_block, Vblockmat, 0, 1, M1mat, 0, 0, block_size);
                    copy_block(M2mat, 0, 0, Bmat, k_block, i_block, block_size);
                    mult_block(Bmat, k_block, j_block, Vblockmat, 1, 1, M2mat, 0, 0, block_size);
                    matrix_add(M1mat, M2mat, M2mat);
                    copy_block(M2mat, 0, 0, Bmat, k_block, j_block, block_size);
                }

                //возвращение транспонированного блока U^T в исходное состояние U 
                matrix_transpose(Ublockmat, Ublockmat);

                //обновление блоков исходной матрицы U по столбцам i j путем умножения на матрицу подпроблемы Ublockmat
                for (std::size_t k_block = 0; k_block < n_blocks; ++k_block) {
                    mult_block(Umat, k_block, i_block, Ublockmat, 0, 0, M1mat, 0, 0, block_size);
                    mult_block(Umat, k_block, j_block, Ublockmat, 1, 0, M2mat, 0, 0, block_size);
                    matrix_add(M1mat, M2mat, M2mat);
                    mult_block(Umat, k_block, i_block, Ublockmat, 0, 1, M1mat, 0, 0, block_size);
                    copy_block(M2mat, 0, 0, Umat, k_block, i_block, block_size);
                    mult_block(Umat, k_block, j_block, Ublockmat, 1, 1, M2mat, 0, 0, block_size);
                    matrix_add(M1mat, M2mat, M2mat);
                    copy_block(M2mat, 0, 0, Umat, k_block, j_block, block_size);
                }

                //обновление блоков исходной матрицы V по столбцам i j путем умножения на матрицу подпроблемы Vblockmat
                for (std::size_t k_block = 0; k_block < n_blocks; ++k_block) {
                    mult_block(Vmat, k_block, i_block, Vblockmat, 0, 0, M1mat, 0, 0, block_size);
                    mult_block(Vmat, k_block, j_block, Vblockmat, 1, 0, M2mat, 0, 0, block_size);
                    matrix_add(M1mat, M2mat, M2mat);
                    mult_block(Vmat, k_block, i_block, Vblockmat, 0, 1, M1mat, 0, 0, block_size);
                    copy_block(M2mat, 0, 0, Vmat, k_block, i_block, block_size);
                    mult_block(Vmat, k_block, j_block, Vblockmat, 1, 1, M2mat, 0, 0, block_size);
                    matrix_add(M1mat, M2mat, M2mat);
                    copy_block(M2mat, 0, 0, Vmat, k_block, j_block, block_size);
                }
            }
        }

        matrix_frobenius(Bmat, &norm, &off_norm);
		converged = sqrt(off_norm) > tol * sqrt(norm);
		if (sweeps > 30)
			converged = false;
		else
	        sweeps++;
    }

	t2 = omp_get_wtime();

	if (sweeps > 30)
		return 0;

	*Time = t2 - t1;
    return sweeps;
}

/**
 * @param matrix_t Amat квадратная матрица A
 * @param matrix_t Bmat матрица A после деления на блоки
 * @param matrix_t Umat матрица левых сингулярных векторов
 * @param matrix_t Vmat матрица правых сингулярных векторов
 * @param std::size_t block_size размер блока матрицы Bmat
 * @return std::size_t sweeps число разверток методом Якоби
 **/
std::size_t rrbnsvd_parallel(struct matrix_t Amat, struct matrix_t Bmat, struct matrix_t Umat, struct matrix_t Vmat,
	std::size_t ThreadsNum, bool vectorization, double* Time, struct string_t errors) {

	std::size_t subproblem_blocks = 4; //число блоков, которое формирует подзадачу
	std::size_t max_sweeps = 40; //максимальное число повторения итераций
	std::size_t sweeps = 0;  //число повторений цикла развертки
	const double tol = 1e-10;  //точность предела сходимости
	std::size_t n = Amat.rows; //размер матрицы
	double norm = 0.0;      //норма Фробениуса матрицы B
	double off_norm = 0.0;  //норма Фробениуса только недиагональных элементов матрицы B
	double t1, t2;              //замер времени

	matrix_copy(Bmat, Amat); //инициализация рабочей матрицы B
	//матрицы U и V инициализируются как единичные матрицы
	matrix_identity(Umat);
	matrix_identity(Vmat);
	matrix_frobenius(Bmat, &norm, &off_norm);

	//Входная матрица должна быть квадратной
	if (Amat.cols != Amat.rows) {
		*errors.len = sprintf(errors.ptr, "matrix must be square");
		return 0;
	}

	std::size_t rr_pairs = ThreadsNum; //число пар индексов (по одному блоку на один поток)
	std::size_t n_blocks = rr_pairs * 2; //число блоков вдоль строки/столбца
	std::size_t block_size = n / n_blocks; //число элементов на один блок

	//общее число элементов всех блоков строки/столбца должно быть равно размерности матрицы
	if (n_blocks * block_size != n) {
		*errors.len = sprintf(errors.ptr, "matrix must be correctly divided into blocks");
		return 0;
	}

	//если строка/столбец состоят из двух блоков, лучше вычислить SVD классическим не блочным Якоби
	if (n <= 2 * block_size) {
		t1 = omp_get_wtime();
		std::size_t block_iters;
		if (vectorization)
			block_iters = svd_subprocedure_vectorized(Bmat, Umat, Vmat);
		else
			block_iters = svd_subprocedure(Bmat, Umat, Vmat);
		t2 = omp_get_wtime();
		*Time = t2 - t1;

		if (block_iters == 0) {
			*errors.len = sprintf(errors.ptr, "algorithm did not converge after %lu sweeps", block_iters);
			return 0;
		}
		return block_iters;
	}

	std::vector<std::size_t> up(rr_pairs); //массив хранит индекс i блока для одного потока
	std::vector<std::size_t> dn(rr_pairs); //массив хранит индекс j блока для одного потока

	//инициализация массива пар индексов вращений 
	//нечетными (up[]) и четными (dn[]) числами
	for (std::size_t i = 0; i < rr_pairs; i++) {
		up[i] = (2 * i);
		dn[i] = (2 * i) + 1;
	}

	if(ThreadsNum > 1)
		omp_set_num_threads(ThreadsNum);
	else {
		*errors.len = sprintf(errors.ptr, "minimum 2 threads needed");
		return 0;
	}
	t1 = omp_get_wtime();
	bool converged = sqrt(off_norm) > tol * sqrt(norm);

	//основной цикл развретки, он продолжается пока 
	while (converged) {
		//цикл обхода над/поддиагональных блоков
		for (std::size_t iteration = 0; iteration < n_blocks - 1; ++iteration) {
//Выполнение параллельных итераций цикла заменяется на выполнение в параллельном регионе. Один поток фактически параллельно
//выполняет одну итерацию цикла
#pragma omp parallel shared(Bmat, Umat, Vmat, block_size, up, dn, n_blocks, subproblem_blocks) if(ThreadsNum > 1)
{
			////выделение памяти для хранения блоков матриц B U V M1 M2
			std::vector<double> Bblock(subproblem_blocks * block_size * block_size);
			std::vector<double> Ublock(subproblem_blocks * block_size * block_size);
			std::vector<double> Vblock(subproblem_blocks * block_size * block_size);
			//выделение памяти для хранения промежуточных значений умножения блоков
			std::vector<double> M1(block_size * block_size);
			std::vector<double> M2(block_size * block_size);

			//хранение блоков в виде структур matrix_t
			matrix_t Bblockmat = { &Bblock[0], 2 * block_size, 2 * block_size, Bmat.storage };
			matrix_t Ublockmat = { &Ublock[0], 2 * block_size, 2 * block_size, Umat.storage };
			matrix_t Vblockmat = { &Vblock[0], 2 * block_size, 2 * block_size, Vmat.storage };
			matrix_t M1mat = { &M1[0], block_size, block_size, Bmat.storage };
			matrix_t M2mat = { &M2[0], block_size, block_size, Bmat.storage };

			int rr_pair = omp_get_thread_num();
			std::size_t i_block = up[rr_pair];
			std::size_t j_block = dn[rr_pair];
			if (i_block > j_block)
				std::swap(i_block, j_block);

			//в Bblockmat копируются блоки с индексами ii ij ji jj из матрицы Bmat
			copy_subproblem_blocks(Bmat, i_block, j_block, Bblockmat, 0, 1, block_size);

			//вычисление SVD разложения циклическим методом Якоби над блоком Bblockmat
			//Bblockmat используется только в этой процедуре для упрощения обращения к индексам внутри
			if(vectorization)
				svd_subprocedure_vectorized(Bblockmat, Ublockmat, Vblockmat);
			else
				svd_subprocedure(Bblockmat, Ublockmat, Vblockmat);

			//транспонировать блок матрицы U 
			matrix_transpose(Ublockmat, Ublockmat);

			//выполнение операции U^T*B*V, т.е. обновление строк и столбцов в двух циклах
			//элементами U, V выступают матрицы текущей подпроблемы, B - исходная матрица 
			//при этом матрица B считается диагональной
			//обновление блоков исходной матрицы B по строкам i j с помощью U
			for (std::size_t k_block = 0; k_block < n_blocks; ++k_block) {
				mult_block(Ublockmat, 0, 0, Bmat, i_block, k_block, M2mat, 0, 0, block_size);
				mult_add_block(Ublockmat, 0, 1, Bmat, j_block, k_block, M2mat, 0, 0, block_size);
				mult_block(Ublockmat, 1, 0, Bmat, i_block, k_block, M1mat, 0, 0, block_size);
				copy_block(M2mat, 0, 0, Bmat, i_block, k_block, block_size);
				mult_add_block(Ublockmat, 1, 1, Bmat, j_block, k_block, M1mat, 0, 0, block_size);
				copy_block(M1mat, 0, 0, Bmat, j_block, k_block, block_size);
			}
	#pragma omp barrier
			//обновление блоков исходной матрицы B по столбцам i j с помощью V
			for (std::size_t k_block = 0; k_block < n_blocks; ++k_block) {
				mult_block(Bmat, k_block, i_block, Vblockmat, 0, 0, M2mat, 0, 0, block_size);
				mult_add_block(Bmat, k_block, j_block, Vblockmat, 1, 0, M2mat, 0, 0, block_size);
				mult_block(Bmat, k_block, i_block, Vblockmat, 0, 1, M1mat, 0, 0, block_size);
				copy_block(M2mat, 0, 0, Bmat, k_block, i_block, block_size);
				mult_add_block(Bmat, k_block, j_block, Vblockmat, 1, 1, M1mat, 0, 0, block_size);
				copy_block(M1mat, 0, 0, Bmat, k_block, j_block, block_size);
			}

			//возвращение транспонированного блока U^T в исходное состояние U 
			matrix_transpose(Ublockmat, Ublockmat);
			//обновление блоков исходной матрицы U по столбцам i j путем умножения на матрицу подпроблемы Ublockmat
			for (std::size_t k_block = 0; k_block < n_blocks; ++k_block) {
				mult_block(Umat, k_block, i_block, Ublockmat, 0, 0, M2mat, 0, 0, block_size);
				mult_add_block(Umat, k_block, j_block, Ublockmat, 1, 0, M2mat, 0, 0, block_size);
				mult_block(Umat, k_block, i_block, Ublockmat, 0, 1, M1mat, 0, 0, block_size);
				copy_block(M2mat, 0, 0, Umat, k_block, i_block, block_size);
				mult_add_block(Umat, k_block, j_block, Ublockmat, 1, 1, M1mat, 0, 0, block_size);
				copy_block(M1mat, 0, 0, Umat, k_block, j_block, block_size);
			}
			//обновление блоков исходной матрицы V по столбцам i j путем умножения на матрицу подпроблемы Vblockmat
			for (std::size_t k_block = 0; k_block < n_blocks; ++k_block) {
				mult_block(Vmat, k_block, i_block, Vblockmat, 0, 0, M2mat, 0, 0, block_size);
				mult_add_block(Vmat, k_block, j_block, Vblockmat, 1, 0, M2mat, 0, 0, block_size);
				mult_block(Vmat, k_block, i_block, Vblockmat, 0, 1, M1mat, 0, 0, block_size);
				copy_block(M2mat, 0, 0, Vmat, k_block, i_block, block_size);
				mult_add_block(Vmat, k_block, j_block, Vblockmat, 1, 1, M1mat, 0, 0, block_size);
				copy_block(M1mat, 0, 0, Vmat, k_block, j_block, block_size);
			}
}
			round_robin(&up[0], &dn[0], rr_pairs);
		}

		matrix_frobenius(Bmat, &norm, &off_norm);
		converged = sqrt(off_norm) > tol * sqrt(norm);
		if (sweeps > max_sweeps)
			converged = false;
		else
			sweeps++;
	}

	t2 = omp_get_wtime();

	if (sweeps > max_sweeps) {
		*errors.len = sprintf(errors.ptr, "algorithm did not converge after %lu sweeps", sweeps);
		return 0;
	}

	*Time = t2 - t1;
	return sweeps;
}

/**
 * @param matrix_t Amat квадратная матрица A
 * @param matrix_t Bmat матрица A после деления на блоки
 * @param matrix_t Umat матрица левых сингулярных векторов
 * @param matrix_t Vmat матрица правых сингулярных векторов
 * @param std::size_t block_size размер блока матрицы Bmat
 * @return std::size_t sweeps число разверток методом Якоби
 **/
std::size_t rrbnsvd_seq(struct matrix_t Amat, struct matrix_t Bmat, struct matrix_t Umat, struct matrix_t Vmat,
	std::size_t ThreadsNum,	bool vectorization, double* Time, struct string_t errors) {

	std::size_t subproblem_blocks = 4; //число блоков, которое формирует подзадачу
	std::size_t max_sweeps = 40; //максимальное число повторения итераций
	std::size_t sweeps = 0;  //число повторений цикла развертки
	const double tol = 1e-10;  //точность предела сходимости
	std::size_t n = Amat.rows; //размер матрицы
	double norm = 0.0;      //норма Фробениуса матрицы B
	double off_norm = 0.0;  //норма Фробениуса только недиагональных элементов матрицы B
	double t1, t2;              //замер времени

	matrix_copy(Bmat, Amat); //инициализация рабочей матрицы B
	//матрицы U и V инициализируются как единичные матрицы
	matrix_identity(Umat);
	matrix_identity(Vmat);
	matrix_frobenius(Bmat, &norm, &off_norm);

	//Входная матрица должна быть квадратной
	if (Amat.cols != Amat.rows) {
		*errors.len = sprintf(errors.ptr, "matrix must be square");
		return 0;
	}

	std::size_t rr_pairs = ThreadsNum; //число пар индексов (по одному блоку на один поток)
	std::size_t n_blocks = rr_pairs * 2; //число блоков вдоль строки/столбца
	std::size_t block_size = n / n_blocks; //число элементов на один блок

	//общее число элементов всех блоков строки/столбца должно быть равно размерности матрицы
	if (n_blocks * block_size != n) {
		*errors.len = sprintf(errors.ptr, "matrix must be correctly divided into blocks");
		return 0;
	}

	if (vectorization && (n % (4 * rr_pairs) != 0)) {
		*errors.len = sprintf(errors.ptr, "matrix must be correctly divided into blocks for vectorization");
		return 0;
	}

	//если строка/столбец состоят из двух блоков, лучше вычислить SVD классическим не блочным Якоби
	if (n <= 2 * block_size) {
		t1 = omp_get_wtime();
		std::size_t block_iters;
		if (vectorization)
			block_iters = svd_subprocedure_vectorized(Bmat, Umat, Vmat);
		else
			block_iters = svd_subprocedure(Bmat, Umat, Vmat);
		t2 = omp_get_wtime();
		*Time = t2 - t1;

		if (block_iters == 0) {
			*errors.len = sprintf(errors.ptr, "algorithm did not converge after %lu sweeps", block_iters);
			return 0;
		}
		return block_iters;
	}

	std::vector<std::size_t> up(rr_pairs); //массив хранит индекс i блока для одного потока
	std::vector<std::size_t> dn(rr_pairs); //массив хранит индекс j блока для одного потока

	//инициализация массива пар индексов вращений 
	//нечетными (up[]) и четными (dn[]) числами
	for (std::size_t i = 0; i < rr_pairs; i++) {
		up[i] = (2 * i);
		dn[i] = (2 * i) + 1;
	}

	//выделение памяти для хранения блоков матриц B U V M1 M2
	std::vector<double> Bblock(subproblem_blocks * block_size * block_size);
	std::vector<double> Ublock(subproblem_blocks * block_size * block_size);
	std::vector<double> Vblock(subproblem_blocks * block_size * block_size);
	//выделение памяти для хранения промежуточных значений умножения блоков
	std::vector<double> M1(block_size * block_size);
	std::vector<double> M2(block_size * block_size);

	//хранение блоков в виде структур matrix_t
	matrix_t Bblockmat = { &Bblock[0], 2 * block_size, 2 * block_size, 'R' };
	matrix_t Ublockmat = { &Ublock[0], 2 * block_size, 2 * block_size, 'R' };
	matrix_t Vblockmat = { &Vblock[0], 2 * block_size, 2 * block_size, 'R' };
	matrix_t M1mat = { &M1[0], block_size, block_size, 'R' };
	matrix_t M2mat = { &M2[0], block_size, block_size, 'R' };

	t1 = omp_get_wtime();
	bool converged = sqrt(off_norm) > tol * sqrt(norm);

	//основной цикл развретки, он продолжается пока 
	while (converged) {
		//цикл обхода над/поддиагональных блоков
		for (std::size_t iteration = 0; iteration < n_blocks - 1; ++iteration) {
			//Выполнение параллельных итераций цикла заменяется на выполнение в параллельном регионе. Один поток фактически параллельно
			//выполняет одну итерацию цикла
			for (size_t rr_pair = 0; rr_pair < rr_pairs; ++rr_pair)
			{
				std::size_t i_block = up[rr_pair];
				std::size_t j_block = dn[rr_pair];
				if (i_block > j_block)
					std::swap(i_block, j_block);

				//в Bblockmat копируются блоки с индексами ii ij ji jj из матрицы Bmat
				copy_subproblem_blocks(Bmat, i_block, j_block, Bblockmat, 0, 1, block_size);

				//вычисление SVD разложения циклическим методом Якоби над блоком Bblockmat
				//Bblockmat используется только в этой процедуре для упрощения обращения к индексам внутри
				std::size_t subproc_iter;
				if (vectorization)
					subproc_iter = svd_subprocedure_vectorized(Bblockmat, Ublockmat, Vblockmat);
				else
					subproc_iter = svd_subprocedure(Bblockmat, Ublockmat, Vblockmat);

				if (subproc_iter == 0) {
					*errors.len = sprintf(errors.ptr, "algorithm did not converge after %lu sweeps", sweeps);
					return 0;
				}

				//транспонировать блок матрицы U 
				matrix_transpose(Ublockmat, Ublockmat);

				//выполнение операции U^T*B*V, т.е. обновление строк и столбцов в двух циклах
				//элементами U, V выступают матрицы текущей подпроблемы, B - исходная матрица 
				//при этом матрица B считается диагональной
				//обновление блоков исходной матрицы B по строкам i j с помощью U
				for (std::size_t k_block = 0; k_block < n_blocks; ++k_block) {
					mult_block(Ublockmat, 0, 0, Bmat, i_block, k_block, M2mat, 0, 0, block_size);
					mult_add_block(Ublockmat, 0, 1, Bmat, j_block, k_block, M2mat, 0, 0, block_size);
					mult_block(Ublockmat, 1, 0, Bmat, i_block, k_block, M1mat, 0, 0, block_size);
					copy_block(M2mat, 0, 0, Bmat, i_block, k_block, block_size);
					mult_add_block(Ublockmat, 1, 1, Bmat, j_block, k_block, M1mat, 0, 0, block_size);
					copy_block(M1mat, 0, 0, Bmat, j_block, k_block, block_size);
				}

				//обновление блоков исходной матрицы B по столбцам i j с помощью V
				for (std::size_t k_block = 0; k_block < n_blocks; ++k_block) {
					mult_block(Bmat, k_block, i_block, Vblockmat, 0, 0, M2mat, 0, 0, block_size);
					mult_add_block(Bmat, k_block, j_block, Vblockmat, 1, 0, M2mat, 0, 0, block_size);
					mult_block(Bmat, k_block, i_block, Vblockmat, 0, 1, M1mat, 0, 0, block_size);
					copy_block(M2mat, 0, 0, Bmat, k_block, i_block, block_size);
					mult_add_block(Bmat, k_block, j_block, Vblockmat, 1, 1, M1mat, 0, 0, block_size);
					copy_block(M1mat, 0, 0, Bmat, k_block, j_block, block_size);
				}

				//возвращение транспонированного блока U^T в исходное состояние U 
				matrix_transpose(Ublockmat, Ublockmat);
				//обновление блоков исходной матрицы U по столбцам i j путем умножения на матрицу подпроблемы Ublockmat
				for (std::size_t k_block = 0; k_block < n_blocks; ++k_block) {
					mult_block(Umat, k_block, i_block, Ublockmat, 0, 0, M2mat, 0, 0, block_size);
					mult_add_block(Umat, k_block, j_block, Ublockmat, 1, 0, M2mat, 0, 0, block_size);
					mult_block(Umat, k_block, i_block, Ublockmat, 0, 1, M1mat, 0, 0, block_size);
					copy_block(M2mat, 0, 0, Umat, k_block, i_block, block_size);
					mult_add_block(Umat, k_block, j_block, Ublockmat, 1, 1, M1mat, 0, 0, block_size);
					copy_block(M1mat, 0, 0, Umat, k_block, j_block, block_size);
				}
				//обновление блоков исходной матрицы V по столбцам i j путем умножения на матрицу подпроблемы Vblockmat
				for (std::size_t k_block = 0; k_block < n_blocks; ++k_block) {
					mult_block(Vmat, k_block, i_block, Vblockmat, 0, 0, M2mat, 0, 0, block_size);
					mult_add_block(Vmat, k_block, j_block, Vblockmat, 1, 0, M2mat, 0, 0, block_size);
					mult_block(Vmat, k_block, i_block, Vblockmat, 0, 1, M1mat, 0, 0, block_size);
					copy_block(M2mat, 0, 0, Vmat, k_block, i_block, block_size);
					mult_add_block(Vmat, k_block, j_block, Vblockmat, 1, 1, M1mat, 0, 0, block_size);
					copy_block(M1mat, 0, 0, Vmat, k_block, j_block, block_size);
				}
			}
			round_robin(&up[0], &dn[0], rr_pairs);
		}

		matrix_frobenius(Bmat, &norm, &off_norm);
		converged = sqrt(off_norm) > tol * sqrt(norm);
		if (sweeps > max_sweeps)
			converged = false;
		else
			sweeps++;
	}

	t2 = omp_get_wtime();

	if (sweeps > max_sweeps) {
		*errors.len = sprintf(errors.ptr, "algorithm did not converge after %lu sweeps", sweeps);
		return 0;
	}

	*Time = t2 - t1;
	return sweeps;
}