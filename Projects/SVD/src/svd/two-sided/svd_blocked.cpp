#include <cassert>
#include <omp.h>
#include <math.h>
#include <cstdlib>
#include <algorithm>
#include "../../utils/block.hpp"
#include "../../utils/matrix.hpp"
#include "nsvd.hpp"
#include "svd.hpp"
#include "svd_subprocedure.hpp"
#include "../../utils/types.hpp"
#include "../../utils/util.hpp"

/**
 * @param matrix_t Amat квадратная матрица A
 * @param matrix_t Bmat матрица A после деления на блоки
 * @param matrix_t Umat матрица левых сингулярных векторов
 * @param matrix_t Vmat матрица правых сингулярных векторов
 * @param size_t block_size размер блока матрицы Bmat
 * @return size_t sweeps число разверток методом Якоби
 **/
size_t colbnsvd(struct matrix_t Amat, struct matrix_t Bmat, struct matrix_t Umat, struct matrix_t Vmat,
    size_t block_size, size_t ThreadsNum, double* Time) {

    size_t sweeps = 0;  //число повторений цикла развертки
    size_t block_iter = 0;
    const double tol = 1e-10;  //точность предела сходимости
    const size_t n = Amat.rows; //размер матрицы
    double norm = 0.0;      //норма Фробениуса матрицы B
    double off_norm = 0.0;  //норма Фробениуса только недиагональных элементов матрицы B
	double t1, t2;              //замер времени

    matrix_copy(Bmat, Amat); //инициализация рабочей матрицы B
    //матрицы U и V инициализируются как единичные матрицы
    matrix_identity(Umat);
    matrix_identity(Vmat);
    matrix_frobenius(Bmat, &norm, &off_norm);

    const size_t n_blocks = n / block_size; //число блоков вдоль строки/столбца

    //если строка/столбец состоят из двух блоков, лучше вычислить SVD классическим не блочным Якоби
    if (n <= 2 * block_size) {
        size_t block_iters = svd_subprocedure(Bmat, Umat, Vmat);
        return block_iters;
    }

    //общее число элементов всех блоков строки/столбца должно быть равно размерности матрицы
    assert(n_blocks * block_size == n);

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

	omp_set_num_threads(ThreadsNum);
	t1 = omp_get_wtime();
	bool converged = sqrt(off_norm) > tol * sqrt(norm);

    //основной цикл развретки, он продолжается пока 
    while (converged) {
        //цикл обхода над/поддиагональных блоков
        for (size_t i_block = 0; i_block < n_blocks - 1; ++i_block) {
            for (size_t j_block = i_block + 1; j_block < n_blocks; ++j_block) {
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
                for (size_t k_block = 0; k_block < n_blocks; ++k_block) {
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
                for (size_t k_block = 0; k_block < n_blocks; ++k_block) {
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
                for (size_t k_block = 0; k_block < n_blocks; ++k_block) {
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
                for (size_t k_block = 0; k_block < n_blocks; ++k_block) {
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
 * @param size_t block_size размер блока матрицы Bmat
 * @return size_t sweeps число разверток методом Якоби
 **/
size_t rrbnsvd(struct matrix_t Amat, struct matrix_t Bmat, struct matrix_t Umat, struct matrix_t Vmat,
	size_t block_size, size_t ThreadsNum, double* Time) {

	size_t sweeps = 0;  //число повторений цикла развертки
	size_t block_iter = 0;
	const double tol = 1e-10;  //точность предела сходимости
	const size_t n = Amat.rows; //размер матрицы
	double norm = 0.0;      //норма Фробениуса матрицы B
	double off_norm = 0.0;  //норма Фробениуса только недиагональных элементов матрицы B
	double t1, t2;              //замер времени

	matrix_copy(Bmat, Amat); //инициализация рабочей матрицы B
	//матрицы U и V инициализируются как единичные матрицы
	matrix_identity(Umat);
	matrix_identity(Vmat);
	matrix_frobenius(Bmat, &norm, &off_norm);

	const size_t n_blocks = n / block_size; //число блоков вдоль строки/столбца

	//если строка/столбец состоят из двух блоков, лучше вычислить SVD классическим не блочным Якоби
	if (n <= 2 * block_size) {
		size_t block_iters = svd_subprocedure(Bmat, Umat, Vmat);
		return block_iters;
	}

	//Входная матрица должна быть квадратной
	assert(Amat.cols == Amat.rows);
	//общее число элементов всех блоков строки/столбца должно быть равно размерности матрицы
	assert(n_blocks * block_size == n);
	//число блоков должно быть четным
	assert((n_blocks % 2) == 0);
	size_t rr_pairs = n_blocks / 2; //число пар индексов
	std::vector<size_t> up(rr_pairs); //массив хранит индекс i блока для одного потока
	std::vector<size_t> dn(rr_pairs); //массив хранит индекс j блока для одного потока

	//инициализация массива пар индексов вращений 
	//нечетными (up[]) и четными (dn[]) числами
	for (size_t i = 0; i < rr_pairs; i++) {
		up[i] = (2 * i);
		dn[i] = (2 * i) + 1;
	}

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

	omp_set_num_threads(ThreadsNum);
	t1 = omp_get_wtime();
	bool converged = sqrt(off_norm) > tol * sqrt(norm);

	//основной цикл развретки, он продолжается пока 
	while (converged) {
		//цикл обхода над/поддиагональных блоков
		for (size_t iteration = 0; iteration < n_blocks - 1; ++iteration) {
#pragma omp parallel shared(Bmat, Umat, Vmat, block_size, up, dn, n_blocks) \
	firstprivate(Bblockmat, Ublockmat, Vblockmat, block_iter, M1mat, M2mat)
{
	#pragma omp for schedule(guided)
			for (size_t rr_pair = 0; rr_pair < rr_pairs; ++rr_pair) {
				size_t i_block = up[rr_pair];
				size_t j_block = dn[rr_pair];

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
				for (size_t k_block = 0; k_block < n_blocks; ++k_block) {
					mult_block(Ublockmat, 0, 0, Bmat, i_block, k_block, M1mat, 0, 0, block_size);
					mult_block(Ublockmat, 0, 1, Bmat, j_block, k_block, M2mat, 0, 0, block_size);
					matrix_add(M1mat, M2mat, M2mat);
					mult_block(Ublockmat, 1, 0, Bmat, i_block, k_block, M1mat, 0, 0, block_size);
					copy_block(M2mat, 0, 0, Bmat, i_block, k_block, block_size);
					mult_block(Ublockmat, 1, 1, Bmat, j_block, k_block, M2mat, 0, 0, block_size);
					matrix_add(M1mat, M2mat, M2mat);
					copy_block(M2mat, 0, 0, Bmat, j_block, k_block, block_size);
				}
			}
	#pragma omp for schedule(guided)
			for (size_t rr_pair = 0; rr_pair < rr_pairs; ++rr_pair) {
				size_t i_block = up[rr_pair];
				size_t j_block = dn[rr_pair];
				//обновление блоков исходной матрицы B по столбцам i j с помощью V
				for (size_t k_block = 0; k_block < n_blocks; ++k_block) {
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
				for (size_t k_block = 0; k_block < n_blocks; ++k_block) {
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
				for (size_t k_block = 0; k_block < n_blocks; ++k_block) {
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
			round_robin(&up[0], &dn[0], rr_pairs);
		}

		matrix_frobenius(Bmat, &norm, &off_norm);
		converged = sqrt(off_norm) > tol * sqrt(norm);
		if (sweeps > 30)
			converged = false;

		sweeps++;
	}

	t2 = omp_get_wtime();

	if (sweeps > 30)
		return 0;

	*Time = t2 - t1;
	return sweeps;
}