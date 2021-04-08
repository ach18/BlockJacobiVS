#pragma once
#include "../../utils/types.hpp"

/**
 * @param matrix_t Amat симметричная квадратная (или прямоугольная) матрица A
 * @param vector_t svec вектор сингулярных чисел
 * @param matrix_t Umat матрица левых сингулярных векторов
 * @param matrix_t Vmat матрица правых сингулярных векторов
 * @param size_t n_iter число необходимых для сходимости разверток
 * @return size_t sweeps число разверток методом Якоби
 **/
size_t svd(struct matrix_t A, struct vector_t s, struct matrix_t U, struct matrix_t V, size_t n_iter = 100);