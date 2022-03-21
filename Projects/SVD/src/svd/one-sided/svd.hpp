#pragma once
#include "../../utils/types.hpp"

/**
* ќдносторонний метод якоби (Hestenes Jacobi), 
  элементы выбираютс€ последовательно по столбцу матрицы
 * @param matrix_t Amat пр€моугольна€ матрица A
 * @param vector_t svec вектор сингул€рных чисел
 * @param matrix_t Umat матрица левых сингул€рных векторов
 * @param matrix_t Vmat матрица правых сингул€рных векторов
 * @return size_t sweeps число разверток методом якоби
 **/
size_t coloshjac(struct matrix_t A, struct vector_t s, struct matrix_t U, struct matrix_t V, size_t ThreadsNum, double* Time);

/**
* Ѕлочный односторонний метод якоби (Block Jacobi Relaxasion),
* Ѕлоки выбираютс€ в соответствии со стратегией шахматного турнира (Round Robin) 
**/
size_t rrbjrs(struct matrix_t A, struct vector_t s, struct matrix_t U, struct matrix_t V, size_t ThreadsNum, double* Time);