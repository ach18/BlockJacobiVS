#pragma once
#include "../../utils/types.hpp"

/**
* ������������� ����� ����� (Hestenes Jacobi), 
  �������� ���������� ��������������� �� ������� �������
 * @param matrix_t Amat ������������� ������� A
 * @param vector_t svec ������ ����������� �����
 * @param matrix_t Umat ������� ����� ����������� ��������
 * @param matrix_t Vmat ������� ������ ����������� ��������
 * @return std::size_t sweeps ����� ��������� ������� �����
 **/
std::size_t coloshjac(struct matrix_t A, struct vector_t s, struct matrix_t U, struct matrix_t V, std::size_t ThreadsNum, double* Time, struct string_t errors);

/**
* ������� ������������� ����� ����� (Block Jacobi Relaxasion),
* ����� ���������� � ������������ �� ���������� ���������� ������� (Round Robin) 
**/
std::size_t rrbjrs(struct matrix_t A, struct matrix_t Bmat, struct vector_t s, struct matrix_t U, struct matrix_t V, std::size_t ThreadsNum, double* Time, struct string_t errors);

/**
* ������� ������������� ����� ����� (Block Jacobi Relaxasion) + ������������ ����,
* ����� ���������� � ������������ �� ���������� ���������� ������� (Round Robin)
**/
std::size_t rrbjrs_vectorization(struct matrix_t A, struct matrix_t Bmat_aligned, struct vector_t s_aligned, struct matrix_t U_aligned, struct matrix_t V_aligned, std::size_t ThreadsNum, double* Time, struct string_t errors);