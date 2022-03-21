#pragma once
#include "../../utils/types.hpp"

/**
* ������������� ����� ����� (Hestenes Jacobi), 
  �������� ���������� ��������������� �� ������� �������
 * @param matrix_t Amat ������������� ������� A
 * @param vector_t svec ������ ����������� �����
 * @param matrix_t Umat ������� ����� ����������� ��������
 * @param matrix_t Vmat ������� ������ ����������� ��������
 * @return size_t sweeps ����� ��������� ������� �����
 **/
size_t coloshjac(struct matrix_t A, struct vector_t s, struct matrix_t U, struct matrix_t V, size_t ThreadsNum, double* Time);

/**
* ������� ������������� ����� ����� (Block Jacobi Relaxasion),
* ����� ���������� � ������������ �� ���������� ���������� ������� (Round Robin) 
**/
size_t rrbjrs(struct matrix_t A, struct vector_t s, struct matrix_t U, struct matrix_t V, size_t ThreadsNum, double* Time);