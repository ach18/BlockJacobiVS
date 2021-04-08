#pragma once
#include "../../utils/types.hpp"

/**
 * @param matrix_t Amat ������������ ���������� (��� �������������) ������� A
 * @param vector_t svec ������ ����������� �����
 * @param matrix_t Umat ������� ����� ����������� ��������
 * @param matrix_t Vmat ������� ������ ����������� ��������
 * @param size_t n_iter ����� ����������� ��� ���������� ���������
 * @return size_t sweeps ����� ��������� ������� �����
 **/
size_t svd(struct matrix_t A, struct vector_t s, struct matrix_t U, struct matrix_t V, size_t n_iter = 100);