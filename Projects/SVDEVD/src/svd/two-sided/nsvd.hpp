#pragma once

/*
 * ������������ cos sin �������� ��� ij ��������.
 */
struct svd_2x2_params {
    double d1, d2;
    double c1, s1;
    double c2, s2;
    double k;
};

/*
 *
 * @param w ������� � ������� (i,i).
 * @param x ������� � ������� (i,j).
 * @param y ������� � ������� (j,i).
 * @param z ������� � ������� (j,j).
 * @return ������������ ������� 2x2.
 */
struct svd_2x2_params nsvd(double w, double x, double y, double z);
