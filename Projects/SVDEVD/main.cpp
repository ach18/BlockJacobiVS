#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include "src/utils/types.hpp"
#include "src/utils/util.hpp"
#include "src/svd/two-sided/svd.hpp"
#include "src/svd/one-sided/svd.hpp"
#include "alglib/linalg.h"

int main(int argc, char* argv[])
{
    size_t n; //Размер квадратной симметричной матрицы A(nxn)
    size_t block_size; //Размер блока матрицы (минимум 8)

    if (argc < 3)
    {
        std::cout << "First argument must be a size of square matrix" << std::endl;
        std::cout << "Second argument must be a block size (minimum 8)" << std::endl;
        return 1;
    }

    n = atoi(argv[1]);
    block_size = atoi(argv[2]);
    std::cout << "Singular decomposition of double square matrix by two-sided NSVD Jacobi" << std::endl;

    //Объявление структур и типов данных
    aligned_vector<double> A(n * n); 
    aligned_vector<double> B(n * n);
    aligned_vector<double> U(n * n, 0);
    aligned_vector<double> V(n * n, 0);
    //Data_matr - исходная матрица A
    //B_mat - сингулярные числа матрицы
    //U_mat - левые сингулярные векторы
    //V_mat - правые сингулярные векторы
    matrix_t Data_matr = { &A[0], n, n };
    matrix_t B_mat = { &B[0], n, n };
    matrix_t U_mat = { &U[0], n, n };
    matrix_t V_mat = { &V[0], n, n };

    //Инициализация матрицы A
    matrix_from_file(Data_matr, "./LocalData/in/square.in");
    assert(Data_matr.rows == Data_matr.cols);
    assert(Data_matr.rows == B_mat.rows && Data_matr.cols == B_mat.cols);
    assert(Data_matr.rows == U_mat.rows && Data_matr.cols == U_mat.cols);
    assert(Data_matr.rows == V_mat.rows && Data_matr.cols == V_mat.cols);

    //Блочное двустороннее сингулярное (SVD) разложение методом NSVD квадратной вещественной матрицы A
    //NSVD алгоритм описан в работе https://maths-people.anu.edu.au/~brent/pd/rpb080i.pdf (стр. 12)
    //Сейчас используется циклический перебор наддиагональных блоков матрицы
    size_t iterations = svd_blocked(Data_matr, B_mat, U_mat, V_mat, block_size);
    
    //Запись в файлы полученных данных
    //А - сингулярные числа матрицы
    //U - левые сингулярные векторы
    //V - правые сингулярные векторы
    matrix_to_file(B_mat, "./LocalData/out/SVD/2DNSVD/square/A.to");
    matrix_to_file(U_mat, "./LocalData/out/SVD/2DNSVD/square/V.to");
    matrix_to_file(V_mat, "./LocalData/out/SVD/2DNSVD/square/U.to");

    std::cout << "Singular decomposition of double square matrix by Alglib sequence rmatrixsvd function" << std::endl;
    // Инициализация структур данных библиотеки Alglib
    alglib::real_2d_array A_alglib, U_alglib, V_alglib;
    alglib::real_1d_array W_alglib;
    A_alglib.attach_to_ptr(n,n, Data_matr.ptr);
    W_alglib.setlength(n);
    U_alglib.setlength(n, n);
    V_alglib.setlength(n, n);
    //Сингулярное (SVD) разложение функцией rmatrixsvd квадратной (или прямоугольной) вещественной матрицы A
    //Функция rmatrixsvd часть библиотеки Alglib
    //https://www.alglib.net/matrixops/general/svd.php
    bool rmatrixsvd_result = alglib::rmatrixsvd(A_alglib, n, n, 2, 2, 0, W_alglib, U_alglib, V_alglib);

    //Запись в файлы полученных данных
    U_mat.ptr = &U_alglib[0][0];
    V_mat.ptr = &V_alglib[0][0];
    vector_t S_vect = { &W_alglib[0], n };
    matrix_to_file(U_mat, "./LocalData/out/SVD/Alglib/V.to");
    matrix_to_file(V_mat, "./LocalData/out/SVD/Alglib/U.to");
    vector_to_file(S_vect, "./LocalData/out/SVD/Alglib/A.to");


    std::cout << "Singular decomposition of double square matrix by one-sided Hestenes Jacobi" << std::endl;
    //Одностороннее сингулярное (SVD) разложение по столбцам методом Hestenes one-sided Jacobi квадратной (или прямоугольной) 
    // вещественной матрицы A
    //Алгоритм описан в работе https://www.cs.utexas.edu/users/inderjit/public_papers/HLA_SVD.pdf (Algorithm 6, стр. 11)
    //Сейчас используется циклический перебор наддиагональных блоков матрицы
    iterations = svd(Data_matr, S_vect, U_mat, V_mat, 3);

    matrix_to_file(U_mat, "./LocalData/out/SVD/1DSVD/square/V.to");
    matrix_to_file(V_mat, "./LocalData/out/SVD/1DSVD/square/U.to");
    vector_to_file(S_vect, "./LocalData/out/SVD/1DSVD/square/A.to");
    return 0;
}