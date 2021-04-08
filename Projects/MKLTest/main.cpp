#include <iostream>
#include <fstream>
#include <mkl.h>
#include "types.hpp"
#include "boost/format.hpp"

void matrix_from_file(matrix_t A, const char* path);
void matrix_to_file(matrix_t A, const char* path);
void vector_to_file(vector_t V, const char* path);

int main(int argc, char* argv[])
{
	int n;
	int m;

    if (argc < 3)
    {
        std::cout << "First argument must be a size N of matrix" << std::endl;
        std::cout << "Second argument must be a size M of matrix (M >= N)" << std::endl;
        return 1;
    }

    n = atoi(argv[1]);
    m = atoi(argv[2]);

    int matr_size = m * n;
    std::vector<double> A(matr_size);
    matrix_t Data_matr = { &A[0], m, n };

	std::cout << "Singular decomposition of double square matrix by MKL sequence dgesvj function" << std::endl;
    matrix_from_file(Data_matr, "./LocalData/in/square.in");

    char joba[] = "G"; //Указывает, что входящая матрица A(mxn) имееет общий вид (m >= n)
    char jobu[] = "U"; //Указывает, что ненулевые левые сингулярные векторы будут вычислены, и сохранены в матрице A
    char jobv[] = "V"; //Указывает, что правые сингулярные векторы будут вычислены, и сохранены в матрице V
    int lwork = m + n; //Размер массива сингулярных чисел
    int info;          //Состояние вычисление. Если 0, то успешно. Если > 0, то функция не сходится уже больше 30 разверток
    int mv = 0;
    int lda = m;       //Ведущая "ось" матрицы A (строки)
    int ldv = n;       //Ведущая "ось" матрицы V (столбцы) 
    int v_size = ldv * ldv; //Размер вектора V
    std::vector<double> sigma(n); 
    std::vector<double> v(v_size);
    matrix_t V_matr = { &v[0], ldv, ldv };
    vector_t S_vect = { &sigma[0], n};
    std::vector<double> workspace(lwork);

    //Описание функции dgesvj OneMKL
    //https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-fortran/top/lapack-routines/lapack-least-squares-and-eigenvalue-problem-routines/lapack-least-squares-and-eigenvalue-problem-driver-routines/singular-value-decomposition-lapack-driver-routines/gesvj.html
    dgesvj(joba, jobu, jobv, &m, &n, &A[0], &lda, &sigma[0], &mv, &v[0], &ldv, &workspace[0], &lwork, &info);
    
    matrix_to_file(Data_matr, "./LocalData/out/SVD/dgesvj/square/U.to");
    matrix_to_file(V_matr, "./LocalData/out/SVD/dgesvj/square/V.to");
    vector_to_file(S_vect, "./LocalData/out/SVD/dgesvj/square/A.to");

    return 0;
}

void matrix_from_file(matrix_t A, const char* path) {
    std::ifstream file;
    file.open(path);

    for (size_t i = 0; i < (A.rows * A.cols); i++)
        file >> A.ptr[i];

    file.close();
}

void matrix_to_file(matrix_t A, const char* path) {
    std::ofstream output(path);
    int n = A.rows;

    for (size_t i = 0; i < A.rows; ++i) {
        for (size_t j = 0; j < A.cols; ++j) {
            output << boost::format{ "%f" } % A.ptr[n * i + j] << " \t";
        }
        output << "\n";
    }
    output.close();
}

void vector_to_file(vector_t V, const char* path) {
    std::ofstream output(path);
    int n = V.len;

    for (size_t i = 0; i < n; ++i) {
        output << boost::format{ "%f" } % V.ptr[i] << " \t";
    }
    output.close();
}