#include <iostream>
#include <fstream>
#include <map>
#include <iterator>
#include <mkl.h>
#include <omp.h>
#include <vector>
//#include "types.hpp"

struct vector_t {
    double* ptr;
    size_t len;
};

struct matrix_t {
    double* ptr;
    size_t rows;
    size_t cols;
};

struct index_t {
    size_t i;
    size_t j;
};

void random_matrix(matrix_t A);
void matrix_from_file(matrix_t A, const char* path);
void matrix_to_file(matrix_t A, const char* path);
void vector_to_file(vector_t V, const char* path);
void map_to_file(std::map<int, double> Map, const char* path);
void dgesvj_loop();

int main(int argc, char* argv[])
{
    long long n;
    double t_mkl1, t_mkl2, t_jac1, tjac2, t_result;
    int square_matrix_size;
    std::map<int, double> mkl_seq_times;

    std::cout << "Singular decomposition of double square matrix by MKL parallel dgesvj function" << std::endl;

    n = 100;
    square_matrix_size = n * n;
    std::vector<double> A_square(square_matrix_size);
    matrix_t Data_square_matr = { &A_square[0], n, n };
    matrix_from_file(Data_square_matr, "./LocalData/in/square/100_100.in");

    //MKL параметры
    char joba[] = "G"; //Указывает, что входящая матрица A(mxn) имееет общий вид (m >= n)
    char jobu[] = "U"; //Указывает, что ненулевые левые сингулярные векторы будут вычислены, и сохранены в матрице A
    char jobv[] = "V"; //Указывает, что правые сингулярные векторы будут вычислены, и сохранены в матрице V
    long long lwork = n + n; //Размер массива сингулярных чисел
    long long info;          //Состояние вычисление. Если 0, то успешно. Если > 0, то функция не сходится уже больше 30 разверток
    long long mv = 0;
    long long lda = n;       //Ведущая "ось" матрицы A (строки)
    long long ldv = n;       //Ведущая "ось" матрицы V (столбцы) 
    long long v_size = ldv * ldv; //Размер вектора V
    std::vector<double> sigma(n);
    std::vector<double> v(v_size);
    matrix_t V_matr = { &v[0], ldv, ldv };
    vector_t S_vect = { &sigma[0], n };
    std::vector<double> workspace(lwork);

    t_mkl1 = omp_get_wtime();
    //Описание функции dgesvj OneMKL
    //https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-fortran/top/lapack-routines/lapack-least-squares-and-eigenvalue-problem-routines/lapack-least-squares-and-eigenvalue-problem-driver-routines/singular-value-decomposition-lapack-driver-routines/gesvj.html
    dgesvj(joba, jobu, jobv, &n, &n, &Data_square_matr.ptr[0], &lda, &S_vect.ptr[0], &mv, &V_matr.ptr[0], &ldv, &workspace[0], &lwork, &info);
    t_mkl2 = omp_get_wtime();
    t_result = t_mkl2 - t_mkl1;
    mkl_seq_times[n] = t_result;        

    vector_to_file(S_vect, "./LocalData/out/SVD/dgesvj/square/A.to");
    matrix_to_file(Data_square_matr, "./LocalData/out/SVD/dgesvj/square/U.to");
    matrix_to_file(V_matr, "./LocalData/out/SVD/dgesvj/square/V.to");
    return 0;
}

void dgesvj_loop() {
    long long n;
    double t_mkl1, t_mkl2, t_jac1, tjac2, t_result;
    int square_matrix_size;
    std::map<int, double> mkl_seq_times;

    std::cout << "Singular decomposition of double square matrix by MKL sequence dgesvj function" << std::endl;
    int sweeps = 30;
    long long sizes[] = { 100, 200, 300, 500, 1000, 1100, 1200, 1300, 1500, 2000, 2100, 2200, 2300, 2500, 3000,
        3100, 3200, 3300, 3500, 4000, 4100, 4200, 4300, 4500, 5000, 5100, 5200, 5300, 5500, 6000 };

    for (int i = 0; i < sweeps; i++) {
        n = sizes[i];
        square_matrix_size = n * n;
        std::vector<double> A_square(square_matrix_size);
        matrix_t Data_square_matr = { &A_square[0], n, n };
        random_matrix(Data_square_matr);

        //MKL параметры
        char joba[] = "G"; //Указывает, что входящая матрица A(mxn) имееет общий вид (m >= n)
        char jobu[] = "U"; //Указывает, что ненулевые левые сингулярные векторы будут вычислены, и сохранены в матрице A
        char jobv[] = "V"; //Указывает, что правые сингулярные векторы будут вычислены, и сохранены в матрице V
        long long lwork = n + n; //Размер массива сингулярных чисел
        long long info;          //Состояние вычисление. Если 0, то успешно. Если > 0, то функция не сходится уже больше 30 разверток
        long long mv = 0;
        long long lda = n;       //Ведущая "ось" матрицы A (строки)
        long long ldv = n;       //Ведущая "ось" матрицы V (столбцы) 
        long long v_size = ldv * ldv; //Размер вектора V
        std::vector<double> sigma(n);
        std::vector<double> v(v_size);
        matrix_t V_matr = { &v[0], ldv, ldv };
        vector_t S_vect = { &sigma[0], n };
        std::vector<double> workspace(lwork);

        t_mkl1 = omp_get_wtime();
        //Описание функции dgesvj OneMKL
        //https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-fortran/top/lapack-routines/lapack-least-squares-and-eigenvalue-problem-routines/lapack-least-squares-and-eigenvalue-problem-driver-routines/singular-value-decomposition-lapack-driver-routines/gesvj.html
        dgesvj(joba, jobu, jobv, &n, &n, &A_square[0], &lda, &sigma[0], &mv, &v[0], &ldv, &workspace[0], &lwork, &info);
        t_mkl2 = omp_get_wtime();
        t_result = t_mkl2 - t_mkl1;
        mkl_seq_times[n] = t_result;
    }

    map_to_file(mkl_seq_times, "./size_per_time.to");
}

void random_matrix(matrix_t A) {
    for (size_t i = 0; i < (A.rows * A.cols); i++)
        A.ptr[i] = rand() % 100;
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
            output << std::fixed << A.ptr[n * i + j] << " \t";
        }
        output << "\n";
    }
    output.close();
}

void vector_to_file(vector_t V, const char* path) {
    std::ofstream output(path);
    int n = V.len;

    for (size_t i = 0; i < n; ++i) {
        output << std::fixed << V.ptr[i] << " \t";
    }
    output.close();
}

void map_to_file(std::map<int, double> Map, const char* path) {
    std::ofstream output(path);
    std::map<int, double>::iterator iter = Map.begin();

    //for (size_t i = 0; i < n; ++i) {
    while (iter != Map.end()) {
        output << std::fixed << iter->first << "\t" << std::fixed << iter->second << "\n";
        iter++;
    }
    output.close();
}