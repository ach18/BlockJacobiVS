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

struct compute_params {
    size_t m;
    size_t n;
    size_t threads;
    size_t iterations;
    double time;
};

void random_matrix(matrix_t A);
void matrix_from_file(matrix_t A, const char* path);
void matrix_to_file(matrix_t A, const char* path);
void vector_to_file(vector_t V, const char* path);
void map_to_file(std::map<int, double> Map, const char* path);
void compute_params_to_file(std::vector<compute_params> params, const char* path);
void dgesvj_loop();
void single_dgesvj(size_t n);


int main(int argc, char* argv[])
{
    std::vector<index_t> sizes = { {100, 100}, {200, 100}, {200, 200}, {300, 100}, {300, 200}, {300, 300}, {500, 100}, {500, 200}, {500, 300}, {500, 500}, 
        {1000, 100}, {1000, 200}, {1000, 300}, {1000, 500}, {1000, 1000}, {1100, 100}, {1100, 200}, {1100, 300}, {1100, 500}, {1100, 1000}, {1100, 1100}, 
        {1200, 100}, {1200, 200}, {1200, 300}, {1200, 500}, {1200, 1000}, {1200, 1100}, {1200, 1200}, {1300, 100}, {1300, 200}, {1300, 300}, {1300, 500},
        {1300, 1000}, {1300, 1100}, {1300, 1200}, {1300, 1300}, {1500, 100}, {1500, 200}, {1500, 300}, {1500, 500}, {1500, 1000}, {1500, 1100}, {1500, 1200},
        {1500, 1300}, {1500, 1500}, {2000, 100}, {2000, 200}, {2000, 300}, {2000, 500}, {2000, 1000}, {2000, 1100}, {2000, 1200}, {2000, 1300}, {2000, 1500}, 
        {2000, 2000}, {2100, 100}, {2100, 200}, {2100, 300}, {2100, 500}, {2100, 1000}, {2100, 1100}, {2100, 1200}, {2100, 1300}, {2100, 1500}, {2100, 2000},
        {2100, 2100}, {2200, 100}, {2200, 200}, {2200, 300}, {2200, 500}, {2200, 1000}, {2200, 1100}, {2200, 1200}, {2200, 1300}, {2200, 1500}, {2200, 2000},
        {2200, 2100}, {2200, 2200}, {2300, 100}, {2300, 200}, {2300, 300}, {2300, 500}, {2300, 1000}, {2300, 1100}, {2300, 1200}, {2300, 1300}, {2300, 1500},
        {2300, 2000}, {2300, 2100}, {2300, 2200}, {2300, 2300}, {2500, 100}, {2500, 200}, {2500, 300}, {2500, 500}, {2500, 1000}, {2500, 1100}, {2500, 1200},
        {2500, 1300}, {2500, 1500}, {2500, 2000}, {2500, 2100}, {2500, 2200}, {2500, 2300}, {2500, 2500}, {3000, 100}, {3000, 200}, {3000, 300}, {3000, 500},
        {3000, 1000}, {3000, 1100}, {3000, 1200}, {3000, 1300}, {3000, 1500}, {3000, 2000}, {3000, 2100}, {3000, 2200}, {3000, 2300}, {3000, 2500}, 
        {3000, 3000}, {3100, 100}, {3100, 200}, {3100, 300}, {3100, 500}, {3100, 1000}, {3100, 1100}, {3100, 1200}, {3100, 1300}, {3100, 1500}, {3100, 2000},
        {3100, 2100}, {3100, 2200}, {3100, 2300}, {3100, 2500}, {3100, 3000}, {3100, 3100}, {3200, 100}, {3200, 200}, {3200, 300}, {3200, 500}, {3200, 1000},
        {3200, 1100}, {3200, 1200}, {3200, 1300}, {3200, 1500}, {3200, 2000}, {3200, 2100}, {3200, 2200}, {3200, 2300}, {3200, 2500}, {3200, 3000}, 
        {3200, 3100}, {3200, 3200}, {3300, 100}, {3300, 200}, {3300, 300}, {3300, 500}, {3300, 1000}, {3300, 1100}, {3300, 1200}, {3300, 1300}, {3300, 1500},
        {3300, 2000}, {3300, 2100}, {3300, 2200}, {3300, 2300}, {3300, 2500}, {3300, 3000}, {3300, 3100}, {3300, 3200}, {3300, 3300}, {3500, 100}, 
        {3500, 200}, {3500, 300}, {3500, 500}, {3500, 1000}, {3500, 1200}, {3500, 1300}, {3500, 1500}, {3500, 2000}, {3500, 2100}, {3500, 2200}, {3500, 2300},
        {3500, 2500}, {3500, 3000}, {3500, 3100}, {3500, 3200}, {3500, 3300}, {3500, 3500}, {4000, 100}, {4000, 200}, {4000, 300}, {4000, 500}, {4000, 1000},
        {4000, 1100}, {4000, 1200}, {4000, 1300}, {4000, 1500}, {4000, 2000}, {4000, 2100}, {4000, 2200}, {4000, 2300}, {4000, 2500}, {4000, 3000}, 
        {4000, 3100}, {4000, 3200}, {4000, 3300}, {4000, 3500}, {4000, 4000}, {4100, 100}, {4100, 200}, {4100, 300}, {4100, 500}, {4100, 500}, {4100, 1000}, 
        {4100, 1100}, {4100, 1200}, {4100, 1300}, {4100, 1500}, {4100, 2000}, {4100, 2100}, {4100, 2200}, {4100, 2300}, {4100, 2500}, {4100, 3000}, 
        {4100, 3100}, {4100, 3200}, {4100, 3300}, {4100, 3500}, {4100, 4000}, {4100, 4100}, {4200, 100}, {4200, 200}, {4200, 500}, {4200, 1000}, 
        {4200, 1100}, {4200, 1200}, {4200, 1300}, {4200, 1500}, {4200, 2000}, {4200, 2100}, {4200, 2200}, {4200, 2300}, {4200, 2500}, {4200, 3000}, 
        {4200, 3100}, {4200, 3200}, {4200, 3300}, {4200, 3500}, {4200, 4000}, {4200, 4100}, {4200, 4200}, {4300, 100}, {4300, 200}, {4300, 300}, {4300, 500},
        {4300, 1000}, {4300, 1100}, {4300, 1200}, {4300, 1300}, {4300, 1500}, {4300, 2000}, {4300, 2100}, {4300, 2200}, {4300, 2300}, {4300, 2500}, 
        {4300, 3000}, {4300, 3100}, {4300, 3200}, {4300, 3300}, {4300, 3500}, {4300, 4000}, {4300, 4100}, {4300, 4200}, {4300, 4300}, {4500, 100}, 
        {4500, 200}, {4500, 300}, {4500, 500}, {4500, 1000}, {4500, 1100}, {4500, 1200}, {4500, 1300}, {4500, 1500}, {4500, 2000}, {4500, 2100}, 
        {4500, 2200}, {4500, 2300}, {4500, 2500}, {4500, 3000}, {4500, 3100}, {4500, 3200}, {4500, 3300}, {4500, 3500}, {4500, 4000}, {4500, 4100}, 
        {4500, 4200}, {4500, 4300}, {4500, 4500}, {5000, 100}, {5000, 200}, {5000, 300}, {5000, 500}, {5000, 1000}, {5000, 1200}, {5000, 1300}, {5000, 1500},
        {5000, 2000}, {5000, 2100}, {5000, 2200}, {5000, 2300}, {5000, 2500}, {5000, 3000}, {5000, 3100}, {5000, 3200}, {5000, 3300}, {5000, 3500}, 
        {5000, 4000}, {5000, 4100}, {5000, 4200}, {5000, 4300}, {5000, 4500}, {5000, 5000}, {5100, 100}, {5100, 200}, {5100, 300}, {5100, 500}, {5100, 1000},
        {5100, 1100}, {5100, 1200}, {5100, 1300}, {5100, 1500}, {5100, 2000}, {5100, 2100}, {5100, 2200}, {5100, 2300}, {5100, 2500}, {5100, 3000}, 
        {5100, 3100}, {5100, 3200}, {5100, 3300}, {5100, 3500}, {5100, 4000}, {5100, 4100}, {5100, 4200}, {5100, 4300}, {5100, 4500}, {5100, 5000}, 
        {5100, 5100}, {5200, 100}, {5200, 200}, {5200, 300}, {5200, 500}, {5200, 1000}, {5200, 1100}, {5200, 1200}, {5200, 1300}, {5200, 1500}, 
        {5200, 2000}, {5200, 2100}, {5200, 2200}, {5200, 2300}, {5200, 2500}, {5200, 3000}, {5200, 3100}, {5200, 3200}, {5200, 3300}, {5200, 3500}, 
        {5200, 4000}, {5200, 4100}, {5200, 4200}, {5200, 4300}, {5200, 4500}, {5200, 5000}, {5200, 5100}, {5200, 5200}, {5300, 100}, {5300, 200}, {5300, 300},
        {5300, 500}, {5300, 1000}, {5300, 1100}, {5300, 1200}, {5300, 1300}, {5300, 1500}, {5300, 2000}, {5300, 2100}, {5300, 2200}, {5300, 2300}, 
        {5300, 2500}, {5300, 3000}, {5300, 3100}, {5300, 3200}, {5300, 3300}, {5300, 3500}, {5300, 4000}, {5300, 100}, {5300, 200}, {5300, 300}, {5300, 500},
        {5300, 1000}, {5300, 1100}, {5300, 1200}, {5300, 1300}, {5300, 1500}, {5300, 2000}, {5300, 2100}, {5300, 2200}, {5300, 2300}, {5300, 2500}, 
        {5300, 3000}, {5300, 3100}, {5300, 3200}, {5300, 3300}, {5300, 3500}, {5300, 4000}, {5300, 4100}, {5300, 4200}, {5300, 4300}, {5300, 4500},
        {5300, 5000}, {5300, 5100}, {5300, 5200}, {5300, 5300}, {5500, 100}, {5500, 200}, {5500, 300}, {5500, 500}, {5500, 1000}, {5500, 1100}, {5500, 1200},
        {5500, 1300}, {5500, 1500}, {5500, 2000}, {5500, 2100}, {5500, 2200}, {5500, 2300}, {5500, 2500}, {5500, 3000}, {5500, 3100}, {5500, 3200},
        {5500, 3300}, {5500, 3500}, {5500, 4000}, {5500, 4100}, {5500, 4200}, {5500, 4300}, {5500, 4500}, {5500, 5000}, {5500, 5100}, {5500, 5200},
        {5500, 5300}, {5500, 5500}, {6000, 100}, {6000, 200}, {6000, 300}, {6000, 500}, {6000, 1000}, {6000, 1200}, {6000, 1300}, 
        {6000, 1500}, {6000, 2000}, {6000, 2100}, {6000, 2200}, {6000, 2300}, {6000, 2500}, {6000, 3000}, {6000, 3100}, {6000, 3200}, {6000, 3500}, 
        {6000, 4000}, {6000, 4100}, {6000, 4200}, {6000, 4300}, {6000, 4500}, {6000, 5000}, {6000, 5100}, {6000, 5200}, {6000, 5300}, {6000, 5500}, 
        {6000, 6000} };

    char in_path[100];
    size_t m, n;

    for (size_t i = 0; i < sizes.size(); i++) {
        m = sizes[i].i;
        n = sizes[i].j;

        std::vector<double> A_matr(m * n);
        matrix_t Data_matr = { &A_matr[0], m, n };
        random_matrix(Data_matr);
        sprintf(in_path, "./LocalData/in/%d_%d.in", m, n);
        matrix_to_file(Data_matr, in_path);
    }

    return 0;
}

void single_dgesvj(long long n) {

    double t_mkl1, t_mkl2, t_jac1, tjac2, t_result;
    size_t square_matrix_size;
    std::map<int, double> mkl_seq_times;

    std::cout << "Singular decomposition of double square matrix by MKL parallel dgesvj function" << std::endl;

    square_matrix_size = n * n;
    std::vector<double> A_square(square_matrix_size);
    matrix_t Data_square_matr = { &A_square[0], n, n };
    matrix_from_file(Data_square_matr, "./LocalData/in/100_100.in");

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
    //https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/lapack-least-squares-and-eigenvalue-problem/lapack-least-squares-eigenvalue-problem-driver/singular-value-decomposition-lapack-driver/gesvj.html
    dgesvj(joba, jobu, jobv, &n, &n, &Data_square_matr.ptr[0], &lda, &S_vect.ptr[0], &mv, &V_matr.ptr[0], &ldv, &workspace[0], &lwork, &info);
    t_mkl2 = omp_get_wtime();
    t_result = t_mkl2 - t_mkl1;
    mkl_seq_times[n] = t_result;

    vector_to_file(S_vect, "./LocalData/out/SVD/dgesvj/square/A.to");
    matrix_to_file(Data_square_matr, "./LocalData/out/SVD/dgesvj/square/U.to");
    matrix_to_file(V_matr, "./LocalData/out/SVD/dgesvj/square/V.to");
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
    int n = A.cols;

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

void compute_params_to_file(std::vector<compute_params> params, const char* path) {
    std::ofstream output(path);
    output << "|rows|\t|cols|\t|threads|\t|iterations|\t|time|" << "\n\n";

    for (size_t i = 0; i < params.size(); i++) {
        output << std::fixed << params[i].m << "\t"
            << std::fixed << params[i].n << "\t"
            << std::fixed << params[i].threads << "\t"
            << std::fixed << params[i].iterations << "\t"
            << std::fixed << params[i].time << "\t"
            << "\n";
    }
    output.close();
}