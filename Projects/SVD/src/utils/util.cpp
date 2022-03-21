#include "util.hpp"
#include <cassert>
#include <math.h>
#include <limits>
#include "types.hpp"
#include <iostream>
#include <fstream>

bool isclose(double x, double y, double eps) { return fabs(x - y) <= eps * fabs(x + y); }

bool is_normalized(double v) { return fabs(v) >= std::numeric_limits<double>::min(); }

void sym_jacobi_coeffs(double x_ii, double x_ij, double x_jj, double* c, double* s) {
    if (!isclose(x_ij, 0)) {
    //if(x_ij != 0) {
        double tau, t, out_c;
        tau = (x_jj - x_ii) / (2 * x_ij);
        if (tau >= 0) {
            t = 1.0 / (tau + sqrt(1 + tau * tau));
        }
        else {
            t = -1.0 / (-tau + sqrt(1 + tau * tau));
            //t = -1.0 / (tau + sqrt(1 + tau * tau));
        }
        //t = sign(tau) / (abs(tau) + sqrt(1 + tau * tau));
        out_c = 1.0 / sqrt(1 + t * t);
        *c = out_c;
        *s = t * out_c;
    }
    else {
        *c = 1.0;
        *s = 0.0;
    }
}

void jrs_jacobi_coeffs(double x_ii, double x_ij, double x_jj, double* c, double* s) {
    if (!isclose(x_ij, 0)) {
        //if(x_ij != 0) {
        double tau, t, out_c;
        tau = (x_jj - x_ii) / (2 * x_ij);
        t = sign(tau) / (abs(tau) + sqrt(1 + (tau * tau)));
        //t = sign(tau) / (abs(tau) + sqrt(1 + tau * tau));
        out_c = 1.0 / sqrt(1 + (t * t));
        *c = out_c;
        *s = t * out_c;
    }
    else {
        *c = 1.0;
        *s = 0.0;
    }
}

int less(double x, double y) { return x < y; }

int greater(double x, double y) { return x > y; }

double sign(double x) { return (x > 0.0) ? 1.0 : -1.0; }

void reorder_decomposition(struct vector_t vals, struct matrix_t* matrices, int n_matrices, comparator cmp_fn) {
    double* s = vals.ptr;
    const int n_vals = vals.len;
    for (int i = 0; i < n_vals; ++i) {
        double s_last = s[i];
        int i_last = i;
        for (int j = i + 1; j < n_vals; ++j) {
            if (cmp_fn(s[j], s_last)) {
                s_last = s[j];
                i_last = j;
            }
        }
        if (i_last != i) {
            double tmp;
            tmp = s[i];
            s[i] = s[i_last];
            s[i_last] = tmp;

            for (int j = 0; j < n_matrices; ++j) {
                int rows = matrices[j].rows;
                int cols = matrices[j].cols;
                double* M = matrices[j].ptr;
                for (int k = 0; k < rows; ++k) {
                    tmp = M[k * cols + i];
                    M[k * cols + i] = M[k * cols + i_last];
                    M[k * cols + i_last] = tmp;
                }
            }
        }
    }
}

//строчное представление в памяти
void matrix_from_file(matrix_t A, const char *path) {
    std::ifstream file;
    file.open(path);

    for (size_t i = 0; i < A.rows * A.cols; i++)
        file >> A.ptr[i];

    file.close();
}

//строчное представление в памяти
void matrix_to_file(matrix_t A, const char *path) {
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

void vector_to_file(vector_t V, const char* path) {
    std::ofstream output(path);
    int n = V.len;

    for (size_t i = 0; i < n; ++i) {
        output << std::fixed << V.ptr[i] << " \t";
    }
    output.close();
}

bool modulus_pair(int num_blocks, int index, int iteration, int* i, int* j) {
    if (iteration < (num_blocks - (2 * index))) {
        *i = index + iteration;
        *j = num_blocks - index - 1;
        return false;
    }
    else if (iteration == (num_blocks - (2 * index))) {
        *i = index + iteration - (num_blocks / 2);
        *j = index + iteration;
        return true;
    }
    else {
        *i = (num_blocks / 2) - index;
        *j = index + iteration - (num_blocks / 2);
        return false;
    }
}

void round_robin(size_t* up, size_t* dn, size_t ThreadsNum) {
    size_t x = up[ThreadsNum - 1];
    size_t y = dn[0];

    for (size_t i = (ThreadsNum - 1); i > 0; i--)
        up[i] = up[i - 1];

    if (ThreadsNum > 2) {
        for (size_t i = 0; i < (ThreadsNum - 2); i++)
            dn[i] = dn[i + 1];
    }

    if (ThreadsNum > 1) {
        up[0] = y;
        dn[ThreadsNum - 2] = x;
    }
}

bool rrbjrs_column_limits(struct matrix_t A, size_t ThreadsNum, struct index_t* SOB) {
    size_t quotient = (A.cols / (2 * ThreadsNum)); //размер блоков при кратном деленим
    size_t remainder = A.cols % (2 * ThreadsNum); //число блоков размера (quotient + 1) при делении с остатком
    size_t multiple = (2 * ThreadsNum) - remainder; //число блоков размера quotient при кратном делении

    if (quotient < 2)
        return false;

    SOB[0].i = 0;
    SOB[0].j = quotient - 1;

    for (size_t i = 1; i < multiple; i++) {
        SOB[i].i = SOB[i - 1].j + 1;
        SOB[i].j = SOB[i].i + (quotient - 1);
    }

    if (remainder > 0) {
        for (size_t i = multiple; i < (multiple + remainder); i++) {
            SOB[i].i = SOB[i - 1].j + 1;
            SOB[i].j = SOB[i].i + quotient;
        }
    }
    return true;
}

void random_matrix(matrix_t A) {
    for (size_t i = 0; i < (A.rows * A.cols); i++)
        A.ptr[i] = rand() % 100;
}