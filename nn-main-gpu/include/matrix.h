#ifndef __MATRIX_H
#define __MATRIX_H

#include <stdbool.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

double **alloc_matrix_1v(int n_layers, int *size, double (*init_weight_ptr)(void));

double **alloc_matrix_2v(int n_layers, int *size, int *size_prev, double (*init_weight_ptr)(void));

double *alloc_array(int length);

double *alloc_matrix(int rows, int cols);

void matrix_free_2D(double **m, int n_layers);

void matrix_mul_trans(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols);

void print_matrix(double *m, int m_rows, int m_cols);

#endif
