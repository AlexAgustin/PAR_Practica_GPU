#include "nn.h"
#include "nn_aux.h"
#include "ds.h"
#include "matrix.h"
#include "gpu_matrix.cuh"
#include <stdbool.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

void gpu_forward_pass(nn_t *nn, double *input, double ***A, double ***Z, int size_batch);

void gpu_back_prop(nn_t *nn, double *output, double ***A, double ***Z, double ***d_D, double ***d_d, int size_batch, double **loss, double ***D_aux, double ***E);

void gpu_update(nn_t *nn, double ***d_D, double ***d_d, double lr, int batch_size);

__device__ void matrix_mul_add(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols, double* d);

__device__ double *m_elem(double *m, int length, int x, int y);

__device__ void matrix_func(double *n, double *m, int m_rows, int m_cols, double (*func)(double));

__device__ void matrix_sub(double *c, double *a, double *b, int rows, int cols);

__device__ void matrix_mul_dot(double *c, double *a, double *b, int rows, int cols);

__device__ double *matrix_transpose(double *m, int rows, int cols);

__device__ void matrix_mul(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols);

__device__ void matrix_free(double *m);

__device__ void matrix_sum(double *c, double *a, double *b, int rows, int cols);

__device__ void matrix_mul_cnt(double *m, int rows, int cols, double cnt);

__device__ void matrix_zero(double *m, int rows, int cols);