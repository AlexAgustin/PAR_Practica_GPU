#include "nn.h"
#include "nn_aux.h"
#include "ds.h"
#include "matrix.h"
#include "utils.h"
#include "gpu_matrix.cuh"
#include <stdbool.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

void gpu_forward_pass_test(nn_t *nn, double *input, double ***A);

__device__ void test_matrix_mul_add(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols, double* d);

__device__ double *test_m_elem(double *m, int length, int x, int y);

__device__ void test_matrix_func(double *n, double *m, int m_rows, int m_cols, double (*func)(double));