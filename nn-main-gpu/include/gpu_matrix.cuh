#include <stdbool.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

/* Macro for checking cuda errors following a cuda launch or api call
 Taken from: https://gist.github.com/jefflarkin/5390993 */
#define cudaCheckError() {                                          \
    cudaError_t e=cudaGetLastError();                                 \
    if(e!=cudaSuccess) {                                              \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
    exit(0); \
    }                                                                 \
}

//---------------------------------------------------------------------------------------------------------//
double **gpu_alloc_matrix_2v(int n_layers, int *size, int *size_prev, double (*init_weight_ptr)(void));

double **gpu_alloc_matrix_1v(int n_layers, int *size, double (*init_weight_ptr)(void));

void gpu_matrix_free_2D(double **m, int n_layers);

void gpu_matrix_free(double *m);

double *gpu_m_elem(double *m, int length, int x, int y);