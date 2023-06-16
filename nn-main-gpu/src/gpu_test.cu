#include "gpu_test.cuh"

//Kernel de la funcion forward_pass en el momento de validacion
__global__ void forward_pass_test_kernel(nn_t *nn, double *input, double ***A){
    //Se identifica el bloque al que pertenece el kernel
    int col = blockIdx.x;

    int i;

    //Se inicializan los valores de entrada
    for(i = 0; i < nn[col].layers_size[0]; i++){
        A[col][0][i] = input[i];
    }
    
    //Se ejecutan las operaciones matriciales pertinentes
    for(i = 1; i < nn[col].n_layers; i++){
        test_matrix_mul_add(A[col][i], nn[col].WH[i - 1], A[col][i - 1],  nn[col].layers_size[i], nn[col].layers_size[i - 1], nn[col].layers_size[i - 1], 1, nn[col].BH[i - 1]);  
        test_matrix_func(A[col][i], A[col][i], nn[col].layers_size[i], 1, nn[col].activation_ptr[i - 1]);
    }
}

/*---------------------------------------------------------------------------------------------------*/

void gpu_forward_pass_test(nn_t *nn, double *input, double ***A){
    //Se general un total de size_batch hilos, cada uno en un bloque
    int thr_per_block = 1;
    int blk_in_grid = ceil((float)batches / thr_per_block);

    forward_pass_test_kernel<<<blk_in_grid, thr_per_block>>>(nn,input,A);
    
}

/*---------------------------------------------------------------------------------------------------*/


__device__ void test_matrix_mul_add(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols, double *d){

    int i, col, row;
    double sum;

    for (row = 0; row < a_rows; row++) {
        for(col = 0; col < b_cols; col++) {
            sum = 0.0;
            for (i = 0; i < a_cols; i++) {
                sum += *test_m_elem(a, a_cols, row, i) * *test_m_elem(b, b_cols, i, col);
            }
            *test_m_elem(c, b_cols, row, col) = sum + *test_m_elem(d, b_cols, row, col);
        }
    }
}

__device__ double *test_m_elem(double *m, int length, int x, int y){

    return (double*)&m[length * x + y];
}

__device__ void test_matrix_func(double *n, double *m, int rows, int cols, double (*func)(double)){
    
    int col, row;

    for (row = 0; row < rows; row++){
        for(col = 0; col < cols; col++){
            *test_m_elem(n, cols, row, col) = func(*test_m_elem(m, cols, row, col));
        }
    }
}