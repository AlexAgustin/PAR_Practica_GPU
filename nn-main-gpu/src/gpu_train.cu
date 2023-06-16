#include "gpu_train.cuh"
#include <assert.h>

#ifdef TIMING
    #include <time.h>
    #include "utils.h"
#endif

//Kernel de la funcion forward_pass
__global__ void forward_pass_kernel(nn_t *nn, double *input, double ***A, double ***Z, int size_batch) {
    //Se identifica el bloque al que pertenece el kernel
    int col = blockIdx.x;

    int i;

    //Se inicializan los valores de entrada
    for(i = 0; i < nn[col].layers_size[0]; i++){
        A[col][0][i] = input[i];
    }
    
    //Se ejecutan las operaciones matriciales pertinentes
    for(i = 1; i < nn[col].n_layers; i++){
        matrix_mul_add(Z[col][i], nn[col].WH[i - 1], A[col][i - 1],  nn[col].layers_size[i], nn[col].layers_size[i - 1], nn[col].layers_size[i - 1], 1, nn[col].BH[i - 1]);  
        matrix_func(A[col][i], Z[col][i], nn[col].layers_size[i], 1, nn[col].activation_ptr[i - 1]);
        matrix_func(Z[col][i], Z[col][i], nn->layers_size[i], 1, nn[col].dactivation_ptr[i - 1]);
    }
}

//Kernel de la back_prop
__global__ void back_prop_kernel(nn_t *nn, double *output, double ***A, double ***Z, double ***d_D, double ***d_d, int size_batch, double **loss, double ***E, double ***D_aux){
    //Se identifica el bloque al que pertenece el kernel
    int col = blockIdx.x;

    int i;
    double *T;
    int n_l;
    int *l_s;

    n_l = nn[col].n_layers;
    l_s = nn[col].layers_size;

    *loss[col] = nn[col].loss(A[col][n_l - 1], output, l_s[n_l - 1]);

    //Se ejecutan las operaciones matriciales pertinentes
    matrix_sub(E[col][n_l - 2], A[col][n_l - 1], output, l_s[n_l - 1], 1);
    matrix_mul_dot(E[col][n_l - 2], E[col][n_l - 2], Z[col][n_l - 1], l_s[n_l - 1], 1);  

    T = matrix_transpose(A[col][n_l - 2], l_s[n_l - 2], 1); 
    matrix_mul(D_aux[col][n_l - 2], E[col][n_l - 2], T, l_s[n_l - 1], 1, 1, l_s[n_l - 2]);
    matrix_free(T);

    matrix_sum(d_D[col][n_l - 2], d_D[col][n_l - 2], D_aux[col][n_l - 2], l_s[n_l - 1], l_s[n_l - 2]);
    matrix_sum(d_d[col][n_l - 2], d_d[col][n_l - 2], E[col][n_l - 2], l_s[n_l - 1], 1);

    for (i = n_l - 2; i > 0; i--) {
            
        T = matrix_transpose(nn[col].WH[i], l_s[i + 1], l_s[i]);
        matrix_mul(E[col][i - 1], T, E[col][i], l_s[i], l_s[i + 1], l_s[i + 1], 1);
        matrix_free(T);

        matrix_mul_dot(E[col][i - 1], E[col][i - 1], Z[col][i], l_s[i], 1);

        matrix_mul(D_aux[col][i - 1], E[col][i - 1], A[col][i - 1], l_s[i], 1, 1, l_s[i - 1]);

        matrix_sum(d_D[col][i - 1], d_D[col][i - 1], D_aux[col][i - 1], l_s[i], l_s[i - 1]);
        matrix_sum(d_d[col][i - 1], d_d[col][i - 1], E[col][i - 1], l_s[i], 1);
    }

}

__global__ void update_kernel(nn_t *nns, double ***d_D, double ***d_d, double lr, int batch_size){
    //Se identifica el bloque al que pertenece el kernel
    int col = blockIdx.x;
    
    int i;

    //Se ejecutan las operaciones matriciales pertinentes
    for(i = 0; i < nns[col].n_layers - 1; i++){

        matrix_mul_cnt(d_D[col][i], nns[col].layers_size[i + 1], nns[col].layers_size[i],  lr * (1.0 / batch_size));
        matrix_mul_cnt(d_d[col][i], nns[col].layers_size[i + 1], 1,  lr * (1.0 / batch_size));
        matrix_sub(nns[col].WH[i], nns[col].WH[i], d_D[col][i],  nns[col].layers_size[i + 1], nns[col].layers_size[i]);
        matrix_sub(nns[col].BH[i], nns[col].BH[i], d_d[col][i],  nns[col].layers_size[i + 1], 1);
        matrix_zero(d_D[col][i], nns[col].layers_size[i + 1], nns[col].layers_size[i]);
        matrix_zero(d_d[col][i], nns[col].layers_size[i + 1], 1);
    }
}

/*-----------------------------------------------------------------------------------------------------------*/

void gpu_forward_pass(nn_t *nn, double *input, double ***A, double ***Z, int size_batch){
    //Se general un total de size_batch hilos, cada uno en un bloque
    int thr_per_block = 1;
    int blk_in_grid = ceil((float)size_batch / thr_per_block);

    forward_pass_kernel<<<blk_in_grid, thr_per_block>>>(nn,input,A,Z,size_batch);
}

void gpu_back_prop(nn_t *nn, double *output, double ***A, double ***Z, double ***d_D, double ***d_d, int size_batch, double **loss, double ***D_aux, double ***E){
    //Se general un total de size_batch hilos, cada uno en un bloque
    int thr_per_block = 1;
    int blk_in_grid = ceil((float)size_batch / thr_per_block);

    back_prop_kernel<<<blk_in_grid, thr_per_block>>>(nn,output,A,Z,d_D,d_d,size_batch,loss,E,D_aux);

}

void gpu_update(nn_t *nn, double ***d_D, double ***d_d, double lr, int batch_size){
    //Se general un total de size_batch hilos, cada uno en un bloque
    int thr_per_block = 1;
    int blk_in_grid = ceil((float)batch_size / thr_per_block);

    update_kernel<<<blk_in_grid, thr_per_block>>>(nn,d_D,d_d,lr,batch_size);

}

/*---------------------------------------------------------------------------------------------------------------------------------*/

__device__ void matrix_mul_add(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols, double *d){

    int i, col, row;
    double sum;

    for (row = 0; row < a_rows; row++) {
        for(col = 0; col < b_cols; col++) {
            sum = 0.0;
            for (i = 0; i < a_cols; i++) {
                sum += *m_elem(a, a_cols, row, i) * *m_elem(b, b_cols, i, col);
            }
            *m_elem(c, b_cols, row, col) = sum + *m_elem(d, b_cols, row, col);
        }
    }
}

__device__ double *m_elem(double *m, int length, int x, int y){

    return (double*)&m[length * x + y];
}

__device__ void matrix_func(double *n, double *m, int rows, int cols, double (*func)(double)){
    
    int col, row;

    for (row = 0; row < rows; row++){
        for(col = 0; col < cols; col++){
            *m_elem(n, cols, row, col) = func(*m_elem(m, cols, row, col));
        }
    }
}

__device__ void matrix_sub(double *c, double *a, double *b, int rows, int cols){

    int col, row;
    double sum;

    for (row = 0; row < rows; row++) {
        for(col = 0; col < cols; col++) {
            sum = *m_elem(a, cols, row, col) - *m_elem(b, cols, row, col);
            *m_elem(c, cols, row, col) = sum;
        }
    }
}

__device__ void matrix_mul_dot(double *c, double *a, double *b, int rows, int cols){

    int col, row;
    double prod;

    for (row = 0; row < rows; row++) {
        for(col = 0; col < cols; col++) {
            prod = *m_elem(a, cols, row, col) * *m_elem(b, cols, row, col);
            *m_elem(c, cols, row, col) = prod;
        }
    }
}

__device__ double *matrix_transpose(double *m, int rows, int cols){

    double *m_t;
    int i, j;

    if ((m_t = (double*)malloc(rows * cols * sizeof(double))) == NULL) {
        return(NULL);
    }

    for (i = 0; i < rows; i++){
        for (j = 0; j < cols; j++){
            *m_elem(m_t, rows, j, i) = *m_elem(m, cols, i, j);
        }
    }
    
    return(m_t);
}

__device__ void matrix_mul(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols){

    assert(a_cols == b_rows);

    int i, col, row;
    double sum;

#ifdef TIMING
    int res_time;
    struct timespec t1, t2;
    clockid_t clk_id = CLOCK_MONOTONIC;
    res_time = clock_gettime(clk_id, &t1);
#endif

    for (row = 0; row < a_rows; row++) {
        for(col = 0; col < b_cols; col++) {
            sum = 0.0;
            for (i = 0; i < a_cols; i++) {
                sum += *m_elem(a, a_cols, row, i) * *m_elem(b, b_cols, i, col);
            }
            *m_elem(c, b_cols, row, col) = sum;
        }
    }

#ifdef TIMING
    res_time = clock_gettime(clk_id, &t2);
    printf("Matrix mul execution time: %ld us \n", diff_time(t2, t1));
#endif

}

__device__ void matrix_free(double *m){

    if (m != NULL)
        free(m);
}

__device__ void matrix_sum(double *c, double *a, double *b, int rows, int cols){

    int  col, row;
    double sum;

    for (row = 0; row < rows; row++) {
        for(col = 0; col < cols; col++) {
            sum = *m_elem(a, cols, row, col) + *m_elem(b, cols, row, col);
            *m_elem(c, cols, row, col) = sum;
        }
    }
}

__device__ void matrix_mul_cnt(double *m, int rows, int cols, double cnt){

    int col, row;

    for (row = 0; row < rows; row++) {
        for(col = 0; col < cols; col++) {
            *m_elem(m, cols, row, col) *= cnt;
        }
    }
}

__device__ void matrix_zero(double *m, int rows, int cols){

    int col, row;

    for (row = 0; row < rows; row++) {
        for(col = 0; col < cols; col++) {
            *m_elem(m, cols, row, col) = 0.0;
        }
    }
}