#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "matrix.h"
#include "nn_aux.h"
#include "globals.h"
#include "gpu_train.cuh"

#ifdef TIMING
    #include <time.h>
    #include "utils.h"
#endif

double **alloc_matrix_2v(int n_layers, int *size, int *size_prev, double (*init_weight_ptr)(void)){

    double **m;
    int i, j;

    if ((m = (double**)malloc(n_layers * sizeof(double*))) == NULL) {
        return(NULL);
    }

    for (i=0; i < n_layers; i++)
        if ((m[i] = (double*)malloc(size[i] * size_prev[i] * sizeof(double))) == NULL) {
            matrix_free_2D(m, n_layers);
            return(NULL);
        }

    for (i = 0; i < n_layers; i++){
        for (j =0; j < size[i] * size_prev[i]; j++){
            m[i][j] = init_weight_ptr();
        }
    }

    return(m);
}

double **alloc_matrix_1v(int n_layers, int *size, double (*init_weight_ptr)(void)){

    double **m;
    int i, j;

    if ((m = (double**)malloc(n_layers * sizeof(double*))) == NULL) {
        return(NULL);
    }

    for (i=0; i < n_layers; i++)
        if ((m[i] = (double*)malloc(size[i] * sizeof(double))) == NULL) {
            matrix_free_2D(m, n_layers);
            return(NULL);
        }

    for (i = 0; i < n_layers; i++){
        for (j =0; j < size[i]; j++){
            m[i][j] = init_weight_ptr();
        }
    }

    return(m);
}

double *alloc_array(int length){

    double *v;
    int i;

    if ((v = (double*)malloc(length* sizeof(double))) == NULL) {
        return(NULL);
    }

    for (i = 0; i < length; i++){
        v[i] = 0.0;
    }
    
    return(v);
}

double *alloc_matrix(int rows, int cols){

    double *m;
    int i;

    if ((m = (double*)malloc(rows * cols * sizeof(double))) == NULL) {
        return(NULL);
    }

    for (i = 0; i < rows * cols; i++){
        m[i] = 0.0;
    }
    
    return(m);
}

void matrix_free_2D(double **m, int n_layers){

    int i;

    for (i=0; i < n_layers; ++i) {
        if (m[i] != NULL) {
            free(m[i]);
        }
    }
    free(m);
}

void print_matrix(double *m, int m_rows, int m_cols){
    
    int col, row;
    printf("%d %d\n", m_rows, m_cols);
    for (row = 0; row < m_rows; row++){
        for(col = 0; col < m_cols; col++){
            printf("(%d %d) %.*lf ", row, col, 10, *m_elem(m, m_cols, row, col));
        }
        printf("\n");
    }
    printf("\n");
}
