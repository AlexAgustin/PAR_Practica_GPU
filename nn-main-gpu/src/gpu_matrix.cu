#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include "gpu_matrix.cuh"
#include "matrix.h"

#define THR_PER_BLOCK 1024 

//Reserva memoria para la matriz DE DOS DIMENSIONES tanto en el host como en el dispositivo
double **gpu_alloc_matrix_2v(int n_layers, int *size, int *size_prev, double (*init_weight_ptr)(void)) {
    double **m;
    int i, j;

    // Reservar memoria para m en el host
    if ((m = (double**)malloc(n_layers * sizeof(double*))) == NULL) {
        return(NULL);
    }

    //Reservar memoria para las layers en el host
    for (i=0; i < n_layers; i++)
        if ((m[i] = (double*)malloc(size[i] * size_prev[i] * sizeof(double))) == NULL) {
            matrix_free_2D(m, n_layers);
            return(NULL);
        }

    //Inicializar la matriz en CPU
    for (i = 0; i < n_layers; i++){
        for (j =0; j < size[i] * size_prev[i]; j++){
            m[i][j] = init_weight_ptr();
        }
    }

    //Reservar memoria para d_m_h en el host
    double **d_m_h = (double**)malloc(n_layers * sizeof(double*));

    for (i = 0; i < n_layers; i++) {
        //Reservar memoria para las layers en el dispositivo
        cudaMalloc((void**)&(d_m_h[i]), size[i] * size_prev[i] * sizeof(double));
        cudaCheckError();
    }

    //Copiar los valores de CPU a GPU
    for (i = 0; i < n_layers; i++) {
        for (j = 0; j < size[i] * size_prev[i]; j++) {
            cudaMemcpy(d_m_h[i], &m[i][j], sizeof(double), cudaMemcpyHostToDevice);
            cudaCheckError();
        }
    }

    free(m);
    return(d_m_h);
}

//Reserva memoria para la matriz tanto en el host como en el dispositivo
double **gpu_alloc_matrix_1v(int n_layers, int *size, double (*init_weight_ptr)(void)) {

    double **m;
    int i, j;

    // Reservar memoria para m en el host
    if ((m = (double**)malloc(n_layers * sizeof(double*))) == NULL) {
        return(NULL);
    }

    //Reservar memoria para las layers en el host
    for (i=0; i < n_layers; i++)
        if ((m[i] = (double*)malloc(size[i] * sizeof(double))) == NULL) {
            matrix_free_2D(m, n_layers);
            return(NULL);
        }

    //Inicializar la matriz en CPU
    for (i = 0; i < n_layers; i++){
        for (j =0; j < size[i]; j++){
            m[i][j] = init_weight_ptr();
        }
    }

    //Reservar memoria para d_m en el host
    double **d_m_h = (double**)malloc(n_layers * sizeof(double*));

    for (i = 0; i < n_layers; i++) {

        //Reservar memoria para las layers en el dispositivo
        cudaMalloc((void**)&(d_m_h[i]), size[i] * sizeof(double));
        cudaCheckError();
    }

    //Copiar los valores de CPU a GPU
    for (i = 0; i < n_layers; i++) {
        for (j = 0; j < size[i]; j++) {
            cudaMemcpy(d_m_h[i], &m[i][j], sizeof(double), cudaMemcpyHostToDevice);
            cudaCheckError();
        }
    }

    free(m);
    return(d_m_h);
}

// Libera la memoria asignada a una matriz de dos dimensiones
void gpu_matrix_free_2D(double **m, int n_layers){

    int i;

    for (i=0; i < n_layers; ++i) {
        if (m[i] != NULL) {
            cudaFree(m[i]);
            cudaCheckError();
        }
    }
    cudaFree(m);
    cudaCheckError();
}

// Libera la memoria asiganda a M
void gpu_matrix_free(double *m){

    if (m != NULL){
        cudaFree(m);
        cudaCheckError();
    }
}

//Combierte los indices 2D a 1D para el acceso
double *gpu_m_elem(double *m, int length, int x, int y){
    return (double *)&m[length * x + y];
}
