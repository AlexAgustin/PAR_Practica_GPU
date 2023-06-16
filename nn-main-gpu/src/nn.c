#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring>
#include "ds.h"
#include "nn.h"
#include "nn_aux.h"
#include "utils.h"
#include "matrix.h"
#include "test.h"
#include "train.h"
#include "globals.h"
#include "gpu_matrix.cuh"
#include "gpu_train.cuh"
#include "gpu_test.cuh"

void init_nn(nn_t *nn, int n_layers, int *layers_size){

    int i;

    nn->n_layers = n_layers;
    nn->layers_size = layers_size;
    nn->init_weight_ptr = init_weight_rnd;
    nn->activation_ptr= (activation_ptr_t*)malloc((nn->n_layers - 1) * sizeof(activation_ptr_t));
    nn->dactivation_ptr= (activation_ptr_t*)malloc((nn->n_layers - 1) * sizeof(activation_ptr_t));
    for(i = 0; i < n_layers - 1; i++){
        nn->activation_ptr[i] = sigmoid;
        nn->dactivation_ptr[i] = dSigmoid;
    }
    nn->loss = mse;
    nn->BH = alloc_matrix_1v(n_layers - 1, &layers_size[1], nn->init_weight_ptr);
    nn->WH = alloc_matrix_2v(n_layers - 1, &layers_size[1], &layers_size[0], nn->init_weight_ptr);
    
}

#ifdef CPU

void train(nn_t *nn, ds_t *ds, int epochs, int size_batch, double lr){

    int i, n, x, n_batches, min_batch;
    double **A, **Z, **D, **d;;
    int *order;
    double loss;
    struct timespec t1, t2;
    clockid_t clk_id = CLOCK_MONOTONIC;
  
    order = (int*)malloc(ds->n_samples * sizeof(int));
    
    A = alloc_matrix_1v(nn->n_layers, nn->layers_size, init_zero); 
    Z = alloc_matrix_1v(nn->n_layers, nn->layers_size, init_zero); 
    D = alloc_matrix_2v(nn->n_layers - 1, &(nn->layers_size[1]), &(nn->layers_size[0]), init_zero);
    d = alloc_matrix_1v(nn->n_layers - 1, &(nn->layers_size[1]), init_zero);
    
    n_batches = ds->n_samples / size_batch;

    for(i = 0; i < ds->n_samples; i++)
        order[i] = i;
    
    for (n=0; n < epochs;n++) {
            
        if(verbose)
            printf("Epoch %d/%d \n", n, epochs);
        
        loss = 0.0;
        shuffle(order, ds->n_samples);

        clock_gettime(clk_id, &t1);

        for (x = 0; x < n_batches; x++) {
            for(min_batch = (x * size_batch); min_batch < ((x + 1) * size_batch); min_batch++){
            
                i = order[min_batch];
                forward_pass(nn, &ds->inputs[i * ds->n_inputs], A, Z); 
                loss += back_prop(nn, &ds->outputs[i * ds->n_outputs], A, Z, D, d);
            }
            
            update(nn, D, d, lr, size_batch);
        }

        clock_gettime(clk_id, &t2);

        if(verbose)
            printf(" time: %ld us - loss: %.*f\n", diff_time(t2, t1), 12, loss / ds->n_samples);

    }

}

void test(nn_t *nn, ds_t *ds){
    
    int i;
    double **A;

    A = alloc_matrix_1v(nn->n_layers, nn->layers_size, init_zero);

    for(i = 0; i < ds->n_samples; i++){

        forward_pass_test(nn, &ds->inputs[i * ds->n_inputs], A);
    }

    // Precision
    // Recall
    // F1
}

#endif

#ifdef GPU

void train(nn_t *nn, ds_t *ds, int epochs, int size_batch, double lr){

    int i, n, x, n_batches, min_batch;
    int *order;
    struct timespec t1, t2;
    clockid_t clk_id = CLOCK_MONOTONIC;
    double ***A, ***Z, ***d_D, ***d_d, ***d_D_aux, ***d_E;
    nn_t *nns, *d_nns;
    ds_t *dss, *d_dss; 
    double total_loss;
    double *loss, **d_loss;

    //Reserva de memoria en el host de las matrices necesarias para la ejecucion del algoritmo
    A = (double***)malloc(size_batch * sizeof(double**));
    Z = (double***)malloc(size_batch * sizeof(double**));
    d_D = (double***)malloc(size_batch * sizeof(double**));
    d_d = (double***)malloc(size_batch * sizeof(double**));
    d_D_aux = (double***)malloc(size_batch * sizeof(double**));
    d_E = (double***)malloc(size_batch * sizeof(double**));

    //Reserva de datos en el host y device del array de datos del tipo nn_t (uno por cada conjunto de datos)
    nns = (nn_t*)malloc(size_batch * sizeof (nn_t));
    cudaMalloc((void**)&d_nns,size_batch * sizeof (nn_t));
    cudaCheckError();
    for (i = 0; i < size_batch; i++){
        memcpy(&(nns[i]),nn,sizeof(nn_t));
        cudaMemcpy(&(d_nns[i]),nn,sizeof(nn_t),cudaMemcpyHostToDevice);
        cudaCheckError();
    }

    //Reserva de datos en el host y device del array de datos del tipo ds_t (uno por cada conjunto de datos)
    dss = (ds_t*)malloc(size_batch * sizeof (ds_t));
    cudaMalloc((void**)&d_dss,size_batch * sizeof (ds_t));
    cudaCheckError();
    for (i = 0; i < size_batch; i++){
        memcpy(&(dss[i]),ds,sizeof(ds_t));
        cudaMemcpy(&(d_dss[i]),ds,sizeof(ds_t),cudaMemcpyHostToDevice);
        cudaCheckError();
    }

    //Reserva de memoria en GPU de las matrices
    for(i=0;i<size_batch;i++){
        A[i] = gpu_alloc_matrix_1v(nns[i].n_layers, nns[i].layers_size, init_zero); 
        Z[i] = gpu_alloc_matrix_1v(nns[i].n_layers, nns[i].layers_size, init_zero);
        d_D[i] = gpu_alloc_matrix_2v(nns[i].n_layers - 1, &(nns[i].layers_size[1]), &(nns[i].layers_size[0]), init_zero);
        d_d[i] = gpu_alloc_matrix_1v(nns[i].n_layers - 1, &(nns[i].layers_size[1]), init_zero);
        d_D_aux[i] = gpu_alloc_matrix_2v(nns[i].n_layers - 1, &(nns[i].layers_size[1]), &(nns[i].layers_size[0]), init_zero);
        d_E[i] = gpu_alloc_matrix_1v(nns[i].n_layers - 1, &(nns[i].layers_size[1]), init_zero);
    }

    order = (int*)malloc(ds->n_samples * sizeof(int));

    //Reserva de datos en el host y device del array de loss
    loss = (double*)malloc(size_batch * sizeof(double));
    cudaMalloc((void**)&d_loss,size_batch * sizeof(double));
    cudaCheckError();
    
    n_batches = ds->n_samples / size_batch;

    for(i = 0; i < ds->n_samples; i++)
        order[i] = i;
    
    for (n=0; n < epochs;n++) {
            
        if(verbose)
            printf("Epoch %d/%d \n", n, epochs);
        
        total_loss = 0.0;
        shuffle(order, ds->n_samples);

        clock_gettime(clk_id, &t1);
        for (x = 0; x < n_batches; x++) {
            for(min_batch = (x * size_batch); min_batch < ((x + 1) * size_batch); min_batch++){

                i = order[min_batch];
                //LLama a las funciones de GPU (cada una de ellas llamara a los kernels)
                gpu_forward_pass(d_nns, &ds->inputs[i * ds->n_inputs], A, Z,size_batch); 
                gpu_back_prop(d_nns, &ds->outputs[i * ds->n_outputs], A, Z, d_D, d_d, size_batch,d_loss,d_D_aux,d_E);
            }
            gpu_update(d_nns, d_D, d_d, lr, size_batch);
        }

        //Copia de los resultados de loss de la GPU a la CPU
        cudaMemcpy(loss,d_loss,size_batch * sizeof(double),cudaMemcpyDeviceToHost);
        cudaCheckError();
        /*for (i=0;i<size_batch;i++){
            cudaMemcpy(loss[i],d_loss[i],sizeof(double),cudaMemcpyDeviceToHost);
        }*/

        //Suma del loss total
        for (i=0;i<size_batch;i++){
            total_loss += loss[i];
        }
        clock_gettime(clk_id, &t2);

        if(verbose)
            printf(" time: %ld us - loss: %.*f\n", diff_time(t2, t1), 12, total_loss / ds->n_samples);
    }

}

void test(nn_t *nn, ds_t *ds){
    
    int i;
    double ***A;
    nn_t *nns, *d_nns;
    ds_t *dss, *d_dss; 

    //Reserva de memoria en el host de la matrice necesarias para la ejecucion del algoritmo
    A = (double***)malloc(batches * sizeof(double**));

    //Reserva de datos en el host y device del array de datos del tipo nn_t (uno por cada conjunto de datos)
    nns = (nn_t*)malloc(batches * sizeof (nn_t));
    for (i = 0; i < batches; i++){
        memcpy(&(nns[i]),nn,sizeof(nn_t));
    }
    cudaMalloc((void**)&d_nns,batches * sizeof (nn_t));
    cudaCheckError();
    cudaMemcpy(&d_nns,nns,batches * sizeof(nn_t),cudaMemcpyHostToDevice);
    cudaCheckError();

    //Reserva de datos en el host y device del array de datos del tipo ds_t (uno por cada conjunto de datos)
    dss = (ds_t*)malloc(batches * sizeof (ds_t));
    for (i = 0; i < batches; i++){
        memcpy(&(dss[i]),ds,sizeof(ds_t));
    }
    cudaMalloc((void**)&d_dss,batches * sizeof (ds_t));
    cudaCheckError();
    cudaMemcpy(&d_dss,dss,batches * sizeof(ds_t),cudaMemcpyHostToDevice);
    cudaCheckError();

    //Reserva de memoria en GPU de la matriz
    for(i=0;i<batches;i++)
        A[i] = gpu_alloc_matrix_1v(nns[i].n_layers, nns[i].layers_size, init_zero); 


    for(i = 0; i < ds->n_samples; i++){

        gpu_forward_pass_test(d_nns, &(d_dss[i/batches].inputs[i * ds->n_inputs]), A);
    }

    // Precision
    // Recall
    // F1
}

#endif

void print_nn(nn_t *nn){

    int i, j, k;
    
    printf("Layers (I/H/O)\n");

    for (i = 0; i < nn->n_layers; i++) {
        printf("%d ", nn->layers_size[i]);
    }
    printf("\n");
    
    printf("Hidden Biases\n ");

    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            printf("%lf ", nn->BH[i][j]);
        }
        printf("\n");
    }

    printf("Hidden Weights\n ");
    
    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            for(k = 0; k < nn->layers_size[i]; k++) {
                printf("%lf ", nn->WH[i][(j * nn->layers_size[i]) + k]);
            }
            printf("\n");
        }
    }

}

void import_nn(nn_t *nn, char *filename){

    int i, j, k;
    FILE *fd;

    if ((fd = fopen(filename,"r")) == NULL){
        perror("Error importing the model\n");
        exit(1);
    }
    
    fscanf(fd, "%d ", &n_layers);

    layers = (int*)malloc(n_layers * sizeof(int));

    for (i = 0; i < n_layers; i++) {
        fscanf(fd, "%d ", &(layers[i]));
    }

    init_nn(nn, n_layers, layers);
    
    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            fscanf(fd, "%lf ", &(nn->BH[i][j]));
        }
    }

    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            for(k = 0; k < nn->layers_size[i]; k++) {
                fscanf(fd, "%lf ", &(nn->WH[i][(j * nn->layers_size[i]) + k]));
            }
        }
    }
    fclose(fd);
}

void export_nn(nn_t *nn, char *filename){

    int i, j, k;
    FILE *fd;

    if ((fd = fopen(filename,"w")) == NULL){
        perror("Error exporting the model");
        exit(1);
    }
    
    fprintf(fd, "%d\n", nn->n_layers);

    for (i = 0; i < nn->n_layers; i++) {
        fprintf(fd, "%d ", nn->layers_size[i]);
    }
    fprintf(fd, "\n");
    
    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            fprintf(fd, "%lf ", nn->BH[i][j]);
        }
        fprintf(fd, "\n");
    }

    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            for(k = 0; k < nn->layers_size[i]; k++) {
                fprintf(fd, "%lf ", nn->WH[i][(j * nn->layers_size[i]) + k]);
            }
            fprintf(fd, "\n");
        }
    }
    fclose(fd);
}

/*void preparar_update(double ***D, double ***d_D, int size_batch, int n_layers, int *size, int *size_prev, double **D_mean, double ***d, double ***d_d, double **d_mean){
    int i,j,k;
    double suma = 0;

    //Traer los valores de D desde la GPU a la CPU
    for(i = 0; i < size_batch; i++){
        for(j = 0; j < n_layers; j++){
            for(k = 0; k < size[j] * size_prev[j]; k++){
                cudaMemcpy(&D[i][j][k], &d_D[i][j][k], sizeof(double), cudaMemcpyDeviceToHost);
                cudaCheckError();
            }
        }
    }

    //Calcular la media de los distintos D's
    for(j = 0; j < n_layers; j++){
        for(k = 0; k < size[j] * size_prev[j]; k++){
            for(i = 0; i < size_batch; i++){
                suma += D[i][j][k];
            }
            D_mean[j][k] = suma / size_batch;
        }
    }

    //Traer los valores de D desde la GPU a la CPU
    for(i = 0; i < size_batch; i++){
        for(j = 0; j < n_layers; j++){
            for(k = 0; k < size[j]; k++){
                cudaMemcpy(&d[i][j][k], &d_d[i][j][k], sizeof(double), cudaMemcpyDeviceToHost);
                cudaCheckError();
            }
        }
    }

    //Calcular la media de los distintos D's
    for(j = 0; j < n_layers; j++){
        for(k = 0; k < size[j]; k++){
            for(i = 0; i < size_batch; i++){
                suma += d[i][j][k];
            }
            d_mean[j][k] = suma / size_batch;
        }
    }
}*/