#ifndef __TEST_H
#define __TEST_H
#include <assert.h>

#ifdef TIMING
    #include <time.h>
    #include "utils.h"
#endif

#include "nn.h"
#include "nn_aux.h"
#include "ds.h"
#include "matrix.h"
#include "gpu_matrix.cuh"

void forward_pass_test(nn_t *nn, double *input, double ***A);

float precision(int tp, int fp);

float recall(int tp, int fn);

float f1(float p, float r);

#endif
