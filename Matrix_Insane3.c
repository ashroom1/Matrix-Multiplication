#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <pthread.h>
#include <immintrin.h>

#define DEFAULT_MATRIX_SIZE 512
#define MIN_VALUE 0.001
#define MAX_VALUE 100.0
#define MAX_THREADS_1 100
#define MAX_THREADS_2 128    // Perfectly divides min direct multiply matrix size

#define MATRIX_DC_M(curr_size, t_arr, thr_func, ins, T, Thread_created) n_threads >= MAX_THREADS_1 ? (T = Matrix_DC(ins)) : (Thread_created = !pthread_create(t_arr, NULL, Matrix_DC_t, &ins));

int_fast32_t n_threads = 0;

struct Matrix_Instance
{
    float **M1;
    float **M2;
    int_fast32_t size;
};

struct Matrix_RC
{
    float *Row1;
    float *Row2;
    int_fast32_t steps;
};


float **Matrix_DC(struct Matrix_Instance Ins);

void* Matrix_DC_t(void *Ins)
{
    struct Matrix_Instance *temp;
    temp = (struct Matrix_Instance *) Ins;

    struct Matrix_Instance I1 = {temp->M1, temp->M2, temp->size};

    return Matrix_DC(I1);
}


void* Matrix_RC_t(void *RC)
{
    struct Matrix_RC *Ins = (struct Matrix_RC *) RC;

    register __m256 v1, v2;
    __m256 temp = {0};
    float *temp2;
    float *to_return = calloc(1, sizeof(float));

    for (int_fast32_t k = 0; k < Ins->steps; ++k) {
        v1 = _mm256_loadu_ps(Ins->Row1 + (8 * k));
        v2 = _mm256_loadu_ps(Ins->Row2 + (8 * k));
        temp = _mm256_add_ps(temp, _mm256_mul_ps(v1, v2));
    }
    temp2 = (float *) &temp;
    for (int_fast32_t l = 0; l < 8; ++l)
        *to_return += temp2[l];

    return to_return;
}


float float_rand(float min, float max )
{
    //return rand();	
    return min + rand() / (float) RAND_MAX * ( max - min );      /* [min, max] */
}

float **matrix_multiply_t(float **M1, float **M2, int_fast32_t size) {

    float **res = (float **) malloc(size * sizeof(float *));
    for (int_fast32_t i = 0; i < size; ++i)
        res[i] = calloc(size, sizeof(float));

    float **M2_t = (float **) malloc(size * sizeof(float *));
    for (int_fast32_t j = 0; j < size; ++j)
        M2_t[j] = (float *) malloc(size * sizeof(float));

    for (int_fast32_t l = 0; l < size; ++l)
        memcpy(M2_t[l], M2[l], size * sizeof(float));

    int_fast32_t steps = size / 8;        //Can make it global with Macro on when to divide

    struct Matrix_RC Ins[MAX_THREADS_2];
    pthread_t threads[MAX_THREADS_2];

    for (int_fast32_t i = 0; i < size; ++i) {
        for (int_fast32_t k = 0; k < size / MAX_THREADS_2; ++k) {
            for (int_fast32_t j = 0; j < MAX_THREADS_2; ++j) {
                Ins[j].Row1 = *(M1 + i);
                Ins[j].Row2 = *(M2 + (k * MAX_THREADS_2) + j);
                Ins[j].steps = steps;
                pthread_create(threads + j, NULL, Matrix_RC_t, Ins + j);
            }
            float *sum;
            for (int_fast32_t l = 0; l < MAX_THREADS_2; ++l) {
                pthread_join(threads[k], (void *)&sum);
                res[i][(k * MAX_THREADS_2) + l] = *sum;
            }
        }
    }
    free(M2_t);

    return res;
}

float** Matrix_Add(float **M1, float **M2, int_fast32_t size)
{
    float **res = (float **) malloc(size * sizeof(float *));
    for (int_fast32_t k = 0; k < size; ++k)
        res[k] = (float *) malloc(size * sizeof(float));


    for (int_fast32_t i = 0; i < size; ++i)
        for (int_fast32_t j = 0; j < size; ++j)
            res[i][j] = M1[i][j] + M2[i][j];

    return res;
}

float **Matrix_DC(struct Matrix_Instance Ins)
{
    if (Ins.size > 512)
    {
        int_fast32_t new_size = Ins.size / 2;
        register float **a11, **a12, **a21, **a22, **b11, **b12, **b21, **b22;
        pthread_t threads[8];

        a11 = (float **) malloc((new_size) * sizeof(float *));
        a12 = (float **) malloc((new_size) * sizeof(float *));
        a21 = (float **) malloc((new_size) * sizeof(float *));
        a22 = (float **) malloc((new_size) * sizeof(float *));
        b11 = (float **) malloc((new_size) * sizeof(float *));
        b12 = (float **) malloc((new_size) * sizeof(float *));
        b21 = (float **) malloc((new_size) * sizeof(float *));
        b22 = (float **) malloc((new_size) * sizeof(float *));

        for (int_fast32_t i = 0; i < new_size; ++i) {
            a11[i] = *(Ins.M1 + i);
            a12[i] = (*(Ins.M1 + i)) + (new_size);
            a21[i] = *(Ins.M1 + (new_size) + i);
            a22[i] = (*(Ins.M1 + new_size)) + new_size;
            b11[i] = *(Ins.M2 + i);
            b12[i] = (*(Ins.M2 + i)) + new_size;
            b21[i] = *(Ins.M2 + new_size + i);
            b22[i] = (*(Ins.M2 + new_size)) + new_size;
        }

        struct Matrix_Instance I1 = {a11, b11, new_size};
        struct Matrix_Instance I2 = {a12, b21, new_size};
        struct Matrix_Instance I3 = {a11, b12, new_size};
        struct Matrix_Instance I4 = {a12, b22, new_size};
        struct Matrix_Instance I5 = {a21, b11, new_size};
        struct Matrix_Instance I6 = {a22, b21, new_size};
        struct Matrix_Instance I7 = {a21, b12, new_size};
        struct Matrix_Instance I8 = {a22, b22, new_size};

        float **T[8];
        _Bool Thread_created[8] = {0};

        MATRIX_DC_M(new_size, threads, Matrix_DC_t, I1, T[0], Thread_created[0])
        ++n_threads;
        MATRIX_DC_M(new_size, threads + 1, Matrix_DC_t, I2, T[1], Thread_created[1])
        ++n_threads;
        MATRIX_DC_M(new_size, threads + 2, Matrix_DC_t, I3, T[2], Thread_created[2])
        ++n_threads;
        MATRIX_DC_M(new_size, threads + 3, Matrix_DC_t, I4, T[3], Thread_created[3])
        ++n_threads;
        MATRIX_DC_M(new_size, threads + 4, Matrix_DC_t, I5, T[4], Thread_created[4])
        ++n_threads;
        MATRIX_DC_M(new_size, threads + 5, Matrix_DC_t, I6, T[5], Thread_created[5])
        ++n_threads;
        MATRIX_DC_M(new_size, threads + 6, Matrix_DC_t, I7, T[6], Thread_created[6])
        ++n_threads;
        MATRIX_DC_M(new_size, threads + 7, Matrix_DC_t, I8, T[7], Thread_created[7])
        ++n_threads;

        void *temp;
        for (int_fast32_t k = 0; k < 8; ++k)
            if (Thread_created[k]) {
                pthread_join(threads[k], &temp);
                T[k] = (float **) temp;
                --n_threads;
            }

        float **c11 = Matrix_Add(T[0], T[1], new_size);
        float **c12 = Matrix_Add(T[2], T[3], new_size);
        float **c21 = Matrix_Add(T[4], T[5], new_size);
        float **c22 = Matrix_Add(T[6], T[7], new_size);


        float **res = (float **) malloc(Ins.size * sizeof(float *));
        for (int_fast32_t i = 0; i < Ins.size; ++i)
            res[i] = (float *) malloc(Ins.size * sizeof(float));

        for (int_fast32_t i = 0; i < new_size; ++i)
            for (int_fast32_t j = 0; j < new_size; ++j)
                res[i][j] = c11[i][j];

        for (int_fast32_t i = 0; i < new_size; ++i)
            for (int_fast32_t j = new_size; j < Ins.size; ++j)
                res[i][j] = c12[i][j - new_size];

        for (int_fast32_t i = new_size; i < Ins.size; ++i)
            for (int_fast32_t j = 0; j < new_size; ++j)
                res[i][j] = c21[i - new_size][j];

        for (int_fast32_t i = new_size; i < Ins.size; ++i)
            for (int_fast32_t j = new_size; j < Ins.size; ++j)
                res[i][j] = c22[i - new_size][j - new_size];

        for (int_fast32_t i = 0; i < new_size; ++i) {
            free(T[0][i]);
            free(T[1][i]);
            free(T[2][i]);
            free(T[3][i]);
            free(T[4][i]);
            free(T[5][i]);
            free(T[6][i]);
            free(T[7][i]);
            free(c11[i]);
            free(c22[i]);
            free(c12[i]);
            free(c21[i]);
        }
        free(T[0]);
        free(T[1]);
        free(T[2]);
        free(T[3]);
        free(T[4]);
        free(T[5]);
        free(T[6]);
        free(T[7]);
        free(a11);
        free(a12);
        free(a21);
        free(a22);
        free(b11);
        free(b12);
        free(b21);
        free(b22);
        free(c11);
        free(c12);
        free(c21);
        free(c22);


        return res;
    }
    else
        return matrix_multiply_t(Ins.M1, Ins.M2, Ins.size);
}


void print_matrix(float **M, int m, int n)
{
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j)
            printf("%f ", M[i][j]);
        putchar('\n');
    }
}

int main(int argc, char *argv[]) {


    clock_t start1 = clock();

    int_fast32_t MatrixSize = argc < 2 ? DEFAULT_MATRIX_SIZE : atoi(argv[1]);

    float **M1 = (float **) malloc(MatrixSize * sizeof(float *));
    for (int_fast32_t l = 0; l < MatrixSize; ++l)
        M1[l] = calloc(MatrixSize, sizeof(float));

    float **M2 = (float **) malloc(MatrixSize * sizeof(float *));
    for (int_fast32_t l = 0; l < MatrixSize; ++l)
        M2[l] = calloc(MatrixSize, sizeof(float));

    for (int_fast32_t i = 0; i < MatrixSize; ++i)
        for (int_fast32_t j = 0; j < MatrixSize; ++j)
            M1[i][j] = float_rand(MIN_VALUE, MAX_VALUE);

    for (int_fast32_t i = 0; i < MatrixSize; ++i)
        for (int_fast32_t j = 0; j < MatrixSize; ++j)
            M2[i][j] = float_rand(MIN_VALUE, MAX_VALUE);


    struct Matrix_Instance Ins = {M1, M2, MatrixSize};
    float **res = Matrix_DC(Ins);

    FILE *file = fopen("Matrix_time.txt", "a+");
    fprintf(file, "Matrix size : %lu\nTime taken in seconds : %f\n\n", (unsigned long) MatrixSize, (double)(clock()-start1) / CLOCKS_PER_SEC);
    fclose(file);

    //print_matrix(res, MatrixSize, MatrixSize);

    return 0;
}