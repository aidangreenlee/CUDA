#include <iostream>
#include <math.h>
#define N 2560
#define BLOCK_DIM 16

typedef struct{
    int rows;
    int cols;
    float* mat;
} matrix;

__global__ void matMult(matrix a, matrix b, matrix c){

    int n_col = threadIdx.x + blockIdx.x * blockDim.x;
    int n_row = threadIdx.y + blockIdx.y * blockDim.y;

    if(col < b.cols && row < a.rows){
        float sum = 0;
        for(int i = 0; i < a.cols; i++){
            sum += a.mat[row * a.cols + i] * b[b.cols * i + col];
        }
        c[kN * row + col] = sum;
    }
}

int main(void){
    int iN = 1000;
    int jN = 1000;
    int kN = 1000;
    float *dev_x, *dev_y, *dev_z;
    float *x, *y, *z;
    long long sizex = iN * kN * sizeof(float);
    long long sizey = iN * jN * sizeof(float);
    long long sizez = kN * jN * sizeof(float);

    cudaMalloc((void **)&dev_x, sizex);
    cudaMalloc((void **)&dev_y, sizey);
    cudaMalloc((void **)&dev_z, sizez);

    x = (float *)malloc(sizex);
    y = (float *)malloc(sizey);
    z = (float *)malloc(sizez);

    for(int k = 0; k < kN; k++){
        for(int j = 0; j < jN; j++){
            for(int i = 0; i < iN; i++){
                x[k * kN + i] = 1;
                y[j * jN + i] = 1;
            }
        }
    }

    cudaMemcpy(dev_x, x, sizex, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, y, sizey, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_z, z, sizez, cudaMemcpyHostToDevice);

    // FIGURE OUT BLOCK_DIM
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((int)(N/block.x), (int)(N/block.y));

    matMult<<<grid, block>>>(dev_x, dev_y, dev_z, iN, jN, kN);

    cudaMemcpy(z, dev_z, sizez, cudaMemcpyDeviceToHost);

    printf("%f ", z[3]);

    float maxError = 0;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            maxError = fmax(maxError, fabs(z[i * kN + j] - iN));
        }
    }
    std::cout << "Error: " << maxError << std::endl;

    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_z);
    free(x);
    free(y);
    free(z);
    return 0;
}