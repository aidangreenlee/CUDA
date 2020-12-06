#include <iostream>
#include <math.h>
#define N 102
#define BLOCK_DIM 32

typedef struct{
    int rows;
    int cols;
    int stride;
    float* mat;
    float* dev;
}matrix;

__global__ void matMult(matrix a, matrix b, matrix c){
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    float val = 0;
    if(row < a.rows && col < b.cols){
        for(int i = 0; i < a.cols; i++){
            val += a.dev[row * a.cols + i] * b.dev[i * b.cols + col];
        }
        c.dev[row * b.cols + col] = val;
    }
}

int main(void){
    cudaError_t error;
    matrix a;
    matrix b;
    matrix c;
    a.rows = N;
    a.cols = N;
    b.rows = N;
    b.cols = N;
    a.mat = new float[a.rows * a.cols];
    cudaMalloc(&a.dev, a.rows * a.cols * sizeof(float));

    b.mat = new float[b.rows * b.cols];
    cudaMalloc(&b.dev, b.rows * b.cols * sizeof(float));

    c.mat = new float[a.rows * b.cols];
    cudaMalloc(&c.dev, a.rows * b.cols * sizeof(float));

    for(int i = 0; i < a.cols; i++){
        for(int k = 0; k < a.rows; k++){
            a.mat[k * a.cols + i] = 1;
        }
        for(int j = 0; j < b.cols; j++){
            b.mat[i * b.cols + j] = 1;
        }
    }

    cudaMemcpy(a.dev, a.mat, a.rows * a.cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b.dev, b.mat, b.rows * b.cols * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((b.cols + block.x - 1)/block.x, (a.rows + block.y - 1)/block.y);

    matMult<<<grid, block>>>(a, b, c);

    error = cudaMemcpy(c.mat, c.dev, a.rows * b.cols * sizeof(float), cudaMemcpyDeviceToHost);
    printf("C device to host: %s\n", cudaGetErrorString(error));
    printf("%f\n", c.mat[3]);

    float maxError = 0;
    for(int k = 0; k < a.rows; k++){
        for(int j = 0; j < b.cols; j++){
            maxError = fmax(maxError, fabs(c.mat[k * b.cols + j] - a.cols));
        }
    }
    std::cout << "Error: " << maxError << std::endl;

    cudaFree(a.dev);
    cudaFree(b.dev);
    cudaFree(c.dev);
    delete [] a.mat;
    delete [] b.mat;
    delete [] c.mat;
    return 0;
}