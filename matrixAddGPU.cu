#include <iostream>
#include <math.h>
#define N 131072
#define BLOCK_DIM 32

__global__ void matAdd(float *a, float *b, float *c){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if(i < N && j < N){
        c[i * N + j] = a[i * N + j] + b[i * N + j];
    }
}

int main(void){
    float *dev_x, *dev_y, *dev_z;
    float *x, *y, *z;
    long long size = N * N * sizeof(float);

    cudaMalloc((void **)&dev_x, size);
    cudaMalloc((void **)&dev_y, size);
    cudaMalloc((void **)&dev_z, size);

    x = (float *)malloc(size);
    y = (float *)malloc(size);
    z = (float *)malloc(size);

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            x[i*N + j] = 1;
            y[i*N + j] = 1;
        }
    }

    cudaMemcpy(dev_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, y, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_z, z, size, cudaMemcpyHostToDevice);

    // FIGURE OUT BLOCK_DIM
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((int)(N/block.x), (int)(N/block.y));

    matAdd<<<grid, block>>>(dev_x, dev_y, dev_z);

    cudaMemcpy(z, dev_z, size, cudaMemcpyDeviceToHost);

    printf("%f ", z[3]);

    float maxError = 0;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            maxError = fmax(maxError, fabs(z[i*N + j] - 2));
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