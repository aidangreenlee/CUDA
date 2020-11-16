#include <iostream>
#include <math.h>

__global__
void add(int n, float *x, float *y, float *z){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  z[index] = x[index] + y[index];
}

int main(void){
    int N = 1<<20;

    float *x, *y, *z;
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));
    cudaMallocManaged(&z, N * sizeof(float));

    for(int i = 0; i < N; i++){
        x[i] = 1;
        y[i] = 2;
    }

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, x, y, z);

    cudaDeviceSynchronize();

    float maxError = 0;
    for(int i = 0; i < N; i++){
        maxError = fmax(maxError, fabs(z[i] - 3));
    }
    std::cout << "Error: " << maxError << std::endl;

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    return 0;
}