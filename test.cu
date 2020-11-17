#define N 512
#define BLOCK_DIM 512
#include <iostream>

__global__ void matrixAdd (int *a, int *b, int *c);

int main() {
    int a[N][N], b[N][N], c[N][N];
    int *dev_a, *dev_b, *dev_c;
    int size = N * N * sizeof(int);
    
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            a[i][j] = 1;
            b[i][j] = 1;
        }
    }

    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);

    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((int)ceil(N/dimBlock.x),(int)ceil(N/dimBlock.y));
    
    matrixAdd<<<dimGrid,dimBlock>>>(dev_a,dev_b,dev_c);
    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

    float maxError = 0;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            maxError = fmax(maxError, fabs(c[i][j] - 2));
        }
    }
    std::cout << "Error: " << maxError << std::endl;
    
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}

__global__ void matrixAdd (int *a, int *b, int *c) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index = col + row * N; 
    
    if (col < N && row < N) { 
        c[index] = a[index] + b[index]; 
    }
}