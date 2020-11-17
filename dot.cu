#include <iostream>
#define THREADS_PER_BLOCK 512
#define N 2048*2048


__global__ void dot(int *a, int *b, int *c){
    __shared__ int temp[THREADS_PER_BLOCK];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    temp[threadIdx.x] = a[index] * b[index];

    __syncthreads();

    if( threadIdx.x == 0){
        int sum = 0;
        for(int i = 0; i < THREADS_PER_BLOCK; i++){
            sum += temp[i];
        }
        atomicAdd(c, sum);
    }
}

int main(){
    int *a, *b;
    int *c, *dev_c;
    int *dev_a, *dev_b;
    int test = 0.0f;
    long long size = N * sizeof(int);

    cudaMalloc((void **)&dev_a, size);
    cudaMalloc((void **)&dev_b, size);
    cudaMalloc((void **)&dev_c, sizeof(int));

    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(sizeof(int));

    *c = 0;

    for(int i = 0; i < N; i++){
        a[i] = 1;
        b[i] = 1;
        test += a[i] * b[i];
    }

    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, sizeof(int), cudaMemcpyHostToDevice);

    dot<<< N/THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>(dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    
    std::cout << "dot product: " << *c << std::endl;
    std::cout << "test: " << test << std::endl;

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(a);
    free(b);
    free(c);
    return 0;
}