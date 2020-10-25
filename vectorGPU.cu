#include <iostream>
#include <math.h>

__global__
void add(int n, float *x, float *y, float *z)
{
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride)
      z[i] = x[i] + y[i];
}
// void dot(int n, float *x, float *y, float *c){
//     for(int i = 0; i < n; i++){
//         *c += x[i] * y[i];
//     }
// }

int main(void){
    int N = 1<<20;

    float *x, *y, *z, *c;
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));
    cudaMallocManaged(&z, N * sizeof(float));
    cudaMallocManaged(&c, sizeof(float));

    for(int i = 0; i < N; i++){
        x[i] = 1;
        y[i] = 1;
    }

    int blockSize = 256;
        int numBlocks = (N + blockSize - 1) / blockSize;
        add<<<numBlocks, blockSize>>>(N, x, y, z);
    // dot(N, x, y, c);

    cudaDeviceSynchronize();

    float maxError = 0;
    for(int i = 0; i < N; i++){
        maxError = fmax(maxError, fabs(z[i] - 2));
    }
    std::cout << "Error: " << maxError << std::endl;
    std::cout << "dot product: " << *c << std::endl;

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    cudaFree(c);

    return 0;
}