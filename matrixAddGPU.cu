#include <iostream>
#define N (2048*2048)

__global__ void matAdd(int iN, int jN, float *a, float *b, float *c){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if(i < iN && j < jN){
        c[i * iN + j] = a[i * iN + j] + b[i * iN + j];
    }
}

int main(void){

}