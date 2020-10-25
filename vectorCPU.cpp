#include <iostream>
#include <math.h>

void add(int n, float *x, float *y, float *z){
    for(int i = 0; i < n; i++){
        z[i] = x[i] + y[i];
    }
}

void dot(int n, float *x, float *y, float *c){
    for(int i = 0; i < n; i++){
        *c += x[i] * y[i];
    }
}

int main(void){
    int N = 1<<20;

    float *x = new float[N];
    float *y = new float[N];
    float *z = new float[N];
    float *c = new float;

    for(int i = 0; i < N; i++){
        x[i] = 1;
        y[i] = 1;
    }

    add(N, x, y, z);
    dot(N, x, y, c);

    float maxError = 0;
    for(int i = 0; i < N; i++){
        maxError = fmax(maxError, fabs(z[i]) - 3);
    }
    std::cout << "Error: " << maxError << std::endl;
    std::cout << "dot product: " << *c << std::endl;
    delete [] x;
    delete [] y;

    return 0;
}