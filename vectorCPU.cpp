#include <iostream>
#include <chrono>
#include <math.h>

void add(int n, float *x, float *y, float *z){
    for(int i = 0; i < n; i++){
        z[i] = x[i] + y[i];
    }
}

int main(void){
    int N = 1<<20;

    float *x = new float[N];
    float *y = new float[N];
    float *z = new float[N];

    for(int i = 0; i < N; i++){
        x[i] = 1;
        y[i] = 1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    add(N, x, y, z);
    auto end = std::chrono::high_resolution_clock::now();

    double time_taken =  std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(); 
    time_taken *= 1e-9;

    float maxError = 0;
    for(int i = 0; i < N; i++){
        maxError = fmax(maxError, fabs(z[i]) - 2);
    }
    std::cout << "Error: " << maxError << std::endl;
    std::cout << "Time taken by program is : " << time_taken; 
    std::cout << " sec" << std::endl; 

    delete [] x;
    delete [] y;
    delete [] z;

    return 0;
}