#include <iostream>
#include <chrono>
#include <math.h>

void MatAdd(int iN, int jN, float *a, float *b, float *c){
    for(int i = 0; i < iN; i++){
        for(int j = 0; j < jN; j++){
            c[i*iN + j] = a[i*iN + j] + b[i*iN + j];
        }
    }
}

int main(){
    int iN = 30000;
    int jN = 30000;
    float *x = new float[iN * jN];
    float *y = new float[iN * jN];
    float *z = new float[iN * jN];

    for(int i = 0; i < iN; i++){
        for(int j = 0; j < jN; j++){
            x[i*iN + j] = 1;
            y[i*iN + j] = 1;
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    MatAdd(iN, jN, x, y, z);
    auto end = std::chrono::high_resolution_clock::now();

    double time_taken =  std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(); 
    time_taken *= 1e-9;

    float maxError = 0;
    for(int i = 0; i < iN; i++){
        for(int j = 0; j < jN; j++){
            maxError = fmax(maxError, fabs(z[i*iN + j] - 2));
        }
    }


    std::cout << "Error: " << maxError << std::endl;
    std::cout << "Time taken by program is : " << time_taken; 
    std::cout << " sec" << std::endl; 

    delete [] x;
    delete [] y;
    delete [] z;
    return 0;
}