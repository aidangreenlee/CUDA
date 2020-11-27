#include <iostream>
#include <chrono>
#include <math.h>

typedef struct{
    int rows;
    int cols;
    float* mat;
}matrix;
/*  iN = cols of 1st matrix, rows of 2nd matrix
    jN = cols of 2nd matrix
    kN = rows of 1st matrix
*/
void MatMult(matrix a, matrix b, matrix c){
    for(int k = 0; k < a.rows; k++){
        for(int j = 0; j < b.cols; j++){
            float sum = 0;
            for(int i = 0; i < a.cols; i++){
                sum += a.mat[k * a.cols + i] * b.mat[i * b.cols + j];
            }
            c.mat[k * b.cols + j] = sum;
        }
    }
}

int main(){
    matrix a;
    matrix b;
    matrix c;
    // a.cols = b.rows
    a.rows = 1000;
    a.cols = 10000;
    b.rows = 10000;
    b.cols = 1000;
    a.mat = new float[a.cols * a.rows];
    b.mat = new float[b.rows * b.cols];
    c.mat = new float[a.rows * b.cols];


    for(int i = 0; i < a.cols; i++){
        for(int k = 0; k < a.rows; k++){
            a.mat[k * a.cols + i] = 1;
        }
        for(int j = 0; j < b.cols; j++){
            b.mat[i * b.cols + j] = 1;
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    MatMult(a, b, c);
    auto end = std::chrono::high_resolution_clock::now();

    double time_taken =  std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(); 
    time_taken *= 1e-9;

    float maxError = 0;
    for(int k = 0; k < a.rows; k++){
        for(int j = 0; j < b.cols; j++){
            maxError = fmax(maxError, fabs(c.mat[k * b.cols + j] - a.cols));
        }
    }

    std::cout << "Error: " << maxError << std::endl;
    std::cout << "Time taken by program is : " << time_taken; 
    std::cout << " sec" << std::endl; 

    delete [] a.mat;
    delete [] b.mat;
    delete [] c.mat;
    return 0;
}