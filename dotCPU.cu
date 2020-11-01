#include <iostream>
#define N 2048*2048
#include <chrono>
void dot(int *a, int *b, int *c){
    int sum = 0;
    for(int i = 0; i < N; i++){
        sum += a[i] * b[i];
    }
    *c = sum;
}

int main(){
    int *a = new int[N];
    int *b = new int[N];
    int *c = new int;

    *c = 0;

    for(int i = 0; i < N; i++){
        a[i] = 1;
        b[i] = 1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    dot(a, b, c);
    auto end = std::chrono::high_resolution_clock::now();

    double time_taken =  std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(); 
    time_taken *= 1e-9;
    
    std::cout << "dot product: " << *c << std::endl;
    std::cout << "Time taken by program is : " << time_taken; 
    std::cout << " sec" << std::endl; 

    delete(a);
    delete(b);
    delete(c);

    return 0;
}