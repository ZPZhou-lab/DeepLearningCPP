#include "./numtools/ndarray.cpp"
#include "./numtools/numcpp.cpp"
#include "./ANN/Linear/glm.cpp"
#include <bits/stdc++.h>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
using namespace std;
namespace nc = numcpp;


int main(){
    clock_t startTime, endTime;
    
    // int n = 50, p = 10;
    // auto x = nc::random::randn(n,p);
    // auto beta = nc::random::randn(p,1);
    // auto noise = nc::random::randn(n) * 0.1;
    // auto y = x.dot(beta) + noise;
    
    startTime = clock();
    // auto model = glm::LinearRegression();
    // model.fit(x, y);
    // auto y_pred = model.predict(x);
    // auto mse = model.score(y, y_pred);

    auto A = nc::random::randn(2,1,4);
    auto B = nc::random::randn(2,2,4);
    auto C = nc::concat(A,B,1);


    endTime = clock();
    printf("time used: %.4fs\n",(double)(endTime - startTime) / CLOCKS_PER_SEC);

    // cout<<"mse: "<<mse<<endl;

    A.show();
    B.show();
    C.show();


    return 0;
}
