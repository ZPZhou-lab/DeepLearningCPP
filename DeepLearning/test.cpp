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


int main(){
    clock_t startTime, endTime;
    numcpp nc;

    vector<int> shape;
    vector<int> axes;
    vector<int> axis;
    vector<int> strides;
    
    // vector<double> data = {1,0.1,0.1,1};
    // shape = {2,2};
    // auto mat1 = ndarray<double>(data,shape);
    int n = 50, p = 4;
    auto x = nc.random.randn(n,p);
    auto beta = nc.random.randn(p,1);
    auto noise = nc.random.randn(n) * 0.1;
    auto y = x.dot(beta) + noise;
    
    startTime = clock();
    auto model = glm::Linear_Regression();
    model.fit(x, y);
    auto y_pred = model.predict(x);
    auto mse = model.score(y, y_pred);
    endTime = clock();
    printf("time used: %.4fs\n",(double)(endTime - startTime) / CLOCKS_PER_SEC);

    cout<<"mse: "<<mse<<endl;

    return 0;
}
