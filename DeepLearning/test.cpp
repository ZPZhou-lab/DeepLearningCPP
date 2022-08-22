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
    
    int n = 50, p = 10;
    double intercept = 2;

    auto x = nc::random::randn(n,p);
    // norm
    auto x_mean = x.mean(vector<int>{0});
    x = x - x_mean;

    auto beta = nc::random::randn(p,1);
    auto noise = nc::random::randn(n) * 0.01;
    auto y = x.dot(beta) + noise + intercept;

    startTime = clock();
    auto model = glm::LinearRegression(true);
    model.fit(x, y);
    auto y_pred = model.predict(x);
    auto mse = model.score(y, y_pred);

    endTime = clock();
    printf("time used: %.4fs\n",(double)(endTime - startTime) / CLOCKS_PER_SEC);

    cout<<"mse: "<<mse<<endl;
    cout<<"intercept: "<<model.intercept()<<endl;

    return 0;
}
