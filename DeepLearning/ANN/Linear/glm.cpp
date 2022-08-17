// Generalized linear method
#include "../../numtools/ndarray.cpp"
#include "../../numtools/numcpp.cpp"
#include <bits/stdc++.h>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

namespace glm {
// linear Regression
class Linear_Regression{
private:
    ndarray<double> beta;
    double intercept;
    bool _fit_intercept;

protected:


public:
    Linear_Regression(bool fit_intercept=false);
    // fit the model
    void fit(ndarray<double> &X, ndarray<double> &y);
    // predict the result
    ndarray<double> predict(ndarray<double> &X);
    // compute the MSE to do evaluation
    double score(ndarray<double> &y_true, ndarray<double> &y_pred);
};


}



// Linear Regression
glm::Linear_Regression::Linear_Regression(bool fit_intercept){
    this->_fit_intercept = fit_intercept;
}

// fit the model
void glm::Linear_Regression::fit(ndarray<double> &X, ndarray<double> &y){
    auto X_ = X.T();
    auto X_X = X_.dot(X);
    this->beta = numcpp::linaigBase::inv(X_X).dot(X_).dot(y);
}

// predict
ndarray<double> glm::Linear_Regression::predict(ndarray<double> &X){
    return X.dot(this->beta);;
}

// compute MSE
double glm::Linear_Regression::score(ndarray<double> &y_true, ndarray<double> &y_pred){
    auto error = y_true - y_pred;
    auto square_error = numcpp::pow(error, 2);

    return square_error.mean();
}