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
namespace nc = numcpp;

namespace glm {
// linear Regression
class LinearRegression{
private:
    ndarray<double> beta;
    double intercept;
    bool _fit_intercept;
    bool _copy_X;

protected:


public:
    LinearRegression(bool fit_intercept=false, bool copy_X=true);
    // fit the model
    void fit(ndarray<double> &X, ndarray<double> &y);
    // predict the result
    ndarray<double> predict(ndarray<double> &X);
    // compute the MSE to do evaluation
    double score(ndarray<double> &y_true, ndarray<double> &y_pred);
};


}


// Linear Regression
glm::LinearRegression::LinearRegression(bool fit_intercept, bool copy_X){
    this->_fit_intercept = fit_intercept;
    this->_copy_X = copy_X;
    this->intercept = 0;
}

// fit the model
void glm::LinearRegression::fit(ndarray<double> &X, ndarray<double> &y){
    __check_2darray(X.shape());
    // copy X
    auto X_ = X;
    if(this->_copy_X){
        X_ = X.copy();
    }
    
    // fit the intercept
    if(this->_fit_intercept){
        
    }else{
        auto X_trans = X_.T();
        auto X_X = X_trans.dot(X_);
        this->beta = nc::linalg::inv(X_X).dot(X_trans).dot(y);
    }

}

// predict
ndarray<double> glm::LinearRegression::predict(ndarray<double> &X){
    __check_2darray(X.shape());
    return X.dot(this->beta); + this->intercept;
}

// compute MSE
double glm::LinearRegression::score(ndarray<double> &y_true, ndarray<double> &y_pred){
    auto error = y_true - y_pred;
    auto square_error = numcpp::pow(error, 2);

    return square_error.mean();
}