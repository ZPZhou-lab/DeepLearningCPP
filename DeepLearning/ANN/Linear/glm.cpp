// Generalized linear method
#include "../../numtools/ndarray.cpp"
#include "../../numtools/numcpp.cpp"
#include <bits/stdc++.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
using namespace numcpp;
namespace nc = numcpp;

namespace glm {
// linear Regression
class LinearRegression{
private:
    ndarray<double> _beta;
    double _intercept;
    bool _fit_intercept;
    bool _copy_X;
    bool _fitted;
    ndarray<double> __solve_beta(ndarray<double> &X, ndarray<double> &y);
    // the number of feature dimension
    int _p;

protected:


public:
    LinearRegression(bool fit_intercept=false, bool copy_X=true);
    // fit the model
    void fit(ndarray<double> &X, ndarray<double> &y);
    // predict the result
    ndarray<double> predict(ndarray<double> &X);
    // compute the MSE to do evaluation
    double score(ndarray<double> &y_true, ndarray<double> &y_pred);
    // get coef
    ndarray<double> coef(void);
    double intercept(void);
};

}


// Linear Regression
glm::LinearRegression::LinearRegression(bool fit_intercept, bool copy_X){
    this->_fit_intercept = fit_intercept;
    this->_copy_X = copy_X;
    this->_intercept = 0;
    this->_fitted = false;
}

// fit the model
void glm::LinearRegression::fit(ndarray<double> &X, ndarray<double> &y){
    __check_2darray(X.shape());

    // featurn dimension
    this->_p = X.shape()[1];

    // copy X
    auto X_ = X;
    if(this->_copy_X){
        X_ = X.copy();
    }
    
    // fit the intercept
    if(this->_fit_intercept){
        // assign intercept
        this->_intercept = y.mean();
        // fit beta
        auto y_ = y - y.mean();
        this->_beta = this->__solve_beta(X, y_);
    }else{
        this->_beta = this->__solve_beta(X_, y);
    }

    this->_fitted = true;
}

// predict
ndarray<double> glm::LinearRegression::predict(ndarray<double> &X){
    if(!this->_fitted){
        printf("model has not been fitted!\n");
        assert(false);
    }
    __check_2darray(X.shape());
    return X.dot(this->_beta) + this->_intercept;
}

// compute MSE
double glm::LinearRegression::score(ndarray<double> &y_true, ndarray<double> &y_pred){
    auto error = y_true - y_pred;
    auto square_error = numcpp::pow(error, 2);

    return square_error.mean();
}

// solve for beta
ndarray<double> glm::LinearRegression::__solve_beta(ndarray<double> &X, ndarray<double> &y){
    auto X_trans = X.T();
    auto X_X = X_trans.dot(X);
    auto beta = nc::linalg::inv(X_X).dot(X_trans).dot(y);
    return beta;
}

// get coef
ndarray<double> glm::LinearRegression::coef(){
    return this->_beta;
}
double glm::LinearRegression::intercept(){
    return this->_intercept;
}