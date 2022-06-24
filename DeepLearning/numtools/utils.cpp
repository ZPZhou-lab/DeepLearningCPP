#include <bits/stdc++.h>
#include <bits/types/clock_t.h>
#include <cassert>
#include <climits>
#include <complex>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <string>
#include <cmath>
#include <sys/cdefs.h>
#include <unordered_set>
#include <vector>
#include <ctime>
#include <typeinfo>
#include <stdarg.h>

using namespace std;

// time conuter
clock_t tic, toc;

// fetch reduction size
long long __reduce_size(vector<int> &shape, int ndim, vector<int> &axis){
    // init
    long long __size = 1;
        // axis set
    set<int> axis_set;
    for(auto j:axis) axis_set.insert(j);

    // compute new size
    for(int i=0;i<ndim;++i){
        if(axis_set.find(i) == axis_set.end()){
            __size *= shape[i];
        }
    }

    return __size;
}

// fetch reduction shape
vector<int> __reduce_shape(vector<int> &shape, int ndim, vector<int> &axis){
    // init
    vector<int> __shape;
        // axis set
    set<int> axis_set;
    for(auto j:axis) axis_set.insert(j);

    // compute new shape
    for(int i=0;i<ndim;++i){
        if(axis_set.find(i) == axis_set.end()){
            __shape.emplace_back(shape[i]);
        }
    }

    return __shape;
}

// fetch reduction step
long long __reduce_step(vector<int> &shape, int ndim, vector<int> &axis){
    // init
    long long step = 1;
        // axis set
    set<int> axis_set;
    for(auto j:axis) axis_set.insert(j);

    // compute new size
    for(int i=0;i<ndim;++i){
        if(axis_set.find(i) != axis_set.end()){
            step *= shape[i];
        }
    }

    return step;
}

// reduction help function for sum()
template <typename T>
void sum_reduction(T &a, T &b){
    a += b;
}

// reduction help function for max()
template <typename T>
void max_reduction(T &a, T &b){
    a = a < b ? b:a;
}

// reduction help function for min()
template <typename T>
void min_reduction(T &a, T &b){
    a = a > b ? b:a;
}

// reduction help function for argmax()
template <typename T>
bool argmax_reduction(T &a, T &b){
    return a > b ? true:false;
}

// reduction help function for argmin()
template <typename T>
bool argmin_reduction(T &a, T &b){
    return a < b ? true:false;
}