#include <bits/stdc++.h>
#include <cassert>
#include <csignal>
#include <cstdio>
#include <vector>
#include <typeinfo>
#include <stdarg.h>
#include "check.cpp"
#pragma once
using namespace std;

template <typename T>
class ndarray{
private:
    // define data type
    typedef T value_type; 
    value_type _dtype;

    // store all elements in a vector
    vector<T> data; 
    // the numver of elements
    long long _size;

    // the numver of the dimensions
    int _ndim; 
    // a vector represent how many elements stored in each dimension
    vector<int> _shape; 
    // a vector represents how many steps would across for each dimension
    vector<int> _strides; 

    // a vector represents the axes
    vector<int> _axes;

public:
    // default constructer
    ndarray();
    // constructer
    ndarray(vector<T>& array, vector<int>& shape);
    ndarray(vector<T>& array, vector<int>& shape, vector<int>& strides, vector<int>& axes);
    ndarray(vector<T>& array, vector<long>& shape, vector<int>& inv_axes, 
            vector<long long>& raw_idx_prod, vector<long long>& new_idx_prod);
    // destructor
    ~ndarray();
    
    // access element
    T item(long long args); // access element by flat index
    T item(vector<int>& args); // access element by an nd-index into the array
    
    template<typename ...Args> 
    T at(Args...args); // access element by an separate nd-index

    // return data type
    string dtype(void);
    // return the number of elements
    long long size(void);
    // return the number of dimensions
    int ndim(void);
    // return the shape of the nd-array
    vector<int> shape(void);
    // return the strides of the nd-array
    vector<int> strides(void);

    // array transform
    ndarray transpose(vector<int>& axes);
    ndarray reshape(vector<int>& shape);
    ndarray flatten(void);
    ndarray squeeze(void);
    
    // 打印矩阵
    void show(void);
};

// get parameters from an indefinite length parameter list
template <typename T>
vector<T> fetchArgs(vector<T>& fetch, T arg){
    fetch.emplace_back(arg);
    return fetch;
}
template <typename T, typename ...Args>
vector<T> fetchArgs(vector<T>& fetch, T arg, Args...args){
    fetch.emplace_back(arg);
    return fetchArgs(fetch,args...);
}

// permutation of axes
vector<int> permute(vector<int>& vec, vector<int>& axes){
    // temporary variable
    vector<int> permuted(vec.size());
    for(int i=0;i<vec.size();++i) permuted[i] = vec[axes[i]];

    return permuted;
}

// default constructer
template <typename T>
ndarray<T>::ndarray(){
    this->_ndim = 0;
    this->_size = 0;
}

// constructer
template <typename T>
ndarray<T>::ndarray(vector<T>& array, vector<int>& shape){
    // compute size
    long long size = 1;
    for(auto s : shape) size *= s;

    // judge whether the number of array elements matches the dimension
    __check_shape((long long)array.size(),size);

    this->data = array;
    this->_shape = shape;
    this->_ndim = shape.size();
    this->_size = size;

    // compute strides
    this->_strides = vector<int>(this->_ndim,1);
    for(int i=_ndim-1;i>0;i--) this->_strides[i-1] = this->_strides[i] * this->_shape[i];

    // add axes
    this->_axes = vector<int>(this->_ndim,0);
    for(int i=0;i<_ndim;++i) this->_axes[i] = i;

}

// constructer
template <typename T>
ndarray<T>::ndarray(vector<T>& array, vector<int>& shape, vector<int>& strides, vector<int>& axes){
    // compute size
    long long size = 1;
    for(auto s:shape) size *= s;

    // judge whether the number of array elements matches the dimension
    __check_shape((long long)array.size(),size);

    this->data = array;
    this->_shape = shape;
    this->_size = size;
    this->_ndim = shape.size();

    this->_strides = strides;
    this->_axes = axes;
}

//  destructer
template <typename T>
ndarray<T>::~ndarray(){
    cout<<"Destructor called"<<endl;
}

// access element by flat index
template <typename T>
T ndarray<T>::item(long long args){
    // initial flat index
    long long flat = 0;
    // compute flat index
    long long step = this->_size / this->_strides[_ndim - 1];
    flat += args / step;
    flat += (args%step * this->_strides[_ndim - 1]);
    
    return this->data[flat];
}

// access element by nd-array index
template <typename T>
T ndarray<T>::item(vector<int>& args){
    // initial flat index
    long long flat = 0;

    // compute flat index
    for(int i=0;i<_ndim;++i){
        // check index
        __check_index(args[i],this->_shape[i],i);

        flat += this->_strides[i] * args[i];
    }

    return this->data[flat];
}

// access element by seperate nd-array index
template <typename T>
template <typename ...Args>
T ndarray<T>::at(Args...args){
    // fetch nd-array index
    vector<int> idx;
    idx = fetchArgs(idx,args...);

    return item(idx);
}


// return data type
template <typename T>
string ndarray<T>::dtype(void){
    map<string,string> dtypes = {
        {"i","int"}, {"f","float"}, {"d","double"}, {"l","long"}, {"b","bool"},
        {"e", "long double"}, {"x","long long"}
    };
    const type_info &dataInfo = typeid(this->_dtype);
    return dtypes[dataInfo.name()];
}

// the number of elements
template <typename T>
long long ndarray<T>::size(void){
    return this->_size;
}

// the number of dimensions
template <typename T>
int ndarray<T>::ndim(void){
    return this->_ndim;
}

// the number of elements in each dimension
template <typename T>
vector<int> ndarray<T>::shape(void){
    return this->_shape;
}

// the strides of the nd-array
template <typename T>
vector<int> ndarray<T>::strides(void){
    return this->_strides;
}

// print the array
template <typename T>
void ndarray<T>::show(void){
    if(this->_ndim == 1){
        for(auto d : data) printf("%12.4f", d);
        printf("\n");
    }else{
        // compute cummulative prod of shape
        vector<long long> cumprod_shape(_ndim,1);
        for(int j=1;j<_ndim;++j) cumprod_shape[j] = cumprod_shape[j-1] * this->_shape[_ndim-j];

        // print each element
        for(long long i=0;i<this->_size;++i){
            printf("%12.4f", item(i));

            // wrap if crossing a dimension
            for(int j=1;j<this->_ndim;++j){
                if((i+1)%cumprod_shape[j] == 0) printf("\n");
            }
        }
    }
}

// transpose
template <typename T>
ndarray<T> ndarray<T>::transpose(vector<int>& axes){
    // anomaly detection
    __check_axes(axes, this->_ndim);

    // update strides
    vector<int> __strides = permute(this->_strides, axes);
    // update shape
    vector<int> __shape = permute(this->_shape, axes);
    // update axes
    vector<int> __axes = permute(this->_shape, axes);
    
    // transformed array
    ndarray<T> trans(this->data,__shape,__strides,__axes);
    return trans;
}

// reshape
template <typename T>
ndarray<T> ndarray<T>::reshape(vector<int> &shape){
    // anomaly detection
    long long check_size = 1;
    for(auto s:shape) check_size *= s;

    // judge whether the number of array elements matches the dimension
    __check_shape(this->_size,check_size);

    // initialization
    ndarray<T> trans;

    // check whether the position of elements in the array needs to be adjusted
    bool flag = false;
    for(int i=0;i<_ndim;i++){
        if(this->_axes[i] != i){
            flag = true;
            break;
        }
    }

    // update ndim
    int __ndim = shape.size();

    // update strides
    vector<int> __strides = vector<int>(__ndim,1);
    for(int i=__ndim-1;i>0;i--) __strides[i-1] = __strides[i] * shape[i];

    // update axes
    vector<int> __axes = vector<int>(__ndim,0);
    for(int i=0;i<__ndim;++i) __axes[i] = i;

    // use permutation to adjust the position of elements then reshape
    if(flag){
        // adjust the position of elements
        // copy construction
        vector<T> array(this->data);

        // element table switching
        for(long long i=0;i<this->_size;++i){
            array[i] = item(i);
        }

        trans = ndarray<T>(array,shape,__strides,__axes);
    }
    // do not need to adjust the position of elements
    else{
        trans = ndarray<T>(this->data,shape,__strides,__axes);
    }

    return trans;
}

// flatten
template <typename T>
ndarray<T> ndarray<T>::flatten(void){
    vector<int> shape = {(int)this->_size};
    return reshape(shape);
}

// squeeze
template <typename T>
ndarray<T> ndarray<T>::squeeze(void){
    vector<int> __axes, __shape, __strides, __axes_idx;
    int __ndim = 0;
    for(int i=0;i<_ndim;++i){
        // squeeze the dimension when shape equals to 1
        if(this->_shape[i] != 1){
            // update shape and strides
            __shape.emplace_back(this->_shape[i]);
            __strides.emplace_back(this->_strides[i]);

            // update axes
            __axes.emplace_back(this->_axes[i]);
            // record axes index
            __axes_idx.emplace_back(__ndim);
            __ndim++;
        }
    }

    // sort the axes and adjust axes index
    // insert-sort
    for(int j=1;j<__ndim;j++){
        for(int i=j;i>0;i--){
            if(__axes[i] < __axes[i-1]){
                // swap
                swap(__axes[i],__axes[i-1]);
                swap(__axes_idx[i],__axes_idx[i-1]);
            }else{
                break;
            }
        }
    }
    
    ndarray<T> trans(this->data,__shape,__strides,__axes_idx);

    return trans;
}