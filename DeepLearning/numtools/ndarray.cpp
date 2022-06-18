#include <bits/stdc++.h>
#include <cassert>
#include <complex>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <unordered_set>
#include <vector>
#include <typeinfo>
#include <stdarg.h>
#include "check.cpp"
#pragma once
using namespace std;

template <typename T>
class ndarray{
private:
    // define _data type
    typedef T value_type; 
    value_type _dtype;

    // the numver of elements
    long long _size;
   // store all elements in a vector
   T *_data;

    // the numver of the dimensions
    int _ndim; 
    // a vector represent how many elements stored in each dimension
    vector<int> _shape; 
    // a vector represents how many steps would across for each dimension
    vector<int> _strides; 

    // a vector represents the axes
    vector<int> _axes;

    // cumulative prod of shape, used to locate element
    vector<long long> __shape_cumprod;
    // maintenance function of shape_cumprod
    void __update_shape_cumprod(void);

    // internal(private) access method, check is omitted
    T __item(long long args);
    T __item(vector<int>& args);
    vector<int> __item_loc(long long args, vector<int> axis=vector<int>());

public:
    // default constructer
    ndarray();
    // constructer
    ndarray(T *array, vector<int>& shape);
    ndarray(T *array, vector<int>& shape, vector<int>& strides, vector<int>& axes);
    // destructor
    ~ndarray();
    
    // access element
    T item(long long args); // access element by flat index
    T item(vector<int>& args); // access element by an nd-index into the array
    
    template<typename ...Args> 
    T at(Args...args); // access element by an separate nd-index

    // return _data type
    string dtype(void);
    // return the number of elements
    long long size(void);
    // return the number of dimensions
    int ndim(void);
    // return the shape of the nd-array
    vector<int> shape(void);
    // return the strides of the nd-array
    vector<int> strides(void);
    // fetch _data
    T* data(void);

    // array transform
    ndarray transpose(vector<int>& axes);
    ndarray reshape(vector<int>& shape);
    ndarray flatten(void);
    ndarray squeeze(vector<int> axis=vector<int>());

    // array operation
    ndarray sum(vector<int> axis, bool keepdim=false);
    T sum(void);
    ndarray<double> mean(vector<int> axis, bool keepdim=false);
    double mean(void);
    
    // show matrix
    void show(void);

    // opetator reload
    // operation between ndarray and real number
    template<typename T1>
    ndarray operator - (const T1 b);
    template<typename T1>
    ndarray operator * (const T1 b);
    template<typename T1>
    ndarray operator / (const T1 b);
    template<typename T1>
    ndarray operator + (const T1 b);

    // operation between ndarray and ndarray
    template<typename T1>
    ndarray operator + (ndarray<T1> &b);

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
ndarray<T>::ndarray(T *array, vector<int>& shape){
    // compute size
    long long size = 1;
    for(auto s : shape) size *= s;

    // judge whether the number of array elements matches the dimension
    //__check_shape((long long)array.size(),size);

    this->_data = array;
    this->_shape = shape;
    this->_ndim = shape.size();
    this->_size = size;

    // compute strides
    this->_strides = vector<int>(this->_ndim,1);
    for(int i=_ndim-1;i>0;i--) this->_strides[i-1] = this->_strides[i] * this->_shape[i];

    // add axes
    this->_axes = vector<int>(this->_ndim,0);
    for(int i=0;i<_ndim;++i) this->_axes[i] = i;

    __update_shape_cumprod();

}

// constructer
template <typename T>
ndarray<T>::ndarray(T *array, vector<int>& shape, vector<int>& strides, vector<int>& axes){
    // compute size
    long long size = 1;
    for(auto s:shape) size *= s;

    // judge whether the number of array elements matches the dimension
    // __check_shape((long long)array.size(),size);

    this->_data = array;
    this->_shape = shape;
    this->_size = size;
    this->_ndim = shape.size();

    this->_strides = strides;
    this->_axes = axes;

    __update_shape_cumprod();
}

//  destructer
template <typename T>
ndarray<T>::~ndarray(){
    // delete [] _data;
    cout<<"Destructor called"<<endl;
}

// operator reload
template <typename T>
template <typename T1>
ndarray<T> ndarray<T>::operator/(const T1 b){
    double res[this->_size];
    for(long long i=0;i<this->_size;++i) res[i] = this->_data[i] / b;

    ndarray<double> trans(res,this->_shape);

    return trans;
}


template<typename T>
template<typename T1>
ndarray<T> ndarray<T>::operator+(const T1 b){
    T res[this->_size];
    for(long long i=0;i<this->_size;++i) res[i] = this->_data[i] + b;

    ndarray<T> trans(res,this->_shape);

    return trans;
}

template <typename T>
template <typename T1>
ndarray<T> ndarray<T>::operator*(const T1 b){
    double res[this->_size];
    for(long long i=0;i<this->_size;++i) res[i] = this->_data[i] * b;

    ndarray<double> trans(res,this->_shape);

    return trans;
}

template <typename T>
template <typename T1>
ndarray<T> ndarray<T>::operator-(const T1 b){
    T res[this->_size];
    for(long long i=0;i<this->_size;++i) res[i] = this->_data[i] - b;

    ndarray<T> trans(res,this->_shape);

    return trans;
}

template <typename T>
template <typename T1>
ndarray<T> ndarray<T>::operator+(ndarray<T1> &b){
    ndarray<T> flat1 = this->flatten();
    ndarray<T1> flat2 = b.flatten();
    T array[this->_size];
    for(long long i=0;i<this->_size;++i) array[i] = flat1.data()[i] + flat2.data()[i];
    // for(int i=0;i<this->_size;++i) array[i] += flat2.data()[i];

    ndarray<T> trans(array,this->_shape);

    return trans;
}

// maintenance function of shape_cumprod
template <typename T>
void ndarray<T>::__update_shape_cumprod(void){
    this->__shape_cumprod = vector<long long>(_ndim,1);
    for(int i=_ndim-1;i>0;i--) this->__shape_cumprod[i-1] = this->__shape_cumprod[i] * this->_shape[i];
}

// get flat index
long long __flat_idx(vector<int>& loc, vector<int>& strides){
    // initial flat index
    long long flat = 0;

    // compute flat index
    for(int i=0;i<loc.size();++i){
        flat += strides[i] * loc[i];
    }

    return flat;
}

// get item location vector
template <typename T>
vector<int> ndarray<T>::__item_loc(long long args, vector<int> axis){
    // initial lication
    vector<int> loc(_ndim,0);
    
    // compute location
    for(int i=0;i<_ndim;++i){
        loc[i] = (args / this->__shape_cumprod[i]);
        args %= this->__shape_cumprod[i];
    }
    
    for(auto j:axis) loc[j] = 0;

    return loc;
}

// access element by flat index
template <typename T>
T ndarray<T>::item(long long args){
    // check index
    __check_index(args,this->_size);

    // initial flat index
    long long flat = 0;    

    // compute flat index
    for(int i=0;i<_ndim;++i){
        flat += this->_strides[i] * (args / this->__shape_cumprod[i]);
        args %= this->__shape_cumprod[i];
    }
    return this->_data[flat];
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

    return this->_data[flat];
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


// return _data type
template <typename T>
string ndarray<T>::dtype(void){
    map<string,string> dtypes = {
        {"i","int"}, {"f","float"}, {"d","double"}, {"l","long"}, {"b","bool"},
        {"e", "long double"}, {"x","long long"}
    };
    const type_info &_dataInfo = typeid(this->_dtype);
    return dtypes[_dataInfo.name()];
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

// fetch data
template <typename T>
T* ndarray<T>::data(void){
    return this->_data;
}

// print the array
template <typename T>
void ndarray<T>::show(void){
    if(this->_ndim == 1){
        for(long long i=0;i<_size;++i) printf("%12.4f", _data[i]);
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
    ndarray<T> trans(this->_data,__shape,__strides,__axes);
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
        T array[this->_size];

        // element table switching
        for(long long i=0;i<this->_size;++i){
            array[i] = item(i);
        }

        trans = ndarray<T>(array,shape,__strides,__axes);
    }
    // do not need to adjust the position of elements
    else{
        trans = ndarray<T>(this->_data,shape,__strides,__axes);
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
ndarray<T> ndarray<T>::squeeze(vector<int> axis){
    // initialization
    vector<int> __axes, __shape, __strides, __axes_idx;
    int __ndim = 0;

    // squeeze all dimensions whose shape equall to 1
    if(axis.size() == 0){
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
    }
    // squeeze the specified axis
    else{
        // check squeeze
        __check_squeeze(axis,this->_shape);

        // squeeze the dimension specified
        for(int i=0;i<_ndim;++i){
            // determine whether squeeze the axis i
            bool flag = true;
            for(auto j:axis){
                if(i == j){
                    flag = false;
                    break;
                }
            }
            
            if(flag){
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
    
    ndarray<T> trans(this->_data,__shape,__strides,__axes_idx);

    return trans;
}

// sum
template <typename T>
T ndarray<T>::sum(void){
    T s = 0;
    for(auto e:this->_data) s += e;

    return s;
}

template <typename T>
ndarray<T> ndarray<T>::sum(vector<int> axis, bool keepdim){
    // initialization
    ndarray<T> trans;

    // figure sum for all axes by default
    if(axis.size() == 0){
        T array[1] = {this->sum()};
        vector<int> shape = {1};

        trans = ndarray<T>(array,shape);
    }
    // figure sum for specified axis
    else{
        // init new shape
        vector<int> __shape;
        long long __size = 1;

        for(int i=0;i<_ndim;++i){
            // flag variable juege whether squeeze the axis
            bool flag = true;
            for(auto j:axis){
                if(i == j){
                    flag = false;
                    break;
                }
            }

            if(flag){
                __shape.emplace_back(this->_shape[i]);
                __size *= this->_shape[i];
            }else{
                __shape.emplace_back(1);
            }
        }

        // init result
        T array[__size] = {0};
        trans = ndarray<T>(array,__shape);
        vector<int> __strides = trans.strides();

        // assign element
        for(long long i=0;i<_size;++i){
            vector<int> loc = __item_loc(i,axis);
            array[__flat_idx(loc,__strides)] += this->item(i);
        }
        cout<<endl;

        // construct result
        trans = ndarray<T>(array,__shape);
        // squeeze
        if(!keepdim) trans = trans.squeeze();
    }

    return trans;
}

// mean
template <typename T>
double ndarray<T>::mean(void){
    double m;

    T s = this->sum();
    m = s / this->_size;

    return m;
}


template <typename T>
ndarray<double> ndarray<T>::mean(vector<int> axis, bool keepdim){
    // figure sum
    ndarray<T> s = this->sum(axis,keepdim);

    // cimpute number of elements
    long long __num = 1;
    for(auto i:axis) __num *= this->_shape[i];    

    // compute mean
    ndarray<double> trans = s / __num;

    return trans;
}