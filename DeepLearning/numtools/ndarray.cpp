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

    // vector<long> _shape; // 矩阵维度
    // vector<int> _axes; // 轴
    // vector<int> _inv_axes; // 轴的逆
    // vector<T> data; // 数据

    // int _ndim; // 维度数量
    // long long _size; // 元素个数
    // vector<long long> _raw_idx_prod; // 存在数组维度的原始累计乘积
    // vector<long long> _new_idx_prod; // 存在数组维度的新的累计乘积

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

// // 将索引映射到正确的下标
// template <typename ...Args>
// long long to_iloc(vector<long long> idx_prod, int ndim, long long size,Args...args){
//     // 取出索引
//     vector<int> idx;
//     idx = fetchArgs(idx,args...);
//     // 判断维度是否正确
//     assert((int)idx.size() == ndim);
//     long long loc = 0;
//     for(int i=0;i<ndim;++i) loc += idx_prod[ndim-i-1]*idx[i];
//     assert(loc >= 0 && loc < size);
//     return loc;
// }
// // 另一种模板
// long long to_iloc(vector<long long> idx_prod, int ndim, long long size, vector<long> at){
//     long long loc = 0;
//     for(int i=0;i<ndim;++i) loc += idx_prod[ndim-i-1]*at[i];
//     assert(loc >= 0 && loc < size);
//     return loc;
// }

// // 将下标映射到正确的索引
// vector<long> to_at(long long iloc, vector<long long> idx_prod, int ndim){
//     vector<long> at;
//     for(int i=0;i<ndim;++i){
//         long s = iloc / idx_prod[ndim-1-i];
//         at.emplace_back(s);
//         iloc %= idx_prod[ndim-1-i];
//     }
//     return at;
// }

// // 根据axes，将当前索引映射到正确的数据下标
// long long iloc_map(long long iloc, vector<long long>& raw_idx_prod, vector<long long>& new_idx_prod, 
//                    long long size, int ndim, vector<int>& axes){
//     long long loc = 0;
//     for(int i=0;i<ndim;++i){
//         long s = iloc / new_idx_prod[ndim-1-i];
//         iloc %= new_idx_prod[ndim-1-i];
//         // 与其对应的下标进行交换
//         loc += raw_idx_prod[axes[ndim-1-i]] * s;
//     }
//     assert(loc >= 0 && loc < size);
//     return loc;
// }

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

// // 构造函数
// template <typename T>
// ndarray<T>::ndarray(vector<T>& arr, vector<long>& shape, vector<int>& inv_axes, 
//                     vector<long long>& raw_idx_prod, vector<long long>& new_idx_prod){
//     long long size = 1;
//     for(auto s : shape) size *= s;
//     assert((long long)arr.size() == size);
//     this->data = arr;
//     this->_shape = shape;
//     this->_size = size;
//     this->_ndim = shape.size();
//     for(int i=0;i<this->_ndim;++i) this->_axes.emplace_back(i);
//     this->_inv_axes = inv_axes;
//     // 统计累计索引
//     this->_raw_idx_prod = raw_idx_prod;
//     this->_new_idx_prod = new_idx_prod;
// }

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

    // // 新的形状
    // vector<long> new_shape = permutation(this->_shape, axes);
    // // 计算新的idx_prod
    // // 统计累计索引
    // vector<long long> idx_prod(this->_ndim,1);
    // for(int i=this->_ndim-1;i>0;--i) idx_prod[this->_ndim-i] = idx_prod[this->_ndim-i-1] * new_shape[i];
    // // 创建axes的逆变换
    // vector<int> inv_axes(this->_ndim,0);
    // for(int i=0;i<this->_ndim;++i) inv_axes[axes[i]] = i;
    
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
    // use permutation to adjust the position of elements then reshape
    if(flag){
        return *this;
    }
    // do not need to adjust the position of elements
    else{
        // update ndim
        int __ndim = shape.size();

        // update strides
        vector<int> __strides = vector<int>(__ndim,1);
        for(int i=__ndim-1;i>0;i--) __strides[i-1] = __strides[i] * shape[i];

        // update axes
        vector<int> __axes = vector<int>(__ndim,0);
        for(int i=0;i<__ndim;++i) __axes[i] = i;

        trans = ndarray<T>(this->data,shape,__strides,__axes);
    }

    return trans;
}

// 形状变换reshape和转置操作transpose
/*
    数据结构
    ======================================
    1. 使用一维数组data保存同一数据类型的元素
    2. 需要包含dtype
    3. 包含ndim和shape
    4. 包含strides，用于确认多维度数组元素的访问
    5. 用一个scalar确认索引位置
    6. axes存储矩阵轴的情况

    reshape
    ======================================
    1. 不改变data
    2. 操作ndim, shape和strides
    3. 瞬时完成，应与数组元素个数无关

    transpose
    ======================================
    1. 不改变data
    2. 操作ndim, shape和strides
    3. 操作axes
    4. 瞬时完成，应与数组元素个数无关

    reshape 和 transpose
    ======================================
    当对进行过transpose操作的数组reshape时，
    需要重新更改data里的元素位置，并重设数组
    axes，注意，此时该操作与数组元素个数有关
*/