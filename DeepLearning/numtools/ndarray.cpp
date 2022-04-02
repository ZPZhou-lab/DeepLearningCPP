#include<bits/stdc++.h>
#include<vector>
#include <typeinfo>
#include<stdarg.h>
#pragma once
using namespace std;

template <typename T>
class ndarray{
private:
    typedef T value_type; // 定义类别属性
    vector<long> _shape; // 矩阵维度
    vector<int> _axes; // 轴
    vector<int> _inv_axes; // 轴的逆
    vector<T> data; // 数据
    value_type _dtype; // 数据类型
    int _ndim; // 维度数量
    long long _size; // 元素个数
    vector<long long> _raw_idx_prod; // 存在数组维度的原始累计乘积
    vector<long long> _new_idx_prod; // 存在数组维度的新的累计乘积

public:
    // 默认构造函数
    ndarray();
    // 构造函数
    ndarray(vector<T>& arr, vector<long>& shape);
    ndarray(vector<T>& arr, vector<long>& shape, vector<int>& inv_axes, 
            vector<long long>& raw_idx_prod, vector<long long>& new_idx_prod);
    // 析构函数
    ~ndarray();
    
    // 访问元素
    template <typename ...Args>
    T at(Args...args);

    // 用序号访问元素
    T iloc(long long idx);

    // 返回数据类型
    string dtype(void);
    // 返回元素个数
    long long size(void);
    // 返回矩阵维度
    int ndim(void);
    // 矩阵形状
    vector<long> shape(void);

    // 维度变换
    ndarray transpose(vector<int>& axes);
    ndarray reshape(vector<long>& shape);
    ndarray flatten(void);
    
    // 打印矩阵
    void show(void);
};

// 从不定长参数列表中获取参数
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

// 坐标置换辅助函数
vector<long> permutation(vector<long>& at, vector<int>& axes){
    vector<long> permuted(at.size());
    for(int i=0;i<(int)at.size();++i) permuted[i] = at[axes[i]];
    return permuted;
}

// 生成所有可能的排列
void permutation_DFS(vector<vector<long>>& orders, vector<long>& path, vector<long>& shape, int axis){
    if(axis == (int)shape.size()){
        orders.emplace_back(path);
        return;
    }
    for(int i=0;i<shape[axis];++i){
        // 添加元素
        path.emplace_back(i);
        // 递归搜索
        permutation_DFS(orders, path, shape, axis+1);
        // 恢复状态
        path.pop_back();
    }
}
vector<vector<long>> permutation_generator(vector<long>& shape){
    vector<vector<long>> orders;
    vector<long> path;
    permutation_DFS(orders,path,shape,0);
    return orders;
}

// 将索引映射到正确的下标
template <typename ...Args>
long long to_iloc(vector<long long> idx_prod, int ndim, long long size,Args...args){
    // 取出索引
    vector<int> idx;
    idx = fetchArgs(idx,args...);
    // 判断维度是否正确
    assert((int)idx.size() == ndim);
    long long loc = 0;
    for(int i=0;i<ndim;++i) loc += idx_prod[ndim-i-1]*idx[i];
    assert(loc >= 0 && loc < size);
    return loc;
}
// 另一种模板
long long to_iloc(vector<long long> idx_prod, int ndim, long long size, vector<long> at){
    long long loc = 0;
    for(int i=0;i<ndim;++i) loc += idx_prod[ndim-i-1]*at[i];
    assert(loc >= 0 && loc < size);
    return loc;
}

// 将下标映射到正确的索引
vector<long> to_at(long long iloc, vector<long long> idx_prod, int ndim){
    vector<long> at;
    for(int i=0;i<ndim;++i){
        long s = iloc / idx_prod[ndim-1-i];
        at.emplace_back(s);
        iloc %= idx_prod[ndim-1-i];
    }
    return at;
}

// 根据axes，将当前索引映射到正确的数据下标
long long iloc_map(long long iloc, vector<long long>& raw_idx_prod, vector<long long>& new_idx_prod, 
                   long long size, int ndim, vector<int>& axes){
    long long loc = 0;
    for(int i=0;i<ndim;++i){
        long s = iloc / new_idx_prod[ndim-1-i];
        iloc %= new_idx_prod[ndim-1-i];
        // 与其对应的下标进行交换
        loc += raw_idx_prod[axes[ndim-1-i]] * s;
    }
    assert(loc >= 0 && loc < size);
    return loc;
}

// 默认构造函数
template <typename T>
ndarray<T>::ndarray(){
    this->_ndim = 0;
    this->_size = 0;
}

// 构造函数
template <typename T>
ndarray<T>::ndarray(vector<T>& arr, vector<long>& shape){
    long long size = 1;
    for(auto s : shape) size *= s;
    assert((long long)arr.size() == size);
    this->data = arr;
    this->_shape = shape;
    this->_size = size;
    this->_ndim = shape.size();
    for(int i=0;i<this->_ndim;++i) this->_axes.emplace_back(i);
    this->_inv_axes = this->_axes;
    // 统计累计索引
    vector<long long> idx_prod(this->_ndim,1);
    for(int i=this->_ndim-1;i>0;--i) idx_prod[this->_ndim-i] = idx_prod[this->_ndim-i-1] * this->_shape[i];
    this->_raw_idx_prod = idx_prod;
    this->_new_idx_prod = idx_prod;
}
// 构造函数
template <typename T>
ndarray<T>::ndarray(vector<T>& arr, vector<long>& shape, vector<int>& inv_axes, 
                    vector<long long>& raw_idx_prod, vector<long long>& new_idx_prod){
    long long size = 1;
    for(auto s : shape) size *= s;
    assert((long long)arr.size() == size);
    this->data = arr;
    this->_shape = shape;
    this->_size = size;
    this->_ndim = shape.size();
    for(int i=0;i<this->_ndim;++i) this->_axes.emplace_back(i);
    this->_inv_axes = inv_axes;
    // 统计累计索引
    this->_raw_idx_prod = raw_idx_prod;
    this->_new_idx_prod = new_idx_prod;
}

//  析构函数
template <typename T>
ndarray<T>::~ndarray(){
    cout<<"Destructor called"<<endl;
}

// 获取元素的值
template <typename T>
template <typename ...Args>
T ndarray<T>::at(Args...args){
    long long iloc = to_iloc(this->_new_idx_prod,this->_ndim,this->_size,args...);
    iloc = iloc_map(iloc,this->_raw_idx_prod,this->_new_idx_prod,this->_size,this->_ndim,this->_inv_axes);
    return this->data[iloc];
}

// 用序号访问元素
template <typename T>
T ndarray<T>::iloc(long long idx){
    // 取出索引
    idx = iloc_map(idx,this->_raw_idx_prod,this->_new_idx_prod,this->_size,this->_ndim,this->_inv_axes);
    return this->data[idx];
}

// 数据类型
template <typename T>
string ndarray<T>::dtype(void){
    map<string,string> dtypes = {
        {"i","int"}, {"f","float"}, {"d","double"}, {"l","long"}, {"b","bool"},
        {"e", "long double"}, {"x","long long"}
    };
    const type_info &dataInfo = typeid(this->_dtype);
    return dtypes[dataInfo.name()];
}

// 元素个数
template <typename T>
long long ndarray<T>::size(void){
    return this->_size;
}

// 矩阵维度
template <typename T>
int ndarray<T>::ndim(void){
    return this->_ndim;
}

// 矩阵维度
template <typename T>
vector<long> ndarray<T>::shape(void){
    return this->_shape;
}

template <typename T>
void ndarray<T>::show(void){
    if(this->_ndim == 1){
        for(auto d : data) printf("%+12.4f", d);
        printf("\n");
    }else{
        for(long long i=0;i<this->_size;++i){
            printf("%+12.4f", iloc(i));
            for(int j=1;j<this->_ndim;++j){
                if((i+1)%this->_new_idx_prod[j] == 0) printf("\n");
            }
        }
    }
}

// 维度变换
template <typename T>
ndarray<T> ndarray<T>::transpose(vector<int>& axes){
    // 异常检测
    try{
        if((int)axes.size() != this->_ndim){
            throw "axes don't match array";
        }
        set<int> tmp;
        for(auto axis : axes){
            if(axis < -this->_ndim || axis >= this->_ndim){
                throw "axis is out of bounds for array of dimension ";
            }
            if(axis < 0){
                axis = this->_ndim - axis;
            }
            if(tmp.find(axis) != tmp.end()){
                throw "repeated axis in transpose";
            }
            tmp.insert(axis);
        }
    }catch(const char* msg){
        cout<<msg<<endl;
        assert(false);
    }
    // 新的形状
    vector<long> new_shape = permutation(this->_shape, axes);
    // 计算新的idx_prod
    // 统计累计索引
    vector<long long> idx_prod(this->_ndim,1);
    for(int i=this->_ndim-1;i>0;--i) idx_prod[this->_ndim-i] = idx_prod[this->_ndim-i-1] * new_shape[i];
    // 创建axes的逆变换
    vector<int> inv_axes(this->_ndim,0);
    for(int i=0;i<this->_ndim;++i) inv_axes[axes[i]] = i;
    ndarray<T> trans(this->data,new_shape,inv_axes,this->_raw_idx_prod,idx_prod);
    return trans;
}

// // 形状变换
// template <typename T>
// ndarray<T> ndarray<T>::reshape(vector<long>& shape){
//     // 异常检测
//     long long check_size = 1;
//     for(auto s : shape) check_size *= s;
//     try{
//         if(this->_size != check_size){
//             throw "cannot reshape array of size into new shape";
//         }
//     }catch(const char* msg){
//         cout<<msg<<endl;
//         assert(false);
//     }
//     // 新的形状

//     return;
// }
