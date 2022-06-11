#include<bits/stdc++.h>
#include<vector>
#include <typeinfo>
#include<stdarg.h>
#pragma once
using namespace std;

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