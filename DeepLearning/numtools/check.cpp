#include <bits/stdc++.h>
#include <cassert>
#include <csignal>
#include <cstdio>
#include <vector>
#include <typeinfo>
#include <stdarg.h>
using namespace std;


void __check_shape(const long long &array_size, const long long &shape_size){
    if(array_size != shape_size){
        printf("the number of elements in array(%lld) does not match that of shape(%lld)\n",array_size,shape_size);
        assert(array_size == shape_size);
    }
}

void __check_shape(const vector<int>& shape1, const vector<int>& shape2){
    if(shape1.size() != shape2.size()){
        printf("operands could not be broadcast together with shapes\n");
        assert(shape1.size() == shape2.size());
    }
    for(int i=0;i<shape1.size();++i){
        if(shape1[i] != shape2[i]){
            printf("operands could not be broadcast together with shapes\n");
            assert(shape1[i] == shape2[i]);
        }
    }
}

void __check_index(const int &idx, const int &bound, const int &axis){
    if(idx >= bound || idx < 0){
        printf("index %d is out of bounds for axis %d with size %d\n",idx,axis,bound);
        assert(idx >= 0 && idx < bound);
    }
}

void __check_index(const int &idx, const int &bound){
    if(idx >= bound || idx < 0){
        printf("index %d is out of bounds with size %d\n",idx,bound);
        assert(idx >= 0 && idx < bound);
    }   
}

void __check_axes(const vector<int>& axes, const int &ndim){
    try{
        string msg;
        if((int)axes.size() != ndim){
            msg = "axes don't match array";
            throw msg;
        }
        set<int> tmp;
        for(auto axis : axes){
            if(axis < -ndim || axis >= ndim){
                msg = "axis is out of bounds for array of dimension ";
                throw msg;
            }
            if(axis < 0){
                axis = ndim - axis;
            }
            if(tmp.find(axis) != tmp.end()){
                msg = "repeated axis in transpose";
                throw msg;
            }
            tmp.insert(axis);
        }
    }catch(const string msg){
        cout<<msg<<endl;
        assert(false);
    }
}

void __check_squeeze(const vector<int> &axis, const vector<int> &shape){
    for(auto i:axis){
        if(shape[i] > 1){
            printf("cannot select an axis to squeeze out which has size not equal to one\n");
            assert(false);
        }
    }
    
}

void __check_axis(const int &ndim, const int &axis){
    if(axis >= ndim){
        printf("axis %d is out of bounds for array of dimension %d\n",axis,ndim);
        assert(false);
    }
}

void __check_repeat(const int &repeats){
    if(repeats < 1){
        printf("non-positive dimensions are not allowed\n");
        assert(false);
    }
}

void __check_expand(const int ndim, const vector<int> &axis){
    int new_ndim = ndim + axis.size();
    for(auto a:axis){
        if(a >= new_ndim){
            printf("axis %d is out of bounds for array of dimension %d\n",a,new_ndim);
            assert(false);
        }
    }
}

void __check_dot(const long long size1, const long long size2){
    if(size1 != size2){
        printf("shapes (%lld,) and (%lld,) not aligned: %lld (dim 0) != %lld (dim 0)\n",size1,size2,size1,size2);
        assert(false);
    }
}

void __check_one_dimension(const vector<int> &shape){
    if(shape.size() == 1 || (shape.size() == 2 && (shape[0] == 1 || shape[1] == 1))){
        return;
    }else{
        printf("array must be 1-dimensinal!\n");
        assert(false);
    }
}

void __check_choice_sample(const long long size, const long long array_size){
    if(size > array_size){
        printf("Cannot take a larger sample than population when `replace=false`\n");
        assert(false);
    }
}

void __check_propobility(const vector<double> &p, const long long size){
    // check shape and size
    if(p.size() != size){
        printf("array and propobility vector must have the same size!\n");
        assert(false);
    }
    for(auto pr : p){
        if(pr < 0){
            printf("weight in propobility vector must greater or equal to 0!\n");
            assert(false);
        }
    }
}

void __check_2darray(const vector<int> &shape){
    if(shape.size() != 2){
        printf("matrix power only for 2-D array!");
        assert(false);
    }
}

void __check_rows_equal_cols(const vector<int> &shape){
    if(shape[0] != shape[1]){
        printf("The number of rows and columns of the array is not equal!\n");
        assert(false);
    }
}

bool __check_operator_shape(const vector<int> &shape1, const vector<int> &shape2){
    bool flag = true;
    if(shape1.size() == shape2.size()){
        for(int i=0;i<shape1.size();++i){
            if(shape1[i] != shape2[i]){
                flag = false;
                break;
            }
        }
        if(flag){
            return false;
        }
    }else if(shape1.size() < shape2.size()){
        return __check_operator_shape(shape2,shape1);
    }
     
    // check trailing dimension
    flag = true;
    int s1 = shape1.size(), s2 = shape2.size();
    for(int i=0;i<s2;++i){
        if(shape1[s1 - 1 - i] != shape2[s2 - 1 - i] && (shape1[s1 - 1 - i] != 1 && shape2[s2 - 1 - i] != 1)){
            flag = false;
            break;
        }
    }

    if(flag){
        return true;
    }else{
        printf("operands could not be broadcast together with shapes (");
        for(int i=0;i<shape1.size()-1;++i) cout<<shape1[i]<<",";
        cout<<shape1[shape1.size()-1]<<") (";
        for(int i=0;i<shape2.size()-1;++i) cout<<shape2[i]<<",";
        cout<<shape2[shape2.size()-1]<<")"<<endl;
        assert(false);
    }

    return flag;
}

void __check_rows_equal(const int row1, const int row2){
    if(row1 != row2){
        printf("solve: Input operand 1 has a mismatch in its core dimension 0\n");
        assert(false);
    }
}