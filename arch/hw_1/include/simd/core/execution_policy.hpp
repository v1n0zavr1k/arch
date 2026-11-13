#pragma once
#include <cstddef>
#include "../operations/arithmetic.hpp"

namespace simd {

enum class ExecutionPolicy {
    Scalar,
    SIMD
};

} // namespace simd

namespace simd::ops {
template<class T,class BinaryOp>
void binary_transform(const T* a,const T* b,T* out,std::size_t n,BinaryOp op,ExecutionPolicy policy) {
    if(policy==ExecutionPolicy::SIMD) binary_transform_simd(a,b,out,n,op);
    else binary_transform_scalar(a,b,out,n,op);
}
}
