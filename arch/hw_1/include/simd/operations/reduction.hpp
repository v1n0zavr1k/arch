#pragma once

#include <cstddef>
#include <stdexcept>
#include <xsimd/xsimd.hpp>

namespace simd::ops {

template<class T>
T sum_scalar(const T* data, std::size_t size) {
    T result{};
    for (std::size_t i = 0; i < size; ++i) result += data[i];
    return result;
}

template<class T>
T sum_simd(const T* data, std::size_t size) {
    if(size && !data) throw std::invalid_argument("null array pointer");
    using batch=xsimd::batch<T>; constexpr std::size_t width=batch::size;
    batch acc(T{}); std::size_t i=0;
    for(;i+width<=size;i+=width) acc+=batch::load_unaligned(data+i);
    T result=xsimd::reduce_add(acc);
    for(;i<size;++i) result+=data[i];
    return result;
}

template<class T>
double mean_simd(const T* data, std::size_t size) {
    if(size==0) throw std::invalid_argument("mean of an empty range");
    return static_cast<double>(sum_simd(data,size))/static_cast<double>(size);
}

template<class T>
double mean_scalar(const T* data, std::size_t size) {
    if (size == 0) throw std::invalid_argument("mean of an empty range");
    // Accumulate in double to avoid integer division/truncation.
    double result = 0.0;
    for (std::size_t i = 0; i < size; ++i)
        result += static_cast<double>(data[i]);
    return result / static_cast<double>(size);
}

} // namespace simd::ops
