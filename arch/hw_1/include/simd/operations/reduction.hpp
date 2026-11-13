#pragma once

#include <cstddef>
#include <stdexcept>

namespace simd::ops {

template<class T>
T sum_scalar(const T* data, std::size_t size) {
    T result{};
    for (std::size_t i = 0; i < size; ++i) result += data[i];
    return result;
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
