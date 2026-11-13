#pragma once

#include <cstddef>
#include <stdexcept>
#include <xsimd/xsimd.hpp>

namespace simd::matrix {

// Row-major matrices: A is rows x inner, B is inner x cols,
// C is rows x cols.
template<class T>
void matmul_scalar(
    const T* a, const T* b, T* c,
    std::size_t rows, std::size_t inner, std::size_t cols)
{
    if (!a || !b || !c) throw std::invalid_argument("null matrix pointer");

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            T acc{};
            for (std::size_t k = 0; k < inner; ++k)
                acc += a[i * inner + k] * b[k * cols + j];
            c[i * cols + j] = acc;
        }
    }
}

template<class T>
void matmul_simd(const T* a,const T* b,T* c,std::size_t rows,std::size_t inner,std::size_t cols) {
    if(rows && cols && (!c || (inner && (!a || !b)))) throw std::invalid_argument("null matrix pointer");
    using batch=xsimd::batch<T>; constexpr std::size_t width=batch::size;
    #pragma omp parallel for if(rows * cols * inner > 1000000)
    for(std::ptrdiff_t row=0;row<static_cast<std::ptrdiff_t>(rows);++row) {
        const std::size_t i=static_cast<std::size_t>(row);
        std::size_t j=0;
        for(;j+width<=cols;j+=width) {
            batch acc(T{});
            for(std::size_t k=0;k<inner;++k) acc+=batch(a[i*inner+k])*batch::load_unaligned(b+k*cols+j);
            acc.store_unaligned(c+i*cols+j);
        }
        for(;j<cols;++j) { T acc{}; for(std::size_t k=0;k<inner;++k) acc+=a[i*inner+k]*b[k*cols+j]; c[i*cols+j]=acc; }
    }
}

} // namespace simd::matrix
