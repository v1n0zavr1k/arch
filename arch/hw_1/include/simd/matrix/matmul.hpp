#pragma once

#include <cstddef>
#include <stdexcept>

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

} // namespace simd::matrix
