#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <random>
#include <vector>
#include "simd/operations/arithmetic.hpp"

template<class T> void run(const char* type,std::size_t n) {
    std::vector<T> a(n),b(n),out(n);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> values(-100,100);
    for(std::size_t i=0;i<n;++i) { a[i]=static_cast<T>(values(rng)); b[i]=static_cast<T>(values(rng)); }
    constexpr int repetitions=100;
    auto measure=[&](auto kernel) {
        std::vector<double> samples;
        for(int sample=0;sample<7;++sample) {
            for(int r=0;r<5;++r) kernel(); // warm-up and OpenMP runtime initialization
            const auto start=std::chrono::steady_clock::now();
            for(int r=0;r<repetitions;++r) {
                kernel();
                asm volatile("" : : "g"(out.data()) : "memory");
            }
            samples.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/repetitions);
        }
        std::sort(samples.begin(),samples.end());
        return samples[samples.size()/2];
    };
    const double scalar=measure([&]{simd::ops::binary_transform_scalar(a.data(),b.data(),out.data(),n,simd::ops::Add{});});
    const double vector=measure([&]{simd::ops::binary_transform_simd(a.data(),b.data(),out.data(),n,simd::ops::Add{});});
    volatile T checksum=out[n/2];
    (void)checksum;
    std::cout<<type<<','<<n<<','<<std::fixed<<std::setprecision(4)<<scalar<<','<<vector<<','<<scalar/vector<<'\n';
}
int main() {
    std::cout<<"type,elements,scalar_ms,simd_ms,speedup\n";
    for(std::size_t n:{std::size_t(256),std::size_t(4096),std::size_t(65536),std::size_t(1048576)}) {
        run<std::int32_t>("int32",n); run<float>("float32",n); run<double>("float64",n);
    }
}
