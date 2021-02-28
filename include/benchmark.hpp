#if !defined(AMT_BENCHMARK_BENCHMARK_HPP)
#define AMT_BENCHMARK_BENCHMARK_HPP

#include <timer.hpp>
#include <macros.hpp>
#include <limits>
#include <cmath>
#include <functional>

namespace amt{
#ifndef AMT_HAS_NO_INLINE_ASSEMBLY

    template<typename T>
    AMT_ALWAYS_INLINE constexpr void no_opt(T const& val) noexcept{
        asm volatile("" : : "r,m"(val) : "memory");
    }

    template<typename T>
    AMT_ALWAYS_INLINE constexpr void no_opt(T& val) noexcept{
        #if defined(__clang__)
            asm volatile("" : "+r,m"(val) : : "memory");
        #else
            asm volatile("" : "+m,r"(val) : : "memory");
        #endif
    }
#else
    #error "Compiler is not supported"
#endif

namespace detail{
    
    bool approx_equal(float a, float b, float epsilon_factor = 10000.f ){
        return fabs(a - b) <= ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * (std::numeric_limits<float>::epsilon() * epsilon_factor));
    }

} // namespace detail


template<std::size_t MaxIter = 10000000u, typename FnType, typename... FnArgs>
constexpr auto benchmark(FnType&& fn, FnArgs&&... args) noexcept{
    double time{};
    for(auto i = 1ul; i <= MaxIter; ++i){
        double prev_time{time};
        auto t = timer{};
        std::invoke(std::forward<FnType>(fn), std::forward<FnArgs>(args)...);
        time = t();
        if( detail::approx_equal(static_cast<float>(prev_time), static_cast<float>(time)) ) break;
    }
    return time;
}

template<std::size_t MaxIter = 10000000u, typename FnType, typename... FnArgs>
constexpr auto benchmark_timer_as_arg(FnType&& fn, FnArgs&&... args) noexcept{
    double time{};
    for(auto i = 1ul; i <= MaxIter; ++i){
        double prev_time{time};
        auto t = timer{};
        std::invoke(std::forward<FnType>(fn), std::forward<FnArgs>(args)..., t);
        time = t();
        if( detail::approx_equal(static_cast<float>(prev_time), static_cast<float>(time)) ) break;
    }
    return time;
}

} // namespace amt


#endif // AMT_BENCHMARK_BENCHMARK_HPP
