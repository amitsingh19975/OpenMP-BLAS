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

template<std::size_t MaxIter = 100u, typename FnType, typename... FnArgs>
constexpr auto benchmark(FnType&& fn, FnArgs&&... args) noexcept{
    double time{};
    auto t = timer{};
    for(auto i = 0ul; i < MaxIter; ++i){
        t.start();
        std::invoke(std::forward<FnType>(fn), std::forward<FnArgs>(args)...);
        time += t.stop();
    }
    return time / static_cast<double>(MaxIter);
}

template<std::size_t MaxIter = 100u, typename FnType, typename... FnArgs>
constexpr double benchmark_timer_as_arg(FnType&& fn, FnArgs&&... args) noexcept{
    double time{};
    auto t = timer{};
    for(auto i = 0ul; i < MaxIter; ++i){
        t.start();
        std::invoke(std::forward<FnType>(fn), std::forward<FnArgs>(args)..., t);
        time += t.stop();
    }
    return time / static_cast<double>(MaxIter);
}

} // namespace amt


#endif // AMT_BENCHMARK_BENCHMARK_HPP
