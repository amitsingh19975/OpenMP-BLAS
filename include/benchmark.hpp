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
    
    AMT_ALWAYS_INLINE void clobber_mem() {
       asm volatile("" : : : "memory");
    }
#else
    #error "Compiler is not supported"
#endif

template<std::size_t MaxIter = 100u, typename FnType, typename... FnArgs>
constexpr auto benchmark(FnType&& fn, FnArgs&&... args) noexcept{
    double time{};
    auto t = timer{};
    using ret_type = std::invoke_result_t<FnType,FnArgs...>;
    for(auto i = 0ul; i < MaxIter; ++i){
        
        defer(t.start(), t.stop()){
            if constexpr(std::is_same_v<ret_type,void>){
                std::invoke(std::forward<FnType>(fn), std::forward<FnArgs>(args)...);
            }else{
                no_opt(std::invoke(std::forward<FnType>(fn), std::forward<FnArgs>(args)...));
            }
        }

        time += t();
    }
    return time / static_cast<double>(MaxIter);
}

} // namespace amt


#endif // AMT_BENCHMARK_BENCHMARK_HPP
