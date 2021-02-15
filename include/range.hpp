#if !defined(AMT_BENCHMARK_RANGE_HPP)
#define AMT_BENCHMARK_RANGE_HPP

#include <cstdint>
#include <exception>
#include <type_traits>

namespace amt{
    
    template<typename Container, typename ValueType = typename Container::value_type>
    constexpr void range(Container& c, ValueType start, ValueType stride = 1){
        for(auto it = c.begin(); it != c.end(); start += stride, ++it){
            *it = start;
        }
    }

    template<typename Container, typename Fn, typename ValueType = typename Container::value_type, 
        std::enable_if<std::is_invocable_v<Fn>>* = nullptr
    >
    constexpr void range(Container& c, ValueType start, ValueType stride, Fn&& fn){
        for(auto it = c.begin(); it != c.end(); start = std::invoke(std::forward<Fn>(fn), start, stride), ++it){
            *it = start;
        }
    }

    template<typename Container, typename Fn, typename ValueType = typename Container::value_type,
        std::enable_if<std::is_invocable_v<Fn>>* = nullptr
    >
    constexpr void range(Container& c, ValueType start, ValueType end, ValueType stride, Fn&& fn){
        if(start > end) 
            throw std::runtime_error("amt::range(Container&, ValueType, ValueType, ValueType, Fn&&) : start > end");
        std::size_t sz{};
        for(auto i = start; i < end; i = std::invoke(std::forward<Fn>(fn), i, stride)){
            ++sz;
        }
        c.resize(sz);
        for(auto it = c.begin(); it != c.end(); start = std::invoke(std::forward<Fn>(fn), start, stride), ++it){
            *it = start;
        }
    }

    template<typename Container, typename ValueType = typename Container::value_type>
    constexpr void range(Container& c, ValueType start, ValueType end, ValueType stride){
        if(start > end) 
            throw std::runtime_error("amt::range(Container&, ValueType, ValueType, ValueType, Fn&&) : start > end");
        c.resize(static_cast<std::size_t>(end - start));
        for(auto it = c.begin(); it != c.end(); start += stride, ++it){
            *it = start;
        }
    }

} // namespace amt


#endif // AMT_BENCHMARK_RANGE_HPP
