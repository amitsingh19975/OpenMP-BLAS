#if !defined(AMT_BENCHMARK_UTILS_HPP)
#define AMT_BENCHMARK_UTILS_HPP

#include <unordered_map>
#include <ostream>
#include <utility>
#include <tuple>
#include <type_traits>
#include <macros.hpp>
#include <boost/numeric/ublas/tensor/layout.hpp>

namespace amt{

    struct speed_t{
        std::size_t one{};
        std::size_t two{};
    };
    
    void show_intersection_pts(std::ostream& os, 
        std::unordered_map<std::string_view, std::pair<speed_t,speed_t> > const& pts
    ){
        os <<"\n---------Intersection Points---------\n";
        for(auto const& [k,v] : pts){
            auto [up,down] = v;
            os << k << ": [ ( 1 => U: " << up.one<<", D: "<<down.one <<" ), ( 2 => U: "<< up.two<<", D: "<<down.two <<" ) ]\n";
        }
        os <<'\n';
    }

    AMT_ALWAYS_INLINE constexpr std::pair<std::size_t,std::size_t> sqrt_pow_of_two(std::size_t N) noexcept{
        std::size_t p = 0;
        N >>= 1;
        while(N) {
            N >>= 1;
            ++p;
        }
        p >>=1;
        return {1ul<<p, p};
    }

    AMT_ALWAYS_INLINE constexpr auto nearest_power_of_two(std::size_t N) noexcept{
        std::size_t p = 0;
        if (N && !(N & (N - 1)))
            return N;
        
        while(N > (1<<(p+1))){
            ++p;
        }
        return 1ul<<p;
    }

    template<typename T>
    struct is_first_order : std::false_type{};
    
    template<typename T>
    constexpr static bool is_first_order_v = is_first_order<T>::value;
    
    template<>
    struct is_first_order<boost::numeric::ublas::layout::first_order> : std::true_type{};

    template<typename T>
    struct is_last_order : std::false_type{};
    
    template<typename T>
    constexpr static bool is_last_order_v = is_last_order<T>::value;

    template<>
    struct is_last_order<boost::numeric::ublas::layout::last_order> : std::true_type{};

} // namespace amt


#endif // AMT_BENCHMARK_UTILS_HPP
