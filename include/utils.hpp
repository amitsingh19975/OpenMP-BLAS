#if !defined(AMT_BENCHMARK_UTILS_HPP)
#define AMT_BENCHMARK_UTILS_HPP

#include <unordered_map>
#include <ostream>
#include <utility>
#include <tuple>
#include <type_traits>
#include <macros.hpp>
#include <boost/numeric/ublas/tensor.hpp>
#include <iomanip>

namespace amt{

    namespace ub = boost::numeric::ublas;
    using shape_t = ub::extents<2u>;

    template<typename T, typename L>
    using tensor_t = ub::tensor_static_rank<T,2,L>;

    template<typename T, typename L = ub::layout::first_order>
    auto make_tensor(ub::integral auto M, ub::integral auto N){
        return tensor_t<T,L>( static_cast<std::size_t>(M), static_cast<std::size_t>(N) );
    }

    template<typename T, typename L = ub::layout::first_order>
    auto make_tensor(ub::integral auto M, ub::integral auto N, T val){
        auto ret = tensor_t<T,L>( static_cast<std::size_t>(M), static_cast<std::size_t>(N) );
        std::fill(ret.begin(), ret.end(), val);
        return ret;
    }

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

    AMT_ALWAYS_INLINE constexpr auto nearest_mul_of_x(std::size_t N, std::size_t M) noexcept{
        return N - (N % M);
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

    namespace tag{
        struct trans{};
        struct inplace{};
        struct outplace{};
    }

    template<typename ValueType, typename SizeType>
    void pack(
        ValueType* out, SizeType const wo,
        ValueType const* in, SizeType const* wi,
        SizeType const m,
        SizeType const n
    ) noexcept{

        auto in0 = in;
        auto out0 = out;

        for(auto i = 0ul; i < n; ++i){
            auto in1 = in0 + wi[1] * i;
            auto out1 = out0 + wo * i;
            #pragma omp simd
            for(auto j = 0ul; j < m; ++j){
                out1[j] = in1[j * wi[0]];
            }
        }
    }

    template<typename ValueType, typename SizeType>
    void pack(
        ValueType* out, SizeType const wo,
        ValueType const* in, SizeType const* wi,
        SizeType const m,
        SizeType const n,
        tag::trans /*transpose*/
    ) noexcept{

        auto in0 = in;
        auto out0 = out;

        for(auto i = 0ul; i < n; ++i){
            auto in1 = in0 + i * wi[0];
            auto out1 = out0 + wo * i;
            #pragma omp simd
            for(auto j = 0ul; j < m; ++j){
                out1[j] = in1[j * wi[1]];
            }
        }

    }

    namespace debug
    {   
        template<typename ValueType, typename SizeType>
        void show( ValueType* in, SizeType const wm, SizeType const wn, SizeType const m, SizeType const n, int width = 2ul){
            std::stringstream ss;
            auto in0 = in;
            ss<<'\n';
            for(auto i = 0ul; i < m; ++i){
                auto in1 = in0 + wm * i;
                for(auto j = 0ul; j < n; ++j){
                    ss<<std::setfill('0')<<std::setw(width)<<in1[j * wn] << ' ';
                }
                ss<<'\n';
            }
            std::cerr<<ss.str();

        }
        
        template<typename ValueType, typename SizeType>
        void show( ValueType* in, SizeType const n, char del = '\n', int width = 2ul){
            std::stringstream ss;
            ss<<'\n';
            for(auto i = 0ul; i < n; ++i){
                ss<<std::setfill('0')<<std::setw(width)<<in[i] << del;
            }
            ss << '\n';
            std::cerr<<ss.str();

        }

        template<typename ValueType, typename SizeType>
        void show( ValueType* in, SizeType const* w, SizeType const* n, int width = 2ul){
            show(in,w[0],w[1],n[0],n[1],width);
        }
        
        template<typename ValueType, typename SizeType>
        void show( ValueType* in, SizeType const* w, SizeType const m, SizeType const n, int width = 2ul){
            show(in,w[0],w[1],m,n,width);
        }

    } // namespace debug

    constexpr double ct_sqrt(double res, double l, double r) noexcept{
        if(l == r){
            return r;
        } else {
            const auto mid = (r + l) / 2.;

            if(mid * mid >= res){
                return ct_sqrt(res, l, mid);
            } else {
                return ct_sqrt(res, mid + 1., r);
            }
        }
    }

    constexpr double ct_sqrt(double res) noexcept{
        return ct_sqrt(res, 1, res);
    }

} // namespace amt


#endif // AMT_BENCHMARK_UTILS_HPP
