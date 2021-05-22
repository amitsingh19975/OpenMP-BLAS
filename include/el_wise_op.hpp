#if !defined(AMT_BENCHMARK_ELEMENT_WISE_OPERATION_HPP)
#define AMT_BENCHMARK_ELEMENT_WISE_OPERATION_HPP

#include <boost/numeric/ublas/tensor.hpp>
#include <thread_utils.hpp>

namespace amt::el_op {

    namespace detail{
        

        template<typename T>
        struct is_std_pair : std::false_type{};
        
        template<typename T0, typename T1>
        struct is_std_pair< std::pair<T0, T1> > : std::true_type{};

        template<typename T>
        constexpr static bool is_std_pair_v = is_std_pair<std::decay_t<T>>::value;

        struct identity_fn{
            template<typename T, typename... Ts>
            constexpr decltype(auto) operator()(T&& val, Ts&&...) const noexcept{
                return val;
            }
            
            constexpr void operator()() const noexcept{}
        };

    } // namespace detail
    
    
    template<typename Out, typename E1, typename E2, typename Predicate>
    auto apply(
        boost::numeric::ublas::tensor_core<Out>& c,
        boost::numeric::ublas::tensor_core<E1> const& a,
        boost::numeric::ublas::tensor_core<E2> const& b,
        std::optional<std::size_t>,
        Predicate&& pred
    ){
        using out_type          = boost::numeric::ublas::tensor_core<Out>;
        using tensor_type1      = boost::numeric::ublas::tensor_core<E1>;
        using tensor_type2      = boost::numeric::ublas::tensor_core<E1>;
        using value_type1       = typename tensor_type1::value_type;
        using value_type2       = typename tensor_type2::value_type;
        using out_value_type    = typename out_type::value_type;

        static_assert(
            std::is_same_v< out_value_type, value_type1 > &&
            std::is_same_v< out_value_type, value_type2 > ,
            "all the value types must be of same value type"
        );

        // if(c.size() != a.size() && a.size() != b.size()){
        //     throw std::runtime_error(
        //         "amt::apply(boost::numeric::ublas::tensor_core<Out>& c, boost::numeric::ublas::tensor_core<E1> const& a, boost::numeric::ublas::tensor_core<E2> const& b) : "
        //         "a , b, and c must of same size"
        //     );
        // }

        // threads::set_num_threads(num_threads);
        auto sz = c.size();
        #pragma omp simd
        for(auto i = 0ul; i < sz; ++i){
            c[i] = std::invoke(pred, a[i], b[i]);
        }
        
    }

    template<typename Out, typename E1, typename E2>
    constexpr auto add(
        boost::numeric::ublas::tensor_core<Out>& c,
        boost::numeric::ublas::tensor_core<E1> const& a,
        boost::numeric::ublas::tensor_core<E2> const& b,
        std::optional<std::size_t> num_threads
    ){
        apply(c, a, b, num_threads, std::plus<>{});
    }

    template<typename Out, typename E1, typename E2>
    constexpr auto sub(
        boost::numeric::ublas::tensor_core<Out>& c,
        boost::numeric::ublas::tensor_core<E1> const& a,
        boost::numeric::ublas::tensor_core<E2> const& b,
        std::optional<std::size_t> num_threads
    ){
        apply(c, a, b, num_threads, std::minus<>{});
    }

    template<typename Out, typename E1, typename E2>
    constexpr auto mul(
        boost::numeric::ublas::tensor_core<Out>& c,
        boost::numeric::ublas::tensor_core<E1> const& a,
        boost::numeric::ublas::tensor_core<E2> const& b,
        std::optional<std::size_t> num_threads
    ){
        apply(c, a, b, num_threads, std::multiplies<>{});
    }

    template<typename Out, typename E1, typename E2>
    constexpr auto div(
        boost::numeric::ublas::tensor_core<Out>& c,
        boost::numeric::ublas::tensor_core<E1> const& a,
        boost::numeric::ublas::tensor_core<E2> const& b,
        std::optional<std::size_t> num_threads
    ){
        apply(c, a, b, num_threads, std::divides<>{});
    }

    template<typename Out, typename E1, typename E2, typename Predicate>
    constexpr auto apply_ops(
        boost::numeric::ublas::tensor_core<Out>& c,
        boost::numeric::ublas::tensor_core<E1> const& a,
        boost::numeric::ublas::tensor_core<E2> const& b,
        std::optional<std::size_t> num_threads,
        Predicate&& pred
    ){
        apply(c, a, b, num_threads, std::forward<Predicate>(pred));
    }



} // namespace amt::el_op

#endif // AMT_BENCHMARK_ELEMENT_WISE_OPERATION_HPP
