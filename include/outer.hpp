#if !defined(AMT_BENCHMARK_OUTER_PRODUCT_HPP)
#define AMT_BENCHMARK_OUTER_PRODUCT_HPP

#include <boost/numeric/ublas/tensor.hpp>
#include <omp.h>
#include <optional>
#include <cache_manager.hpp>
#include <cstdlib>
#include <macros.hpp>
#include <x86intrin.h>

namespace amt {

    template<typename Out, typename In, typename SizeType>
    AMT_ALWAYS_INLINE void outer_prod_helper(
        Out* c,
        [[maybe_unused]] SizeType const wc,
        In const* a,
        [[maybe_unused]] SizeType const na,
        In const* b,
        [[maybe_unused]] SizeType const nb,
        [[maybe_unused]] int max_threads
    ) noexcept
    {
        static_assert(std::is_same_v<Out,In>);
        namespace ub = boost::numeric::ublas;

        using value_type = Out;
        [[maybe_unused]] constexpr auto size_in_bytes = sizeof(value_type);

        [[maybe_unused]] static auto const number_of_el_l1 = cache_manager::size(0) / size_in_bytes;
        [[maybe_unused]] static auto const number_of_el_l2 = cache_manager::size(1) / size_in_bytes;
        [[maybe_unused]] static auto const number_of_el_l3 = cache_manager::size(2) / size_in_bytes;
        [[maybe_unused]] static auto const block = number_of_el_l1 >> 1;

        constexpr auto simd_loop = [](Out* c, In const* a, In const* b, SizeType const n){
            auto cst = *b;
            #pragma omp simd
            for(auto i = 0ul; i < n; ++i){
                c[i] += (a[i] * cst);
            }
        };

        auto ai = a;
        auto bi = b;
        auto ci = c;
        [[maybe_unused]] constexpr auto max_bl = 256ul;

        #pragma omp parallel for if(nb > max_bl)
        for(auto i = 0ul; i < nb; ++i){
            auto aj = ai;
            auto bj = bi + i;
            auto cj = ci + i * wc;
            simd_loop(cj, aj, bj, na);
        }

    }

    template<typename Out, typename E1, typename E2, typename... Timer>
    constexpr void outer_prod(
        boost::numeric::ublas::tensor_core<Out>& c,
        boost::numeric::ublas::tensor_core<E1> const& a,
        boost::numeric::ublas::tensor_core<E2> const& b,
        std::optional<std::size_t> num_threads,
        Timer&... t
    )
    {
        using out_type          = boost::numeric::ublas::tensor_core<Out>;
        using tensor_type1      = boost::numeric::ublas::tensor_core<E1>;
        using tensor_type2      = boost::numeric::ublas::tensor_core<E2>;
        using value_type1       = typename tensor_type1::value_type;
        using value_type2       = typename tensor_type2::value_type;
        using out_value_type    = typename out_type::value_type;
        using out_layout_type   = typename out_type::layout_type;
        
        static_assert(
            std::is_same_v< value_type1, value_type2 > && 
            std::is_same_v< out_value_type, value_type2 >,
            "both tensor type and result type must be of same value_type"
        );

        static_assert(
            sizeof...(Timer) <= 1u,
            "there can only be one profiler"
        );
        
        std::tuple<Timer&...> timer(t...);
        auto const& na = a.extents();
        auto const& nb = b.extents();

        if( !( boost::numeric::ublas::is_vector(na) && boost::numeric::ublas::is_vector(nb) ) ) {
            throw std::runtime_error(
                "amt::outer_prod(boost::numeric::ublas::tensor_core<Out>&, boost::numeric::ublas::tensor_core<E1> const&, boost::numeric::ublas::tensor_core<E2> const&) : "
                "both tensor must be vector"
            );
        }
        
        std::size_t NA = boost::numeric::ublas::product(na);
        std::size_t NB = boost::numeric::ublas::product(nb);

        auto max_num_threads = 1;
        if( char const* omp_env_var = std::getenv("OMP_NUM_THREADS"); omp_env_var != nullptr ){
            max_num_threads = std::atoi(omp_env_var);
        }else{
            max_num_threads = omp_get_max_threads();
        }
        
        auto nths = static_cast<int>(num_threads.value_or(max_num_threads));
        omp_set_num_threads(nths);
        
        if( c.size(0) != NA || c.size(1) != NB ){
            throw std::runtime_error(
                "amt::outer_prod(boost::numeric::ublas::tensor_core<Out>&, boost::numeric::ublas::tensor_core<E1> const&, boost::numeric::ublas::tensor_core<E2> const&) : "
                "dimension mismatch"
            );
        }

        auto const* aptr = a.data();
        auto const* bptr = b.data();
        auto* cptr = c.data();
        

        if constexpr(sizeof...(Timer) > 0u) std::get<0>(timer).start();
        
        if constexpr( std::is_same_v<out_layout_type, boost::numeric::ublas::layout::first_order> ){
            auto const WC = c.strides()[1];
            outer_prod_helper(cptr,WC,aptr,NA,bptr,NB,nths);
        }else{
            auto const WC = c.strides()[0];
            outer_prod_helper(cptr,WC,bptr,NB,aptr,NA,nths);
        }

        if constexpr(sizeof...(Timer) > 0u) std::get<0>(timer).stop();

    }

} // namespace amt

#endif // AMT_BENCHMARK_OUTER_PRODUCT_HPP
