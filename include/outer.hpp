#if !defined(AMT_BENCHMARK_OUTER_PRODUCT_HPP)
#define AMT_BENCHMARK_OUTER_PRODUCT_HPP

#include <boost/numeric/ublas/tensor.hpp>
#include <optional>
#include <cache_manager.hpp>
#include <macros.hpp>
#include <thread_utils.hpp>
#include <simd_loop.hpp>

namespace amt {

    template<std::size_t MinSize = 256ul, typename ValueType, typename SizeType>
    AMT_ALWAYS_INLINE void outer_prod_helper(
        ValueType* c, [[maybe_unused]] SizeType const* nc, [[maybe_unused]] SizeType const* wc,
        ValueType const* a, [[maybe_unused]] SizeType const* na,
        ValueType const* b, [[maybe_unused]] SizeType const* nb,
        [[maybe_unused]] int max_threads
    ) noexcept
    {
        [[maybe_unused]] static auto const number_of_el_l2 = cache_manager::size(1) / sizeof(ValueType);
        
        constexpr auto simd_type = impl::SIMD_PROD_TYPE::OUTER;
        constexpr auto simd_loop = impl::simd_loop<simd_type>{};

        auto const NA = na[0] * na[1];
        auto const NB = nb[0] * nb[1];
        auto const WC = std::max(wc[0],wc[1]);

        auto ai = a;
        auto bi = b;
        auto ci = c;

        auto NA_temp = static_cast<int>(NA);
        auto num_ths = (static_cast<int>(number_of_el_l2) - NA_temp) / NA_temp;
        auto ths = std::max(1, std::min(max_threads,num_ths));
        threads::set_num_threads(ths);

        #pragma omp parallel for if(NB > MinSize)
        for(auto i = 0ul; i < NB; ++i){
            auto aj = ai;
            auto bj = bi + i;
            auto cj = ci + i * WC;
            simd_loop(cj, aj, bj, NA);
        }
        
    }

    template<typename Out, typename E1, typename E2>
    constexpr auto outer_prod(
        boost::numeric::ublas::tensor_core<Out>& c,
        boost::numeric::ublas::tensor_core<E1> const& a,
        boost::numeric::ublas::tensor_core<E2> const& b,
        std::optional<std::size_t> num_threads
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

        auto const& na = a.extents();
        auto const& nb = b.extents();
        auto const& nc = c.extents();

        if( !( boost::numeric::ublas::is_vector(na) && boost::numeric::ublas::is_vector(nb) ) ) {
            throw std::runtime_error(
                "amt::outer_prod(boost::numeric::ublas::tensor_core<Out>&, boost::numeric::ublas::tensor_core<E1> const&, boost::numeric::ublas::tensor_core<E2> const&) : "
                "both tensor must be vector"
            );
        }
        
        std::size_t NA = boost::numeric::ublas::product(na);
        std::size_t NB = boost::numeric::ublas::product(nb);

        threads::set_num_threads(num_threads);
        auto nths = threads::get_num_threads();
        
        if( nc[0] != NA || nc[1] != NB ){
            throw std::runtime_error(
                "amt::outer_prod(boost::numeric::ublas::tensor_core<Out>&, boost::numeric::ublas::tensor_core<E1> const&, boost::numeric::ublas::tensor_core<E2> const&) : "
                "dimension mismatch"
            );
        }

        using size_type = std::decay_t< std::remove_pointer_t<decltype(nc.data())> >;

        auto const* aptr = a.data();
        auto const* bptr = b.data();
        auto* cptr = c.data();
        auto const* na_ptr = na.data();
        auto const* nb_ptr = nb.data();
        auto const* nc_ptr = nc.data();
        
        
        if constexpr( std::is_same_v<out_layout_type, boost::numeric::ublas::layout::first_order> ){
            return [cptr,nc_ptr,aptr,na_ptr,bptr,nb_ptr,nths]{
                size_type const wc[2] = {1ul, nc_ptr[0]};
                outer_prod_helper(cptr,nc_ptr,wc,aptr,na_ptr,bptr,nb_ptr,nths);
            };
        }else{
            return [cptr,nc_ptr,aptr,na_ptr,bptr,nb_ptr,nths]{
                size_type const wc[2] = {nc_ptr[1],1ul};
                outer_prod_helper(cptr,nc_ptr,wc,bptr,nb_ptr,aptr,na_ptr,nths);
            };
        }

    }

} // namespace amt

#endif // AMT_BENCHMARK_OUTER_PRODUCT_HPP
