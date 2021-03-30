#if !defined(AMT_BENCHMARK_MTV_PRODUCT_HPP)
#define AMT_BENCHMARK_MTV_PRODUCT_HPP

#include <boost/numeric/ublas/tensor.hpp>
#include <optional>
#include <cache_manager.hpp>
#include <utils.hpp>
#include <array>
#include <thread_utils.hpp>
#include <simd_loop.hpp>

namespace amt {

    namespace impl{        

        template<typename ValueType, typename SizeType, typename SIMDLoop>
        AMT_ALWAYS_INLINE void mtv_helper_loop(ValueType* c, 
            ValueType const* a, SizeType const wa, SizeType const na, ValueType const* b, 
            SizeType const nb, SizeType const block, SIMDLoop const& simd_loop
        ) noexcept{
            static_assert(SIMDLoop::type == impl::SIMD_PROD_TYPE::MTV);
            if(nb == 0ul) return;
            auto ai = a;
            auto bi = b;
            auto ci = c;
            #pragma omp for schedule(dynamic)
            for(auto i = 0ul; i < na; i += block){
                auto aj = ai + i;
                auto bj = bi;
                auto cj = ci + i;
                auto ib = std::min(block,na-i);
                for(auto j = 0ul; j < nb; ++j){
                    auto jw = j * SIMDLoop::size;
                    auto ak = aj + jw * wa;
                    auto bk = bj + jw;
                    auto ck = cj;
                    simd_loop(ck, ak, bk, ib, wa);
                }
            }
        }

        template<typename ValueType, typename SizeType>
        AMT_ALWAYS_INLINE void mtv_helper_switch(ValueType* c, 
            ValueType const* a, SizeType const wa, SizeType const na, ValueType const* b, 
            SizeType const nb, SizeType const block, SizeType const small_block
        ) noexcept{
            constexpr auto simd_type = impl::SIMD_PROD_TYPE::MTV;

            auto wraped_fn = [a,b,c,wa,na,nb,block](auto&& loop){
                impl::mtv_helper_loop(c,a,wa,na,b,nb,block,std::forward<decltype(loop)>(loop));
            };
            switch(small_block){
                case 0: break;
                case 1: wraped_fn(impl::simd_loop<simd_type,1ul>{}); break;
                case 2: wraped_fn(impl::simd_loop<simd_type,2ul>{}); break;
                case 4: wraped_fn(impl::simd_loop<simd_type,4ul>{}); break;
                case 8: wraped_fn(impl::simd_loop<simd_type,8ul>{}); break;
                case 16: wraped_fn(impl::simd_loop<simd_type,16ul>{}); break;
                case 32: wraped_fn(impl::simd_loop<simd_type,32ul>{}); break;
                case 64: wraped_fn(impl::simd_loop<simd_type,64ul>{}); break;
                case 128: wraped_fn(impl::simd_loop<simd_type,128ul>{}); break;
                case 256: wraped_fn(impl::simd_loop<simd_type,256ul>{}); break;
                case 512: wraped_fn(impl::simd_loop<simd_type,512ul>{}); break;
                default: assert(false && "Unknown block size");
            }
        }

    } // namespace impl
    

    template<typename ValueType, typename SizeType>
    AMT_ALWAYS_INLINE void mtv_helper(
        ValueType* c, [[maybe_unused]] SizeType const* nc,
        ValueType const* a, [[maybe_unused]] SizeType const* na, [[maybe_unused]] SizeType const* wa,
        ValueType const* b, [[maybe_unused]] SizeType const* nb,
        [[maybe_unused]] int max_threads,
        boost::numeric::ublas::layout::first_order
    ) noexcept
    {
        auto const NA = nc[0] * nc[1];
        auto const NB = nb[0] * nb[1];
        auto const WA = std::max(wa[0],wa[1]);

        [[maybe_unused]] static SizeType const number_of_el_l1 = cache_manager::size(0) / sizeof(ValueType);
        [[maybe_unused]] static SizeType const half_block = number_of_el_l1>>1;
        [[maybe_unused]] static SizeType small_block = sqrt_pow_of_two(number_of_el_l1);
        [[maybe_unused]] static SizeType total_size = small_block * small_block;
        [[maybe_unused]] SizeType const block1 = (NA > number_of_el_l1 ? number_of_el_l1 : small_block);
        [[maybe_unused]] SizeType const block2 = (NA > total_size ? 1ul : small_block);
        
        auto ai = a;
        auto bi = b;
        auto ci = c;
        auto Nitr = NB / block2;
        auto Nrem = NB % block2;

        #pragma omp parallel
        {
            impl::mtv_helper_switch(ci,ai,WA,NA,bi,Nrem,block1,1ul);
            impl::mtv_helper_switch(ci,ai + Nrem * WA,WA,NA,bi + Nrem,Nitr, block1, block2);
        }

        
    }

    template<typename ValueType, typename SizeType>
    AMT_ALWAYS_INLINE void mtv_helper(
        ValueType* c, [[maybe_unused]] SizeType const* nc,
        ValueType const* a, [[maybe_unused]] SizeType const* na, [[maybe_unused]] SizeType const* wa,
        ValueType const* b, [[maybe_unused]] SizeType const* nb,
        [[maybe_unused]] int max_threads,
        boost::numeric::ublas::layout::last_order
    ) noexcept
    {
        constexpr auto simd_type = impl::SIMD_PROD_TYPE::INNER;
        auto simd_loop = impl::simd_loop<simd_type>{};
        auto ai = a;
        auto bi = b;
        auto ci = c;
        auto const NA = na[0];
        auto const NB = nb[0] * nb[1];
        auto const WA = std::max(wa[0],wa[1]);

        #pragma omp parallel for schedule(static) if(NA > 256)
        for(auto i = 0ul; i < NA; ++i){
            auto aj = ai + i * WA;
            auto bj = bi;
            auto cj = ci + i;
            *cj = simd_loop(aj,bj,NB);
        }
        
    }

    template<typename Out, typename E1, typename E2>
    constexpr auto mtv(
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
        using layout_type       = typename tensor_type1::layout_type;
        
        static_assert(
            std::is_same_v< value_type1, value_type2 > && 
            std::is_same_v< out_value_type, value_type2 >,
            "both tensor type and result type must be of same value_type"
        );

        auto const& na = a.extents();
        auto const& nb = b.extents();
        auto const& nc = c.extents();

        if( !( boost::numeric::ublas::is_matrix(na) && boost::numeric::ublas::is_vector(nb) && boost::numeric::ublas::is_vector(nc) ) ) {
            throw std::runtime_error(
                "amt::mtv(boost::numeric::ublas::tensor_core<Out>& c, boost::numeric::ublas::tensor_core<E1> const& a, boost::numeric::ublas::tensor_core<E2> const& b) : "
                "c and b must be vector, and a must be a matrix"
            );
        }
        
        std::size_t NB = boost::numeric::ublas::product(nb);
        std::size_t NC = boost::numeric::ublas::product(nc);
        
        threads::set_num_threads(num_threads);
        auto nths = threads::get_num_threads();
        
        if( (na[1] != NB ) || (na[0] != NC) ){
            throw std::runtime_error(
                "amt::mtv(boost::numeric::ublas::tensor_core<Out>&, boost::numeric::ublas::tensor_core<E1> const&, boost::numeric::ublas::tensor_core<E2> const&) : "
                "dimension mismatch"
            );
        }

        using size_type = std::decay_t< std::remove_pointer_t<decltype(na.data())> >;

        auto const* aptr = a.data();
        auto const* bptr = b.data();
        auto* cptr = c.data();
        auto na_ptr = na.data();
        auto nb_ptr = nb.data();
        auto nc_ptr = nc.data();
        std::array<size_type,2> wa = {na[1], 1ul};
        
        if constexpr( std::is_same_v<layout_type, boost::numeric::ublas::layout::first_order> ) wa = {1ul, na[0]};

        return [cptr,nc_ptr,aptr,na_ptr,wa,bptr,nb_ptr,nths]{
            mtv_helper(cptr,nc_ptr,aptr,na_ptr,wa.data(),bptr,nb_ptr,nths, layout_type{});
        };

    }

    template<typename Out, typename E1, typename E2>
    constexpr auto vtm(
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
        using layout_type       = typename tensor_type1::layout_type;
        
        static_assert(
            std::is_same_v< value_type1, value_type2 > && 
            std::is_same_v< out_value_type, value_type2 >,
            "both tensor type and result type must be of same value_type"
        );

        auto const& na = a.extents();
        auto const& nb = b.extents();
        auto const& nc = c.extents();

        if( !( boost::numeric::ublas::is_matrix(na) && boost::numeric::ublas::is_vector(nb) && boost::numeric::ublas::is_vector(nc) ) ) {
            throw std::runtime_error(
                "amt::mtv(boost::numeric::ublas::tensor_core<Out>& c, boost::numeric::ublas::tensor_core<E1> const& a, boost::numeric::ublas::tensor_core<E2> const& b) : "
                "c and b must be vector, and a must be a matrix"
            );
        }
        
        std::size_t NB = boost::numeric::ublas::product(nb);
        std::size_t NC = boost::numeric::ublas::product(nc);

        threads::set_num_threads(num_threads);
        auto nths = threads::get_num_threads();
        
        if( (na[1] != NC ) || (na[0] != NB) ){
            throw std::runtime_error(
                "amt::mtv(boost::numeric::ublas::tensor_core<Out>&, boost::numeric::ublas::tensor_core<E1> const&, boost::numeric::ublas::tensor_core<E2> const&) : "
                "dimension mismatch"
            );
        }
        
        constexpr bool is_first_order = std::is_same_v<layout_type, boost::numeric::ublas::layout::first_order>;

        using size_type = std::decay_t< std::remove_pointer_t<decltype(na.data())> >;
        using other_layout_type = std::conditional_t<
            is_first_order,
            boost::numeric::ublas::layout::last_order,
            boost::numeric::ublas::layout::first_order
        >;

        auto const* aptr = a.data();
        auto const* bptr = b.data();
        auto* cptr = c.data();
        auto nb_ptr = nb.data();
        auto nc_ptr = nc.data();
        std::array<size_type,2> new_na = {na[1],na[0]};
        std::array<size_type,2> wa = {new_na[1], 1ul};
        
        if constexpr( is_first_order) wa = {1ul, new_na[0]};
        // c = Av
        // c^T = v^TA^T 
        // c = vA^T

        return [cptr,nc_ptr,aptr,new_na,wa,bptr,nb_ptr,nths]{
            mtv_helper(cptr,nc_ptr,aptr,new_na.data(),wa.data(),bptr,nb_ptr,nths, other_layout_type{});
        };

    }

} // namespace amt

#endif // AMT_BENCHMARK_MTV_PRODUCT_HPP
