#if !defined(AMT_BENCHMARK_MTV_PRODUCT_HPP)
#define AMT_BENCHMARK_MTV_PRODUCT_HPP

#include <boost/numeric/ublas/tensor.hpp>
#include <optional>
#include <cache_manager.hpp>
#include <macros.hpp>
#include <array>
#include <thread_utils.hpp>

namespace amt {

    AMT_ALWAYS_INLINE constexpr auto sqrt_pow_of_two(std::size_t N) noexcept{
        std::size_t p = 0;
        N >>= 1;
        while(N) {
            N >>= 1;
            ++p;
        }
        return 1u << (p >> 1);
    }

    template<std::size_t N, typename ValueType, typename SizeType>
    AMT_ALWAYS_INLINE void simd_loop(ValueType* c, ValueType const* a, ValueType const* b, SizeType const n, SizeType const w) noexcept{
        #pragma omp simd
        for(auto i = 0ul; i < n; ++i){
            #pragma unroll(N)
            for(auto j = 0ul; j < N; ++j){
                c[i] += a[i + w * j] * b[j];
            }
        }
    }

    template<std::size_t N, typename ValueType, typename SizeType>
    AMT_ALWAYS_INLINE void mtv_helper_loop(ValueType* c, 
        ValueType const* a, SizeType const wa, SizeType const na, ValueType const* b, 
        SizeType const nb, SizeType const block
    ) noexcept{
        if(nb == 0ul) return;
        auto ai = a;
        auto bi = b;
        auto ci = c;
        #pragma omp parallel for schedule(dynamic)
        for(auto i = 0ul; i < na; i += block){
            auto aj = ai + i;
            auto bj = bi;
            auto cj = ci + i;
            auto ib = std::min(block,na-i);
            for(auto j = 0ul; j < nb; ++j){
                auto jw = j * N;
                auto ak = aj + jw * wa;
                auto bk = bj + jw;
                auto ck = cj;
                simd_loop<N>(ck, ak, bk, ib, wa);
            }
        }
    }

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
        [[maybe_unused]] static SizeType const small_block = sqrt_pow_of_two(number_of_el_l1);
        [[maybe_unused]] SizeType const block = (NA > number_of_el_l1 ? half_block : small_block);
        
        auto ai = a;
        auto bi = b;
        auto ci = c;
        auto Nitr = NB / small_block;
        auto Nrem = NB % small_block;

        mtv_helper_loop<1ul>(ci,ai,WA,NA,bi,Nrem,block);
        ai += Nrem * WA;
        bi += Nrem;

        switch(small_block){
            case 8: mtv_helper_loop<8ul>(ci,ai,WA,NA,bi,Nitr,block); break;
            case 16: mtv_helper_loop<16ul>(ci,ai,WA,NA,bi,Nitr,block); break;
            case 32: mtv_helper_loop<32ul>(ci,ai,WA,NA,bi,Nitr,block); break;
            case 128: mtv_helper_loop<128ul>(ci,ai,WA,NA,bi,Nitr,block); break;
            default: mtv_helper_loop<64ul>(ci,ai,WA,NA,bi,Nitr,block); break;
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
        constexpr auto simd_loop = [](ValueType* c, ValueType const* const a, ValueType const* const b, SizeType const n){
            auto sum = *c;
            #pragma omp simd
            for(auto i = 0ul; i < n; ++i){
                sum += a[i] * b[i];
            }
            *c = sum;
        };

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
            simd_loop(cj,aj,bj,NB);
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
