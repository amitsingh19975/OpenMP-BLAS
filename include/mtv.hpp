#if !defined(AMT_BENCHMARK_MTV_PRODUCT_HPP)
#define AMT_BENCHMARK_MTV_PRODUCT_HPP

#include <boost/numeric/ublas/tensor.hpp>
#include <omp.h>
#include <optional>
#include <cache_manager.hpp>
#include <cstdlib>
#include <macros.hpp>

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
    AMT_ALWAYS_INLINE void mtv_helper_nitr(ValueType* c, 
        ValueType const* a, SizeType const wa, SizeType const na, ValueType const* b, 
        SizeType const nb, SizeType const block
    ) noexcept{
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

    template<typename Out, typename In, typename SizeType>
    AMT_ALWAYS_INLINE void mtv_helper(
        Out* c,
        In const* a,
        [[maybe_unused]] SizeType const wa,
        [[maybe_unused]] SizeType const na,
        In const* b,
        [[maybe_unused]] SizeType const nb,
        [[maybe_unused]] int max_threads,
        boost::numeric::ublas::layout::first_order
    ) noexcept
    {
        static_assert(std::is_same_v<Out,In>);

        [[maybe_unused]] static SizeType const number_of_el_l1 = cache_manager::size(0) / sizeof(In);
        [[maybe_unused]] static SizeType const small_block = sqrt_pow_of_two(number_of_el_l1);
        [[maybe_unused]] SizeType const block = (na > number_of_el_l1 ? number_of_el_l1>>1 : small_block);
        
        auto ai = a;
        auto bi = b;
        auto ci = c;
        auto Nitr = nb / small_block;
        auto Nrem = nb % small_block;

        if(Nrem != 0ul){
            #pragma omp parallel for schedule(dynamic) if (na > small_block)
            for(auto i = 0ul; i < na; i += block){
                auto aj = ai + i;
                auto bj = bi;
                auto cj = ci + i;
                auto ib = std::min(block,na-i);
                for(auto j = 0ul; j < Nrem; ++j){
                    auto ak = aj + j * wa;
                    auto bk = bj + j;
                    auto ck = cj;
                    simd_loop<1ul>(ck,ak,bk,ib,wa);
                }
            }
            ai += Nrem * wa;
            bi += Nrem;
        }

        if(Nitr != 0ul){
            switch(small_block){
                case 8: mtv_helper_nitr<8ul>(ci,ai,wa,na,bi,Nitr,block); break;
                case 16: mtv_helper_nitr<16ul>(ci,ai,wa,na,bi,Nitr,block); break;
                case 32: mtv_helper_nitr<32ul>(ci,ai,wa,na,bi,Nitr,block); break;
                case 128: mtv_helper_nitr<128ul>(ci,ai,wa,na,bi,Nitr,block); break;
                default: mtv_helper_nitr<64ul>(ci,ai,wa,na,bi,Nitr,block); break;
            }
        }

        
    }

    template<typename Out, typename In, typename SizeType>
    AMT_ALWAYS_INLINE void mtv_helper(
        Out* c,
        In const* a,
        [[maybe_unused]] SizeType const wa,
        [[maybe_unused]] SizeType const na,
        In const* b,
        [[maybe_unused]] SizeType const nb,
        [[maybe_unused]] int max_threads,
        boost::numeric::ublas::layout::last_order
    ) noexcept
    {
        static_assert(std::is_same_v<Out,In>);

        constexpr auto simd_loop = [](Out* c, In const* const a, In const* const b, SizeType const n){
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

        #pragma omp parallel for schedule(static) if(na > 256)
        for(auto i = 0ul; i < na; ++i){
            auto aj = ai + i * wa;
            auto bj = bi;
            auto cj = ci + i;
            simd_loop(cj,aj,bj,nb);
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

        if( !( boost::numeric::ublas::is_matrix(na) && boost::numeric::ublas::is_vector(nb) ) ) {
            throw std::runtime_error(
                "amt::mtv(boost::numeric::ublas::tensor_core<Out>&, boost::numeric::ublas::tensor_core<E1> const&, boost::numeric::ublas::tensor_core<E2> const&) : "
                "both tensor must be vector"
            );
        }
        
        std::size_t NB = boost::numeric::ublas::product(nb);
        std::size_t NC = boost::numeric::ublas::product(nc);

        auto max_num_threads = 1;
        if( char const* omp_env_var = std::getenv("OMP_NUM_THREADS"); omp_env_var != nullptr ){
            max_num_threads = std::atoi(omp_env_var);
        }else{
            max_num_threads = omp_get_max_threads();
        }
        
        auto nths = static_cast<int>(num_threads.value_or(max_num_threads));
        omp_set_num_threads(nths);
        
        if( (na[1] != NB ) || (na[0] != NC) ){
            throw std::runtime_error(
                "amt::mtv(boost::numeric::ublas::tensor_core<Out>&, boost::numeric::ublas::tensor_core<E1> const&, boost::numeric::ublas::tensor_core<E2> const&) : "
                "dimension mismatch"
            );
        }

        auto const* aptr = a.data();
        auto const* bptr = b.data();
        auto* cptr = c.data();
        auto WA = a.strides()[0];
        auto NA = na[0];
        
        if constexpr( std::is_same_v<layout_type, boost::numeric::ublas::layout::first_order> ) WA = a.strides()[1];

        return [NA,cptr,aptr,WA,bptr,NB,nths]{
            mtv_helper(cptr,aptr,WA,NA,bptr,NB,nths, layout_type{});
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

        if( !( boost::numeric::ublas::is_matrix(na) && boost::numeric::ublas::is_vector(nb) ) ) {
            throw std::runtime_error(
                "amt::mtv(boost::numeric::ublas::tensor_core<Out>&, boost::numeric::ublas::tensor_core<E1> const&, boost::numeric::ublas::tensor_core<E2> const&) : "
                "both tensor must be vector"
            );
        }
        
        std::size_t NB = boost::numeric::ublas::product(nb);
        std::size_t NC = boost::numeric::ublas::product(nc);

        auto max_num_threads = 1;
        if( char const* omp_env_var = std::getenv("OMP_NUM_THREADS"); omp_env_var != nullptr ){
            max_num_threads = std::atoi(omp_env_var);
        }else{
            max_num_threads = omp_get_max_threads();
        }
        
        auto nths = static_cast<int>(num_threads.value_or(max_num_threads));
        omp_set_num_threads(nths);
        
        if( (na[1] != NB ) || (na[0] != NC) ){
            throw std::runtime_error(
                "amt::mtv(boost::numeric::ublas::tensor_core<Out>&, boost::numeric::ublas::tensor_core<E1> const&, boost::numeric::ublas::tensor_core<E2> const&) : "
                "dimension mismatch"
            );
        }

        auto const* aptr = a.data();
        auto const* bptr = b.data();
        auto* cptr = c.data();
        auto NA = na[0];

        // c = Av
        // c^T = v^TA^T 
        
        if constexpr( std::is_same_v<layout_type, boost::numeric::ublas::layout::first_order> ) {
            auto WA = na[1];
            return [NA,cptr,aptr,WA,bptr,NB,nths]{
                mtv_helper(cptr,aptr,WA,NA,bptr,NB,nths, boost::numeric::ublas::layout::last_order{});
            };
        }else {
            auto WA = na[0];
            return [NA,cptr,aptr,WA,bptr,NB,nths]{
                mtv_helper(cptr,aptr,WA,NA,bptr,NB,nths, boost::numeric::ublas::layout::first_order{});
            };
        }


    }

} // namespace amt

#endif // AMT_BENCHMARK_MTV_PRODUCT_HPP
