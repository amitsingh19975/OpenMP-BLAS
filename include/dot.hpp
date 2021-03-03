#if !defined(AMT_BENCHMARK_DOT_PRODUCT_HPP)
#define AMT_BENCHMARK_DOT_PRODUCT_HPP

#include <boost/numeric/ublas/tensor.hpp>
#include <omp.h>
#include <optional>
#include <cache_manager.hpp>
#include <cstdlib>
#include <macros.hpp>

namespace amt {

    template<typename Out, typename In, typename SizeType>
    AMT_ALWAYS_INLINE void dot_prod_helper(
        Out* c,
        In const* a,
        In const* b,
        [[maybe_unused]] SizeType n,
        [[maybe_unused]] int max_threads
    ) noexcept
    {
        static_assert(std::is_same_v<Out,In>);
        
        using value_type = std::remove_pointer_t<Out>;
        [[maybe_unused]] constexpr auto size_in_bytes = sizeof(value_type);

        [[maybe_unused]] static auto const number_of_el_l1 = cache_manager::size(0) / size_in_bytes;
        [[maybe_unused]] static auto const number_of_el_l2 = cache_manager::size(1) / size_in_bytes;
        [[maybe_unused]] static auto const number_of_el_l3 = cache_manager::size(2) / size_in_bytes;
        [[maybe_unused]] static auto const section_one_block = (number_of_el_l1 << 1);
        [[maybe_unused]] static auto const section_two_block = (number_of_el_l1 >> 1u);
        [[maybe_unused]] static auto const section_three_block = (number_of_el_l2 >> 1u);

        constexpr auto simd_loop = [](In const* a, In const* b, SizeType const n){
            auto sum = value_type{};
            #pragma omp simd reduction(+:sum)
            for(auto i = 0ul; i < n; ++i){
                sum += (a[i] * b[i]);
            }
            return sum;
        };

        value_type sum {};

        if( n < section_one_block ){
            sum = simd_loop(a, b, n);
        }else if( n < number_of_el_l3 ){
            
            auto q = static_cast<int>(n / section_two_block);
            auto num_threads = std::max(1, std::min(max_threads, q));
            
            omp_set_num_threads(num_threads);

            #pragma omp parallel for schedule(static) reduction(+:sum)
            for(auto i = 0ul; i < n; i += section_two_block){
                auto ib = std::min(section_two_block, n - i);
                sum += simd_loop(a + i, b + i, ib);
            }

        }else{

            #pragma omp parallel reduction(+:sum)
            {
                for(auto i = 0ul; i < n; i += number_of_el_l2){
                    auto ib = std::min(number_of_el_l2, n - i);
                    auto ai = a + i;
                    auto bi = b + i;
                    #pragma omp for schedule(dynamic)
                    for(auto j = 0ul; j < ib; j += section_two_block){
                        auto jb = std::min(section_two_block, ib - j);
                        sum += simd_loop(ai + j, bi + j, jb);
                    }
                }
            }

        }
        *c = sum;
    }

    template<std::size_t N, typename Out, typename In>
    void static_simd_loop(
        Out* c,
        In const* a,
        In const* b
    ) noexcept
    {
        static_assert(std::is_same_v<Out,In>);
        
        using value_type = Out;
        value_type sum{0};
        if constexpr( N == 0ul ){
            sum = value_type{};
        }else if constexpr(N == 1ul){
            sum = a[0] * b[0];
        }else if constexpr(N == 2ul){
            sum = (a[0] * b[0])
                + (a[1] * b[1]);
        }else if constexpr(N == 3ul){
            sum = (a[0] * b[0])
                + (a[1] * b[1])
                + (a[2] * b[2]);
        }else if constexpr(N == 4ul){
            sum = (a[0] * b[0])
                + (a[1] * b[1])
                + (a[2] * b[2])
                + (a[3] * b[3]);
        }else if constexpr(N == 5ul){
            sum = (a[0] * b[0])
                + (a[1] * b[1])
                + (a[2] * b[2])
                + (a[3] * b[3])
                + (a[4] * b[4]);
        }else if constexpr(N == 6ul){
            sum = (a[0] * b[0])
                + (a[1] * b[1])
                + (a[2] * b[2])
                + (a[3] * b[3])
                + (a[4] * b[4])
                + (a[5] * b[5]);
        }else if constexpr(N == 7ul){
            sum = (a[0] * b[0])
                + (a[1] * b[1])
                + (a[2] * b[2])
                + (a[3] * b[3])
                + (a[4] * b[4])
                + (a[5] * b[5])
                + (a[7] * b[7]);
        }else{
            constexpr auto Nrem = N % 8ul;
            constexpr auto Nitr = N - Nrem;
            if constexpr(Nrem != 0ul){
                static_simd_loop<Nrem>(&sum,a,b);
                a += Nrem;
                b += Nrem;
            }

            if constexpr(Nitr != 0ul){
                #pragma omp simd
                for(auto i = 0ul; i < Nitr; ++i){
                    sum += (a[i] * b[i]);
                }
            }
        }
        *c = sum;
    }

    template<typename Out, typename E1, typename E2, typename... Timer>
    constexpr void dot_prod(
        Out& c,
        boost::numeric::ublas::tensor_core<E1> const& a,
        boost::numeric::ublas::tensor_core<E2> const& b,
        std::optional<std::size_t> num_threads,
        Timer&... t
    )
    {
        using tensor_type1 = boost::numeric::ublas::tensor_core<E1>;
        using tensor_type2 = boost::numeric::ublas::tensor_core<E2>;
        
        static_assert(
            std::is_same_v< typename tensor_type1::value_type, typename tensor_type2::value_type > && 
            std::is_same_v< Out, typename tensor_type2::value_type >,
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
                "amt::dot_prod(boost::numeric::ublas::tensor_core<E1> const&, boost::numeric::ublas::tensor_core<E2> const&) : "
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

        if( NA != NB ){
            throw std::runtime_error(
                "amt::dot_prod(boost::numeric::ublas::tensor_core<E1> const&, boost::numeric::ublas::tensor_core<E2> const&) : "
                "both vector must be of same size"
            );
        }
        auto const* aptr = a.data();
        auto const* bptr = b.data();
        if constexpr(sizeof...(Timer) > 0u) std::get<0>(timer).start();

        // switch(NA){
        //     case 1: static_simd_loop<1>(&c,aptr,bptr); break;
        //     case 2: static_simd_loop<2>(&c,aptr,bptr); break;
        //     case 3: static_simd_loop<3>(&c,aptr,bptr); break;
        //     case 4: static_simd_loop<4>(&c,aptr,bptr); break;
        //     case 5: static_simd_loop<5>(&c,aptr,bptr); break;
        //     case 6: static_simd_loop<6>(&c,aptr,bptr); break;
        //     case 7: static_simd_loop<7>(&c,aptr,bptr); break;
        //     case 8: static_simd_loop<8>(&c,aptr,bptr); break;
        //     case 9: static_simd_loop<9>(&c,aptr,bptr); break;
        //     case 10: static_simd_loop<10>(&c,aptr,bptr); break;
        //     case 11: static_simd_loop<11>(&c,aptr,bptr); break;
        //     case 12: static_simd_loop<12>(&c,aptr,bptr); break;
        //     case 13: static_simd_loop<13>(&c,aptr,bptr); break;
        //     case 14: static_simd_loop<14>(&c,aptr,bptr); break;
        //     case 15: static_simd_loop<15>(&c,aptr,bptr); break;
        //     case 16: static_simd_loop<16>(&c,aptr,bptr); break;
        //     case 17: static_simd_loop<17>(&c,aptr,bptr); break;
        //     case 18: static_simd_loop<18>(&c,aptr,bptr); break;
        //     case 19: static_simd_loop<19>(&c,aptr,bptr); break;
        //     case 20: static_simd_loop<20>(&c,aptr,bptr); break;
        //     case 21: static_simd_loop<21>(&c,aptr,bptr); break;
        //     case 22: static_simd_loop<22>(&c,aptr,bptr); break;
        //     case 23: static_simd_loop<23>(&c,aptr,bptr); break;
        //     case 24: static_simd_loop<24>(&c,aptr,bptr); break;
        //     case 25: static_simd_loop<25>(&c,aptr,bptr); break;
        //     case 26: static_simd_loop<26>(&c,aptr,bptr); break;
        //     case 27: static_simd_loop<27>(&c,aptr,bptr); break;
        //     case 28: static_simd_loop<28>(&c,aptr,bptr); break;
        //     case 29: static_simd_loop<29>(&c,aptr,bptr); break;
        //     case 30: static_simd_loop<30>(&c,aptr,bptr); break;
        //     case 31: static_simd_loop<31>(&c,aptr,bptr); break;
        //     case 32: static_simd_loop<32>(&c,aptr,bptr); break;
        //     default: dot_prod_helper(&c,aptr,bptr,NA,nths);
        // }

        dot_prod_helper(&c,aptr,bptr,NA,nths);

        if constexpr(sizeof...(Timer) > 0u) std::get<0>(timer).stop();

    }

} // namespace amt

#endif // AMT_BENCHMARK_DOT_PRODUCT_HPP
