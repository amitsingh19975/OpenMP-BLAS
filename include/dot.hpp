#if !defined(AMT_BENCHMARK_DOT_PRODUCT_HPP)
#define AMT_BENCHMARK_DOT_PRODUCT_HPP

#include <boost/numeric/ublas/tensor.hpp>
#include <omp.h>
#include <optional>
#include <cache_manager.hpp>
#include <cstdlib>

#define AMT_INLINE __attribute__((always_inline))
#define AMT_NO_OPT __attribute__ ((optnone))

namespace amt {

    template<typename Out, typename In, typename SizeType>
    AMT_NO_OPT void dot_prod_helper_ref(
        Out* c,
        In const* a,
        In const* b, 
        [[maybe_unused]] SizeType const n
    ) noexcept
    {
        static_assert(std::is_same_v<Out,In>);
        
        using value_type = std::remove_pointer_t<Out>;
        value_type sum = {0};
        auto ai = a;
        auto bi = b;
        for(auto i = 0ul; i < n; ++i){
            sum += *ai * *bi;
            ++ai;
            ++bi;
        }

        *c = sum;
    }

    // template<typename Out, typename In, typename SizeType>
    // void dot_prod_helper_diff_layout_ref(
    //     Out* c,
    //     In const* a,
    //     In const* b, 
    //     [[maybe_unused]] SizeType const wb,
    //     [[maybe_unused]] SizeType const n
    // ) noexcept
    // {
    //     static_assert(std::is_same_v<Out,In>);
        
    //     using value_type = std::remove_pointer_t<Out>;
    //     value_type sum = {0};

    //     for(auto i = 0ul; i < n; ++i){
    //         sum += (a[i] * b[i * wb]);
    //     }

    //     *c = sum;

    // }

    template<typename Out, typename In, typename SizeType>
    AMT_INLINE void dot_prod_helper(
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

        [[maybe_unused]] auto l1_size = cache_manager::size(0) / size_in_bytes;
        [[maybe_unused]] auto l2_size = cache_manager::size(1) / size_in_bytes;
        [[maybe_unused]] auto l3_size = cache_manager::size(2) / size_in_bytes;

        constexpr auto simd_loop = [](In const* a, In const* b, SizeType const n){
            auto sum = value_type{};
            constexpr auto alignement = alignof(In);
            #pragma omp simd reduction(+:sum) aligned(a,b:alignement)
            for(auto i = 0ul; i < n; ++i){
                sum += (a[i] * b[i]);
            }
            return sum;
        };

        value_type sum {};

        if( n < l1_size ){
            sum = simd_loop(a, b, n);
        }else if( n >= l1_size ){
            
            auto q = static_cast<int>(n / l1_size);
            auto num_threads = std::max(1, std::min(max_threads, q));
            #pragma omp parallel for schedule(static) reduction(+:sum) num_threads(num_threads)
            for(auto i = 0ul; i < n; i += l1_size){
                auto ib = std::min(l1_size, n - i);
                sum += simd_loop(a + i, b + i, ib);
            }

        }else{
            #pragma omp parallel num_threads(max_threads) reduction(+:sum)
            {
                for(auto i = 0ul; i < n; i += l2_size){
                    auto ib = std::min(l2_size, n - i);
                    auto ai = a + i;
                    auto bi = b + i;
                    #pragma omp for schedule(dynamic)
                    for(auto j = 0ul; j < ib; j += l1_size){
                        auto jb = std::min(l1_size, ib - j);
                        sum += simd_loop(ai + j, bi + j, jb);
                    }
                }
            }

        }
        *c = sum;
    }

    template<std::size_t N, typename Out, typename In>
    AMT_INLINE void static_dot_prod_helper(
        Out* c,
        In const* a,
        In const* b
    ) noexcept
    {
        static_assert(std::is_same_v<Out,In>);
        
        using value_type = std::remove_pointer_t<Out>;
        [[maybe_unused]] constexpr auto alignment = alignof(value_type);
        value_type sum = {0};
    
        #pragma omp simd safelen(N)  aligned(a,b:alignment)
        for(auto i = 0ul; i < N; ++i){
            sum += (a[i] * b[i]);
        }
        *c = sum;
    }

    // template<typename Out, typename In, typename SizeType>
    // void dot_prod_helper_diff_layout(
    //     Out* c,
    //     In const* a,
    //     In const* b, 
    //     [[maybe_unused]] SizeType const wb,
    //     [[maybe_unused]] SizeType const n
    // ) noexcept
    // {
    //     static_assert(std::is_same_v<Out,In>);
        
    //     using value_type = std::remove_pointer_t<Out>;
    //     [[maybe_unused]] constexpr auto alignment = alignof(value_type);
    //     value_type sum = {0};

    //     // #pragma omp target map(to: a[0:N], b[0:N]) map(tofrom: sum)
    //     // #pragma omp parallel
    //     {
    //         #pragma omp for simd safelen(8) aligned(a,b:alignment)
    //         for(auto i = 0ul; i < n; ++i){
    //             sum += (a[i] * b[i * wb]);
    //         }
    //     }

    //     *c = sum;

    // }

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
        auto nths = static_cast<int>(num_threads.value_or(omp_get_num_procs()));

        if( NA != NB ){
            throw std::runtime_error(
                "amt::dot_prod(boost::numeric::ublas::tensor_core<E1> const&, boost::numeric::ublas::tensor_core<E2> const&) : "
                "both vector must be of same size"
            );
        }

        if constexpr(sizeof...(Timer) > 0u) std::get<0>(timer).start();
        dot_prod_helper(
            &c,
            a.data(),
            b.data(),
            NA,
            nths
        );
        if constexpr(sizeof...(Timer) > 0u) std::get<0>(timer).stop();

    }

    template<typename Out, typename E1, typename E2, typename... Timer>
    constexpr void dot_prod_ref(
        Out& c,
        boost::numeric::ublas::tensor_core<E1> const& a,
        boost::numeric::ublas::tensor_core<E2> const& b,
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

        if( NA != NB ){
            throw std::runtime_error(
                "amt::dot_prod(boost::numeric::ublas::tensor_core<E1> const&, boost::numeric::ublas::tensor_core<E2> const&) : "
                "both vector must be of same size"
            );
        }

        if constexpr(sizeof...(Timer) > 0u){
            std::get<0>(timer).start();
            dot_prod_helper_ref(
                &c,
                a.data(),
                b.data(),
                NA
            );
            std::get<0>(timer).stop();
        }else{
            dot_prod_helper_ref(
                &c,
                a.data(),
                b.data(),
                NA
            );
        }
    }

} // namespace amt

#endif // AMT_BENCHMARK_DOT_PRODUCT_HPP
