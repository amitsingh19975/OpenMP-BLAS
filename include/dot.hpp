#if !defined(AMT_BENCHMARK_DOT_PRODUCT_HPP)
#define AMT_BENCHMARK_DOT_PRODUCT_HPP

#include <boost/numeric/ublas/tensor.hpp>
#include <omp.h>
#include <optional>
#include <cache_manager.hpp>
#include <cstdlib>
#include <macros.hpp>

namespace amt {

    template<bool Serial = false, typename Out, typename In, typename SizeType>
    AMT_ALWAYS_INLINE void dot_prod_helper_tuning_param(
        Out* c,
        In const* a,
        In const* b,
        [[maybe_unused]] SizeType n,
        [[maybe_unused]] int max_threads,
        [[maybe_unused]] SizeType const block1,
        [[maybe_unused]] SizeType const block2,
        [[maybe_unused]] SizeType const block3
    ) noexcept
    {
        static_assert(std::is_same_v<Out,In>);
        
        using value_type = std::remove_pointer_t<Out>;
        [[maybe_unused]] constexpr auto size_in_bytes = sizeof(value_type);

        [[maybe_unused]] static auto const number_of_el_l1 = cache_manager::size(0) / size_in_bytes;
        [[maybe_unused]] static auto const number_of_el_l2 = cache_manager::size(1) / size_in_bytes;
        [[maybe_unused]] static auto const number_of_el_l3 = cache_manager::size(2) / size_in_bytes;

        constexpr auto simd_loop = [](In const* a, In const* b, SizeType const n){
            auto sum = value_type{};
            #pragma omp simd reduction(+:sum)
            for(auto i = 0ul; i < n; ++i){
                sum += (a[i] * b[i]);
            }
            return sum;
        };

        value_type sum {};

        if( n < block1 ){
            sum = simd_loop(a, b, n);
        }else if( n < number_of_el_l3 ){


            auto q = static_cast<int>(n / block2);
            auto num_threads = std::max(1, std::min(max_threads, q));
            
            omp_set_num_threads(num_threads);

            #pragma omp parallel for schedule(static) reduction(+:sum)
            for(auto i = 0ul; i < n; i += block2){
                auto ib = std::min(block2, n - i);
                sum += simd_loop(a + i, b + i, ib);
            }

        }else{

            #pragma omp parallel reduction(+:sum)
            {
                for(auto i = 0ul; i < n; i += block3){
                    auto ib = std::min(block3, n - i);
                    auto ai = a + i;
                    auto bi = b + i;
                    #pragma omp for schedule(dynamic)
                    for(auto j = 0ul; j < ib; j += block2){
                        auto jb = std::min(block2, ib - j);
                        sum += simd_loop(ai + j, bi + j, jb);
                    }
                }
            }

        }
        *c = sum;
    }

    template<typename Out, typename E1, typename E2>
    constexpr auto dot_prod_tuning_param(
        Out& c,
        boost::numeric::ublas::tensor_core<E1> const& a,
        boost::numeric::ublas::tensor_core<E2> const& b,
        std::size_t const b1,
        std::size_t const b2,
        std::size_t const b3,
        std::optional<std::size_t> num_threads
    )
    {
        using tensor_type1 = boost::numeric::ublas::tensor_core<E1>;
        using tensor_type2 = boost::numeric::ublas::tensor_core<E2>;
        
        static_assert(
            std::is_same_v< typename tensor_type1::value_type, typename tensor_type2::value_type > && 
            std::is_same_v< Out, typename tensor_type2::value_type >,
            "both tensor type and result type must be of same value_type"
        );

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

        return [&c,aptr,bptr,NA,nths,b1,b2,b3]{
            dot_prod_helper_tuning_param(&c,aptr,bptr,NA,nths,b1,b2,b3);
        };

    }

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
                for(auto i = 0ul; i < n; i += section_three_block){
                    auto ib = std::min(section_three_block, n - i);
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

    template<bool TuningFlag = false, typename Out, typename E1, typename E2>
    constexpr auto dot_prod(
        Out& c,
        boost::numeric::ublas::tensor_core<E1> const& a,
        boost::numeric::ublas::tensor_core<E2> const& b,
        std::optional<std::size_t> num_threads
    )
    {
        using tensor_type1 = boost::numeric::ublas::tensor_core<E1>;
        using tensor_type2 = boost::numeric::ublas::tensor_core<E2>;
        
        static_assert(
            std::is_same_v< typename tensor_type1::value_type, typename tensor_type2::value_type > && 
            std::is_same_v< Out, typename tensor_type2::value_type >,
            "both tensor type and result type must be of same value_type"
        );

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
    
        return [&c,aptr,bptr,NA,nths]{
            dot_prod_helper(&c,aptr,bptr,NA,nths);
        };

    }

} // namespace amt

#endif // AMT_BENCHMARK_DOT_PRODUCT_HPP
