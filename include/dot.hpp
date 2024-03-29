#if !defined(AMT_BENCHMARK_DOT_PRODUCT_HPP)
#define AMT_BENCHMARK_DOT_PRODUCT_HPP

#include <boost/numeric/ublas/tensor.hpp>
#include <optional>
#include <cache_manager.hpp>
#include <macros.hpp>
#include <thread_utils.hpp>
#include <simd_loop.hpp>

namespace amt {

    namespace dot_impl
    {

        template<typename ValueType, typename SizeType, typename SIMDLoop>
        AMT_ALWAYS_INLINE auto section_two_loop(ValueType const* a, ValueType const* b, 
            SizeType const n, SizeType const block, SIMDLoop const& simd_loop
        ) noexcept{
            static_assert(SIMDLoop::type == impl::SIMD_PROD_TYPE::INNER);

            ValueType sum{};
            #pragma omp parallel for schedule(static) reduction(+:sum)
            for(auto i = 0ul; i < n; i += block){
                auto ib = std::min(block, n - i);
                sum += simd_loop(a + i, b + i, ib);
            }
            return sum;
        };

        template<typename ValueType, typename SizeType, typename SIMDLoop>
        AMT_ALWAYS_INLINE auto section_three_loop(ValueType const* a, ValueType const* b, 
            SizeType const n, SizeType const block2, SizeType const block3, 
            SIMDLoop const& simd_loop
        ) noexcept{
            static_assert(SIMDLoop::type == impl::SIMD_PROD_TYPE::INNER);
            
            ValueType sum{};
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
            return sum;
        };

    } // namespace dot_impl
    


    template<typename ValueType, typename SizeType>
    AMT_ALWAYS_INLINE void dot_prod_helper_tuning_param(
        ValueType* c,
        ValueType const* a,
        ValueType const* b,
        [[maybe_unused]] SizeType n,
        [[maybe_unused]] int max_threads,
        [[maybe_unused]] SizeType const block1,
        [[maybe_unused]] SizeType const block2,
        [[maybe_unused]] SizeType const block3
    ) noexcept
    {   
        [[maybe_unused]] constexpr auto size_in_bytes = sizeof(ValueType);
        constexpr auto simd_type = impl::SIMD_PROD_TYPE::INNER;
        constexpr auto simd_loop = impl::simd_loop<simd_type>{};

        [[maybe_unused]] static auto const number_of_el_l1 = cache_manager::size(0) / size_in_bytes;
        [[maybe_unused]] static auto const number_of_el_l2 = cache_manager::size(1) / size_in_bytes;
        [[maybe_unused]] static auto const number_of_el_l3 = cache_manager::size(2) / size_in_bytes;

        if( n < block1 ){
            *c = simd_loop(a, b, n);
        }else if( n < number_of_el_l3 ){
            auto q = static_cast<int>(n / block2);
            auto num_threads = std::max(1, std::min(max_threads, q));
            threads::set_num_threads(num_threads);

            *c = dot_impl::section_two_loop(a,b,n,block2,simd_loop);
        }else{
            *c = dot_impl::section_three_loop(a,b,n,block2,block3,simd_loop);
        }
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

        threads::set_num_threads(num_threads);
        auto nths = threads::get_num_threads();

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

    template<typename ValueType, typename SizeType>
    AMT_ALWAYS_INLINE void dot_prod_helper(
        ValueType* c,
        ValueType const* a, [[maybe_unused]] SizeType const* na,
        ValueType const* b, [[maybe_unused]] SizeType const* nb,
        [[maybe_unused]] int max_threads
    ) noexcept
    {
        [[maybe_unused]] constexpr auto size_in_bytes = sizeof(ValueType);
        constexpr auto simd_type = impl::SIMD_PROD_TYPE::INNER;
        constexpr auto simd_loop = impl::simd_loop<simd_type>{};

        [[maybe_unused]] static auto const number_of_el_l1 = cache_manager::size(0) / size_in_bytes;
        [[maybe_unused]] static auto const number_of_el_l2 = cache_manager::size(1) / size_in_bytes;
        [[maybe_unused]] static auto const number_of_el_l3 = cache_manager::size(2) / size_in_bytes;
        [[maybe_unused]] static auto const section_one_block = (number_of_el_l1 << 1);
        [[maybe_unused]] static auto const section_two_block = (number_of_el_l1 >> 1u);
        [[maybe_unused]] static auto const section_three_block = (number_of_el_l2 >> 1u);
        auto const n = na[0] * na[1];

        if( n < section_one_block ){
            *c = simd_loop(a, b, n);
        }else if( n < number_of_el_l3 ){
            auto q = static_cast<int>(n / section_two_block);
            auto num_threads = std::max(1, std::min(max_threads, q));
            threads::set_num_threads(num_threads);

            *c = dot_impl::section_two_loop(a,b,n,section_two_block,simd_loop);
        }else{
            *c = dot_impl::section_three_loop(a,b,n,section_two_block,section_three_block,simd_loop);
        }
    }

    template<typename Out, typename E1, typename E2>
    constexpr auto dot_prod(
        Out& c,
        boost::numeric::ublas::tensor_core<E1> const& a,
        boost::numeric::ublas::tensor_core<E2> const& b,
        std::optional<std::size_t> num_threads = std::nullopt
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

        threads::set_num_threads(num_threads);
        auto nths = threads::get_num_threads();

        if( NA != NB ){
            throw std::runtime_error(
                "amt::dot_prod(boost::numeric::ublas::tensor_core<E1> const&, boost::numeric::ublas::tensor_core<E2> const&) : "
                "both vector must be of same size"
            );
        }
        auto const* aptr = a.data();
        auto const* bptr = b.data();
        auto const* na_ptr = na.data();
        auto const* nb_ptr = nb.data();
    
        return [&c,aptr,bptr,na_ptr,nb_ptr,nths]{
            dot_prod_helper(&c,aptr,na_ptr,bptr,nb_ptr,nths);
        };

    }

} // namespace amt

#endif // AMT_BENCHMARK_DOT_PRODUCT_HPP
