#if !defined(AMT_BENCHMARK_DOT_PRODUCT_HPP)
#define AMT_BENCHMARK_DOT_PRODUCT_HPP

#include <boost/numeric/ublas/tensor.hpp>
#include <omp.h>
#include <optional>

namespace amt {

    template<typename Out, typename In, typename SizeType>
    void dot_prod_helper_same_layout(
        Out* c,
        In const* a, [[maybe_unused]] SizeType const* wa, [[maybe_unused]] SizeType const* na,
        In const* b, [[maybe_unused]] SizeType const* wb, [[maybe_unused]] SizeType const* nb,
        [[maybe_unused]] std::size_t num_of_threads
    ) noexcept
    {
        static_assert(std::is_same_v<Out,In>);
        
        auto N = std::max(na[0], na[1]);
        using value_type = std::remove_pointer_t<Out>;
        constexpr auto alignment = alignof(value_type);
        value_type sum{};

        // omp_set_num_threads(static_cast<int>(num_of_threads));
        
        #pragma omp simd reduction(+:sum) safelen(16) aligned(a,b:alignment)
        for(auto i = 0ul; i < N; ++i){
            sum += a[i] * b[i];
        }
        
        *c = sum;
    }

    template<typename Out, typename E1, typename E2>
    constexpr void dot_prod(
        Out& c,
        boost::numeric::ublas::tensor_core<E1> const& a,
        boost::numeric::ublas::tensor_core<E2> const& b,
        std::optional<std::size_t> num_of_threads = {std::nullopt}
    )
    {
        using tensor_type1 = boost::numeric::ublas::tensor_core<E1>;
        using tensor_type2 = boost::numeric::ublas::tensor_core<E2>;
        using layout_type1 = typename tensor_type1::layout_type;
        using layout_type2 = typename tensor_type2::layout_type;

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

        std::size_t threads = num_of_threads.value_or( static_cast<std::size_t>(omp_get_num_procs()) );


        if constexpr( std::is_same_v<layout_type1,layout_type2> ){

            if( !( (na[0] == nb[0]) && (na[1] == nb[1]) ) ){
                throw std::runtime_error(
                    "amt::dot_prod(boost::numeric::ublas::tensor_core<E1> const&, boost::numeric::ublas::tensor_core<E2> const&) : "
                    "both vector must be of same size"
                );
            }

            dot_prod_helper_same_layout(
                &c,
                a.data(), a.strides().data(), a.extents().data(),
                b.data(), b.strides().data(), b.extents().data(),
                threads
            );
        }
    }

} // namespace amt

#endif // AMT_BENCHMARK_DOT_PRODUCT_HPP
