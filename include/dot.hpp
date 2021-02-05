#if !defined(AMT_BENCHMARK_DOT_PRODUCT_HPP)
#define AMT_BENCHMARK_DOT_PRODUCT_HPP

#include <boost/numeric/ublas/tensor.hpp>
#include <omp.h>
#include <optional>
#include <x86intrin.h>

namespace amt {

    template<typename Out, typename In, typename SizeType>
    void dot_prod_helper_ref(
        Out* c,
        In const* a,
        In const* b, 
        [[maybe_unused]] SizeType const n
    ) noexcept
    {
        static_assert(std::is_same_v<Out,In>);
        
        using value_type = std::remove_pointer_t<Out>;
        value_type sum = {0};

        for(auto i = 0ul; i < n; ++i){
            sum += (a[i] * b[i]);
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
    void dot_prod_helper(
        Out* c,
        In const* a,
        In const* b, 
        [[maybe_unused]] SizeType const n
    ) noexcept
    {
        static_assert(std::is_same_v<Out,In>);
        
        using value_type = std::remove_pointer_t<Out>;
        [[maybe_unused]] constexpr auto block = (256ul / (sizeof(value_type) * 8ul)) * 16ul;
        [[maybe_unused]] constexpr auto alignment = alignof(value_type);
        value_type sum = {0};
    
        #pragma omp simd reduction(+:sum) safelen(16) aligned(a,b:alignment)
        for(auto i = 0ul; i < n; ++i){
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

    template<typename Out, typename E1, typename E2>
    constexpr void dot_prod(
        Out& c,
        boost::numeric::ublas::tensor_core<E1> const& a,
        boost::numeric::ublas::tensor_core<E2> const& b
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

        if( NA != NB ){
            throw std::runtime_error(
                "amt::dot_prod(boost::numeric::ublas::tensor_core<E1> const&, boost::numeric::ublas::tensor_core<E2> const&) : "
                "both vector must be of same size"
            );
        }

        dot_prod_helper(
            &c,
            a.data(),
            b.data(),
            NA
        );
    }

    template<typename Out, typename E1, typename E2>
    constexpr void dot_prod_ref(
        Out& c,
        boost::numeric::ublas::tensor_core<E1> const& a,
        boost::numeric::ublas::tensor_core<E2> const& b
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

        if( NA != NB ){
            throw std::runtime_error(
                "amt::dot_prod(boost::numeric::ublas::tensor_core<E1> const&, boost::numeric::ublas::tensor_core<E2> const&) : "
                "both vector must be of same size"
            );
        }

        dot_prod_helper_ref(
            &c,
            a.data(),
            b.data(),
            NA
        );
    }

} // namespace amt

#endif // AMT_BENCHMARK_DOT_PRODUCT_HPP
