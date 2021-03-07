#include <catch2/catch.hpp>
#include <vector>
#include <algorithm>
#include <functional>
#include <boost/numeric/ublas/tensor.hpp>
#include <outer.hpp>
#include <iostream>
#include <blis.h>


template<typename T, typename L = boost::numeric::ublas::layout::first_order>
using VectorType = boost::numeric::ublas::fixed_rank_tensor<T,2u,L>;

template<typename ValueType>
constexpr auto blis_ger(conj_t  conjx,
                        conj_t  conjy,
                        dim_t   m,
                        dim_t   n,
                        ValueType*  alpha,
                        ValueType*  x, inc_t incx,
                        ValueType*  y, inc_t incy,
                        ValueType*  a, inc_t rsa, inc_t csa
){
    if constexpr(std::is_same_v<ValueType,float>){
        bli_sger(conjx, conjy, m, n, alpha, x, incx, y, incy, a, rsa, csa);
    }else if constexpr(std::is_same_v<ValueType,double>){
        bli_dger(conjx, conjy, m, n, alpha, x, incx, y, incy, a, rsa, csa);
    }
};

TEMPLATE_TEST_CASE( "Square Vector Vector Outer Product for Range[Start: 2, End: 32, Step: 1]", "[outer_prod]", float, double ) {
    namespace ub = boost::numeric::ublas;

    constexpr std::size_t max_size = 1 << 5;
    std::vector<std::size_t> sizes(max_size);
    std::iota(sizes.begin(), sizes.end(), 2u);
    
    REQUIRE(sizes.size() == max_size);


    for(auto const& sz : sizes){
        VectorType<TestType> a(ub::extents<2>{1,sz});
        
        ub::dynamic_tensor<TestType> lres(ub::extents<>{sz, sz});
        ub::dynamic_tensor<TestType> rres(ub::extents<>{sz, sz});
        std::iota(a.begin(), a.end(), 1);
        
        auto inc = static_cast<inc_t>(1);
        auto M = static_cast<inc_t>(sz);
        auto N = static_cast<inc_t>(sz);
        auto alpha = TestType(1);
        auto rsa = static_cast<inc_t>(rres.strides()[0]);
        auto csa = static_cast<inc_t>(rres.strides()[1]);
        
        blis_ger(BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, M, N, &alpha, a.data(), inc, a.data(), inc, rres.data(), rsa, csa);
        // amt::blas::outer_prod<TestType>(amt::blas::ColMajor, M, N, TestType{1}, a.data(), inc, a.data(), inc, rres.data(), M);
        amt::outer_prod(lres, a, a, std::nullopt);

        for(auto i = 0ul; i < (sz * sz); ++i)
            REQUIRE(Approx(lres[i]) == rres[i]);
    }
    
}

TEMPLATE_TEST_CASE( "Rectangular Vector Vector Outer Product for Range[Start: 2, End: 32, Step: 1]", "[outer_prod]", float, double ) {
    namespace ub = boost::numeric::ublas;

    constexpr std::size_t max_size = 1 << 5;
    std::vector<std::size_t> sizes(max_size);
    std::iota(sizes.begin(), sizes.end(), 2u);
    
    REQUIRE(sizes.size() == max_size);


    for(auto const& m : sizes){
        VectorType<TestType> a(ub::extents<2>{1,m});
        std::iota(a.begin(), a.end(), 1);
        for(auto const& n : sizes){
            VectorType<TestType> b(ub::extents<2>{1,n});
            std::iota(b.begin(), b.end(), 1);
            
            ub::dynamic_tensor<TestType> lres(ub::extents<>{m, n});
            ub::dynamic_tensor<TestType> rres(ub::extents<>{m, n});
            
            auto inc = static_cast<inc_t>(1);
            auto M = static_cast<inc_t>(m);
            auto N = static_cast<inc_t>(n);
            auto alpha = TestType(1);
            auto aptr = a.data();
            auto bptr = b.data();
            auto cptr = rres.data();
            auto rsa = static_cast<inc_t>(rres.strides()[0]);
            auto csa = static_cast<inc_t>(rres.strides()[1]);
            
            blis_ger(BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, M, N, &alpha, aptr, inc, bptr, inc, cptr, rsa, csa);
            // amt::blas::outer_prod<TestType>(amt::blas::ColMajor, M, N, TestType{1}, a.data(), inc, b.data(), inc, rres.data(), M);
            amt::outer_prod(lres, a, b, std::nullopt);

            for(auto i = 0ul; i < (m * n); ++i)
                REQUIRE(Approx(lres[i]) == rres[i]);
        }
    }

}
