#include <catch2/catch.hpp>
#include <vector>
#include <algorithm>
#include <functional>
#include <boost/numeric/ublas/tensor.hpp>
#include <mtv.hpp>
#include <blis.h>
#include "test_utils.hpp"

template<typename ValueType>
constexpr auto blis_gemv(
        trans_t transa,
        conj_t  conjx,
        dim_t   m,
        dim_t   n,
        ValueType*  alpha,
        ValueType*  a, inc_t rsa, inc_t csa,
        ValueType*  x, inc_t incx,
        ValueType*  beta,
        ValueType*  y, inc_t incy
){
    if constexpr(std::is_same_v<ValueType,float>){
        bli_sgemv(transa, conjx, m, n, alpha, a, rsa, csa, x, incx, beta, y, incy);
    }else if constexpr(std::is_same_v<ValueType,double>){
        bli_dgemv(transa, conjx, m, n, alpha, a, rsa, csa, x, incx, beta, y, incy);
    }
};

TEMPLATE_TEST_CASE( "First Order Matrix Vector Product for Range[Start: 2, End: 512, Step: 1]", "[first_order_mtv_range]", float, double ) {
    namespace ub = boost::numeric::ublas;
    constexpr auto MinSize = 2ul;
    constexpr auto MaxSize = 512ul;
    constexpr auto Step = 1ul;

    for(auto sz = MinSize; sz < MaxSize; sz += Step){
        auto A = amt::make_tensor<TestType>(sz,sz);
        auto v = amt::make_tensor<TestType>(1,sz);

        rand_gen<TestType>(A);
        rand_gen<TestType>(v);

        auto lres = amt::make_tensor<TestType>(1,sz);
        auto rres = amt::make_tensor<TestType>(1,sz);

        auto inc = static_cast<inc_t>(1);
        auto alpha = TestType(1);
        auto M = static_cast<dim_t>(A.size(0));
        auto N = static_cast<dim_t>(A.size(1));
        auto rsa = static_cast<inc_t>(1);
        auto csa = static_cast<inc_t>(M);
        auto aptr = A.data();
        auto bptr = v.data();
        auto cptr = lres.data();

        blis_gemv( 
            BLIS_NO_TRANSPOSE, BLIS_NO_CONJUGATE, 
            M, N, 
            &alpha, 
            aptr, rsa, csa,
            bptr, inc,
            &alpha,
            cptr, inc
        );

        amt::mtv(rres,A,v,std::nullopt)();

        auto rptr = rres.data();
        auto lptr = cptr;
        
        for(auto i = 0ul; i < rres.size(); ++i, ++rptr, ++lptr){
            REQUIRE(Approx(*rptr) == *lptr);
        }
        
    }
    
}

TEMPLATE_TEST_CASE( "Last Order Matrix Vector Product for Range[Start: 2, End: 512, Step: 1]", "[last_order_mtv_range]", float, double ) {
    namespace ub = boost::numeric::ublas;
    using layout_type = ub::layout::last_order;

    constexpr auto MinSize = 2ul;
    constexpr auto MaxSize = 512ul;
    constexpr auto Step = 1ul;

    for(auto sz = MinSize; sz < MaxSize; sz += Step){
        auto A = amt::make_tensor<TestType, layout_type>(sz,sz);
        auto v = amt::make_tensor<TestType>(1,sz);

        rand_gen<TestType>(A);
        rand_gen<TestType>(v);

        auto lres = amt::make_tensor<TestType>(1,sz);
        auto rres = amt::make_tensor<TestType>(1,sz);

        auto alpha = TestType(1);
        auto inc = static_cast<inc_t>(1);
        auto M = static_cast<dim_t>(A.size(0));
        auto N = static_cast<dim_t>(A.size(1));
        auto rsa = static_cast<inc_t>(N);
        auto csa = static_cast<inc_t>(1);
        auto aptr = A.data();
        auto bptr = v.data();
        auto cptr = lres.data();

        blis_gemv( 
            BLIS_NO_TRANSPOSE, BLIS_NO_CONJUGATE, 
            M, N, 
            &alpha, 
            aptr, rsa, csa,
            bptr, inc,
            &alpha,
            cptr, inc
        );

        amt::mtv(rres,A,v,std::nullopt)();

        auto rptr = rres.data();
        auto lptr = cptr;
        for(auto i = 0ul; i < rres.size(); ++i, ++rptr, ++lptr){
            REQUIRE(Approx(*rptr) == *lptr);
        }
        
    }
    
}

TEMPLATE_TEST_CASE( "First Order Matrix Vector Product for Size 32Kib", "[first_order_mtv_large]", float, double ) {
    namespace ub = boost::numeric::ublas;
    constexpr auto sz = 32 * 1024ul;
    auto A = amt::make_tensor<TestType>(sz,sz);
    auto v = amt::make_tensor<TestType>(1,sz);

    rand_gen<TestType>(A);
    rand_gen<TestType>(v);

    auto lres = amt::make_tensor<TestType>(1,sz);
    auto rres = amt::make_tensor<TestType>(1,sz);

    auto alpha = TestType(1);
    auto inc = static_cast<inc_t>(1);
    auto M = static_cast<dim_t>(A.size(0));
    auto N = static_cast<dim_t>(A.size(1));
    auto rsa = static_cast<inc_t>(1);
    auto csa = static_cast<inc_t>(M);
    auto aptr = A.data();
    auto bptr = v.data();
    auto cptr = lres.data();

    blis_gemv( 
        BLIS_NO_TRANSPOSE, BLIS_NO_CONJUGATE, 
        M, N, 
        &alpha, 
        aptr, rsa, csa,
        bptr, inc,
        &alpha,
        cptr, inc
    );

    amt::mtv(rres,A,v,std::nullopt)();

    auto rptr = rres.data();
    auto lptr = cptr;
    for(auto i = 0ul; i < rres.size(); ++i, ++rptr, ++lptr){
        REQUIRE(Approx(*rptr) == *lptr);
    }
    
}

TEMPLATE_TEST_CASE( "Last Order Matrix Vector Product for Size 32Kib", "[last_order_mtv_large]", float, double ) {
    namespace ub = boost::numeric::ublas;
    using layout_type = ub::layout::last_order;

    constexpr auto sz = 32 * 1024ul;
    auto A = amt::make_tensor<TestType,layout_type>(sz,sz);
    auto v = amt::make_tensor<TestType>(1,sz);

    rand_gen<TestType>(A);
    rand_gen<TestType>(v);

    auto lres = amt::make_tensor<TestType>(1,sz);
    auto rres = amt::make_tensor<TestType>(1,sz);

    auto alpha = TestType(1);
    auto inc = static_cast<inc_t>(1);
    auto M = static_cast<dim_t>(A.size(0));
    auto N = static_cast<dim_t>(A.size(1));
    auto rsa = static_cast<inc_t>(N);
    auto csa = static_cast<inc_t>(1);
    auto aptr = A.data();
    auto bptr = v.data();
    auto cptr = lres.data();

    blis_gemv( 
        BLIS_NO_TRANSPOSE, BLIS_NO_CONJUGATE, 
        M, N, 
        &alpha, 
        aptr, rsa, csa,
        bptr, inc,
        &alpha,
        cptr, inc
    );

    amt::mtv(rres,A,v,std::nullopt)();

    auto rptr = rres.data();
    auto lptr = cptr;
    for(auto i = 0ul; i < rres.size(); ++i, ++rptr, ++lptr){
        REQUIRE(Approx(*rptr) == *lptr);
    }
    
}