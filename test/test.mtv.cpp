#include <catch2/catch.hpp>
#include <vector>
#include <algorithm>
#include <functional>
#include <boost/numeric/ublas/tensor.hpp>
#include <mtv.hpp>
#include <blis.h>
#include "test_utils.hpp"

template<typename T, typename L = boost::numeric::ublas::layout::first_order>
using tensor_t = boost::numeric::ublas::fixed_rank_tensor<T,2u,L>;
using shape_t = boost::numeric::ublas::extents<2>;


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

TEMPLATE_TEST_CASE( "Matrix Vector Product for Range[Start: 2, End: 512, Step: 1]", "[dot_prod]", float, double ) {
    namespace ub = boost::numeric::ublas;
    constexpr auto MinSize = 2ul;
    constexpr auto MaxSize = 512ul;
    constexpr auto Step = 1ul;

    for(auto sz = MinSize; sz < MaxSize; sz += Step){
        auto A = tensor_t<TestType>(shape_t{sz,sz});
        auto v = tensor_t<TestType>(shape_t{1,sz});

        rand_gen<TestType>(A);
        rand_gen<TestType>(v);

        auto lres = tensor_t<TestType>(shape_t{1,sz});
        auto rres = tensor_t<TestType>(shape_t{1,sz});

        auto inc = static_cast<inc_t>(1);
        auto M = static_cast<dim_t>(sz);
        auto N = static_cast<dim_t>(sz);
        auto alpha = TestType(1);
        auto rsa = static_cast<inc_t>(A.strides()[0]);
        auto csa = static_cast<inc_t>(A.strides()[1]);
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
        for(auto i = 0ul; i < sz; ++i, ++rptr, ++lptr){
            REQUIRE(Approx(*rptr) == *lptr);
        }
        
    }
    
}

TEMPLATE_TEST_CASE( "Matrix Vector Product for Size 32Kib", "[dot_prod]", float, double ) {
    namespace ub = boost::numeric::ublas;
    constexpr auto sz = 32 * 1024ul;
    auto A = tensor_t<TestType>(shape_t{sz,sz});
    auto v = tensor_t<TestType>(shape_t{1,sz});

    rand_gen<TestType>(A);
    rand_gen<TestType>(v);

    auto lres = tensor_t<TestType>(shape_t{1,sz});
    auto rres = tensor_t<TestType>(shape_t{1,sz});

    auto inc = static_cast<inc_t>(1);
    auto M = static_cast<dim_t>(sz);
    auto N = static_cast<dim_t>(sz);
    auto alpha = TestType(1);
    auto rsa = static_cast<inc_t>(A.strides()[0]);
    auto csa = static_cast<inc_t>(A.strides()[1]);
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
    for(auto i = 0ul; i < sz; ++i, ++rptr, ++lptr){
        REQUIRE(Approx(*rptr) == *lptr);
    }
    
}