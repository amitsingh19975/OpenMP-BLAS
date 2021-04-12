#include <catch2/catch.hpp>
#include <vector>
#include <algorithm>
#include <functional>
#include <boost/numeric/ublas/tensor.hpp>
#include <mtm.hpp>
#include <blis.h>
#include "test_utils.hpp"

template<typename T, typename L = boost::numeric::ublas::layout::first_order>
using tensor_t = boost::numeric::ublas::fixed_rank_tensor<T,2u,L>;
using shape_t = boost::numeric::ublas::extents<2>;


template<typename ValueType>
constexpr auto blis_gemv(
       trans_t transa,
       trans_t transb,
       dim_t   m,
       dim_t   n,
       dim_t   k,
       ValueType*  alpha,
       ValueType*  a, inc_t rsa, inc_t csa,
       ValueType*  b, inc_t rsb, inc_t csb,
       ValueType*  beta,
       ValueType*  c, inc_t rsc, inc_t csc
){
    if constexpr(std::is_same_v<ValueType,float>){
        bli_sgemm(transa,transb,m,n,k,alpha,a,rsa,csa,b,rsb,csb,beta,c,rsc,csc);
    }else if constexpr(std::is_same_v<ValueType,double>){
        bli_dgemm(transa,transb,m,n,k,alpha,a,rsa,csa,b,rsb,csb,beta,c,rsc,csc);
    }
};

TEMPLATE_TEST_CASE( "(FFF) Matrix Matrix Product for Range[Start: 2, End: 32, Step: 1]", "[fff_mtm]", float, double ) {
    namespace ub = boost::numeric::ublas;
    constexpr auto MinSize = 2ul;
    constexpr auto MaxSize = 32ul;
    constexpr auto Step = 1ul;

    for(auto sz = MinSize; sz < MaxSize; sz += Step){
        auto A = tensor_t<TestType>(shape_t{sz,sz});
        auto B = tensor_t<TestType>(shape_t{sz,sz});

        rand_gen<TestType>(A);
        rand_gen<TestType>(B);

        auto lres = tensor_t<TestType>(shape_t{sz,sz});
        auto rres = tensor_t<TestType>(shape_t{sz,sz});

        auto alpha = TestType(1);
        auto M = static_cast<dim_t>(A.size(0));
        auto N = static_cast<dim_t>(B.size(1));
        auto K = static_cast<dim_t>(A.size(1));
        auto* aptr = A.data();
        auto* bptr = B.data();
        auto* cptr = lres.data();
        auto rsa = static_cast<inc_t>(A.strides()[0]);
        auto csa = static_cast<inc_t>(A.strides()[1]);
        auto rsb = static_cast<inc_t>(B.strides()[0]);
        auto csb = static_cast<inc_t>(B.strides()[1]);
        auto rsc = static_cast<inc_t>(lres.strides()[0]);
        auto csc = static_cast<inc_t>(lres.strides()[1]);

        blis_gemv(
            BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, 
            M, N, K,
            &alpha, 
            aptr, rsa, csa,
            bptr, rsb, csb,
            &alpha,
            cptr, rsc, csc
        );

        amt::mtm(rres,A,B,std::nullopt)();

        auto rptr = rres.data();
        auto lptr = cptr;

        
        for(auto i = 0ul; i < rres.size(); ++i, ++rptr, ++lptr){
            REQUIRE(Approx(*rptr) == *lptr);
        }
        
    }
    
}

TEMPLATE_TEST_CASE( "(LFF) Matrix Matrix Product for Range[Start: 2, End: 32, Step: 1]", "[lff_mtm]", float, double ) {
    namespace ub = boost::numeric::ublas;
    constexpr auto MinSize = 2ul;
    constexpr auto MaxSize = 32ul;
    constexpr auto Step = 1ul;

    for(auto sz = MinSize; sz < MaxSize; sz += Step){
        auto A = tensor_t<TestType>(shape_t{sz,sz});
        auto B = tensor_t<TestType>(shape_t{sz,sz});

        rand_gen<TestType>(A);
        rand_gen<TestType>(B);

        auto lres = tensor_t<TestType,boost::numeric::ublas::layout::last_order>(shape_t{sz,sz});
        auto rres = tensor_t<TestType,boost::numeric::ublas::layout::last_order>(shape_t{sz,sz});

        auto alpha = TestType(1);
        auto M = static_cast<dim_t>(A.size(0));
        auto N = static_cast<dim_t>(B.size(1));
        auto K = static_cast<dim_t>(A.size(1));
        auto* aptr = A.data();
        auto* bptr = B.data();
        auto* cptr = lres.data();
        auto rsa = static_cast<inc_t>(A.strides()[0]);
        auto csa = static_cast<inc_t>(A.strides()[1]);
        auto rsb = static_cast<inc_t>(B.strides()[0]);
        auto csb = static_cast<inc_t>(B.strides()[1]);
        auto rsc = static_cast<inc_t>(lres.strides()[0]);
        auto csc = static_cast<inc_t>(lres.strides()[1]);

        blis_gemv(
            BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, 
            M, N, K,
            &alpha, 
            aptr, rsa, csa,
            bptr, rsb, csb,
            &alpha,
            cptr, rsc, csc
        );

        amt::mtm(rres,A,B,std::nullopt)();

        auto rptr = rres.data();
        auto lptr = cptr;

        
        for(auto i = 0ul; i < rres.size(); ++i, ++rptr, ++lptr){
            REQUIRE(Approx(*rptr) == *lptr);
        }
        
    }
    
}
