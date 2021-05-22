#include <catch2/catch.hpp>
#include <vector>
#include <algorithm>
#include <functional>
#include <boost/numeric/ublas/tensor.hpp>
#include <trans.hpp>
#include <blis.h>
#include "test_utils.hpp"
#include <Eigen/Dense>

TEMPLATE_TEST_CASE( "Out-Of-Place Matrix Transpose for Range[Start: 2, End: 32, Step: 1]", "[outplace_transpose]", float, double ) {
    namespace ub = boost::numeric::ublas;
    using namespace Eigen;

    constexpr auto MinSize = 2ul;
    constexpr auto MaxSize = 32ul;
    constexpr auto Step = 1ul;
    using eigen_matrix = Map<Matrix<TestType,-1,-1>>;


    for(auto sz = MinSize; sz < MaxSize; sz += Step){
        auto tA = amt::make_tensor<TestType>(sz,sz);
        auto tres = amt::make_tensor<TestType>(sz,sz);

        rand_gen<TestType>(tA);
        
        auto temp = tA;
        eigen_matrix A(temp.data(), static_cast<Index>(sz), static_cast<Index>(sz));

        auto eres = A.transpose();

        amt::transpose(tres,tA,std::nullopt)();
        
        for(auto i = 0ul; i < sz; ++i){
            for(auto j = 0ul; j < sz; ++j){
                REQUIRE(Approx(tres(i,j)) == eres(static_cast<Index>(i),static_cast<Index>(j)));
            }
        }
        
    }
    
}

TEMPLATE_TEST_CASE( "In-Place Matrix Transpose for Range[Start: 2, End: 32, Step: 1]", "[inplace_transpose]", float, double ) {
    namespace ub = boost::numeric::ublas;
    using namespace Eigen;

    constexpr auto MinSize = 2ul;
    constexpr auto MaxSize = 32ul;
    constexpr auto Step = 1ul;
    using eigen_matrix = Map<Matrix<TestType,-1,-1>>;


    for(auto sz = MinSize; sz < MaxSize; sz += Step){
        auto tA = amt::make_tensor<TestType>(sz,sz);

        rand_gen<TestType>(tA);
        
        auto temp = tA;
        eigen_matrix A(temp.data(), static_cast<Index>(sz), static_cast<Index>(sz));

        auto eres = A.transpose();

        amt::transpose(tA,std::nullopt)();
        
        for(auto i = 0ul; i < sz; ++i){
            for(auto j = 0ul; j < sz; ++j){
                REQUIRE(Approx(tA(i,j)) == eres(static_cast<Index>(i),static_cast<Index>(j)));
            }
        }
        
    }
    
}
