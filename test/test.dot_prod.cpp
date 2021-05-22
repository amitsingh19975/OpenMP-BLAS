#include <catch2/catch.hpp>
#include <vector>
#include <algorithm>
#include <functional>
#include <boost/numeric/ublas/tensor.hpp>
#include <dot.hpp>
#include <openblas.hpp>


TEMPLATE_TEST_CASE( "Vector Vector Inner Product for Range[Start: 2, End: 2^17, Step: 1]", "[dot_prod]", float, double ) {
    namespace ub = boost::numeric::ublas;

    constexpr std::size_t max_size = 1 << 17;
    std::vector<std::size_t> sizes(max_size);
    std::iota(sizes.begin(), sizes.end(), 2u);
    
    REQUIRE(sizes.size() == max_size);


    for(auto const& s : sizes){
        auto a = amt::make_tensor<TestType>(1ul,s);
        std::iota(a.begin(), a.end(), 1);
        TestType my_res{};
        auto openblas_res = amt::blas::dot_prod<TestType>(static_cast<blasint>(s), a.data(), 1, a.data(), 1);
        amt::dot_prod(my_res, a, a, std::nullopt)();
        REQUIRE(Approx(my_res) == openblas_res);
    }
    
}