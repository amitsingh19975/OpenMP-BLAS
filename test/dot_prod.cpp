#include <catch2/catch.hpp>
#include <vector>
#include <algorithm>
#include <functional>
#include <boost/numeric/ublas/tensor.hpp>
#include <dot.hpp>
#include <openblas.hpp>

template<typename T, typename L = boost::numeric::ublas::layout::first_order>
using VectorType = boost::numeric::ublas::fixed_rank_tensor<T,2u,L>;

TEMPLATE_TEST_CASE( "Vector Vector Inner Product for Range[Start: 2, End: 2^17, Step: 1]", "[dot_prod]", float, double ) {
    namespace ub = boost::numeric::ublas;
    amt::OpenBlasFnLoader::init();

    constexpr std::size_t total_size = 1 << 17;
    std::vector<std::size_t> sizes(total_size);
    std::iota(sizes.begin(), sizes.end(), 2u);
    
    REQUIRE(sizes.size() == total_size);

    for(auto const& s : sizes){
        VectorType<TestType> a(ub::extents<2>{1,s});
        std::iota(a.begin(), a.end(), 1);
        auto b = a;
        TestType my_res{};
        auto openblas_res = amt::blas::dot_prod<TestType>(static_cast<blasint>(s), a.data(), 1, b.data(), 1);
        amt::dot_prod(my_res, a, b, std::nullopt);
        REQUIRE(Approx(my_res) == openblas_res);
    }
    
}