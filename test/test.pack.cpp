#include <catch2/catch.hpp>
#include <vector>
#include <algorithm>
#include <functional>
#include <boost/numeric/ublas/tensor.hpp>
#include <utils.hpp>

TEST_CASE( "Pack [3,3] ", "[pack_mtm3x3]" ) {
    constexpr auto MR = 2ul;
    
    // in = 1 4 7
    //      2 5 8
    //      3 6 9
    
    // out =1 5 3
    //      2 7 6
    //      4 8 9

    auto in = amt::make_tensor<float>(3,3);
    std::iota(in.begin(), in.end(),1.f);
    
    auto out = amt::make_tensor<float>(3,3);
    std::iota(out.begin(), out.end(),1.f);

    auto const& n = in.extents();
    auto const& w = in.strides();

    for(auto i = 0ul; i < n[0]; i += MR){
        auto ib = std::min(MR,n[0]-i);
        auto op = out.data() + i * n[1];
        auto in_ptr = in.data() + i * w[0];
        amt::pack(op, ib, in_ptr, w.data(), ib, n[1]);
    }

    REQUIRE(in[0] == out[0]);
    REQUIRE(in[1] == out[1]);
    REQUIRE(in[2] == out[6]);
    REQUIRE(in[3] == out[2]);
    REQUIRE(in[4] == out[3]);
    REQUIRE(in[5] == out[7]);
    REQUIRE(in[6] == out[4]);
    REQUIRE(in[7] == out[5]);
    REQUIRE(in[8] == out[8]);

}

TEST_CASE( "Transposed Pack [3,3] ", "[pack_mtm3x3_trans]" ) {
    constexpr auto MR = 2ul;
    
    // in = 1 4 7
    //      2 5 8
    //      3 6 9
    
    // out =1 5 7
    //      4 3 8
    //      2 6 9

    auto in = amt::make_tensor<float>(3,3);
    std::iota(in.begin(), in.end(),1.f);
    
    auto out = amt::make_tensor<float>(3,3);
    std::iota(out.begin(), out.end(),1.f);

    auto const& n = in.extents();
    auto const& w = in.strides();

    for(auto i = 0ul; i < n[0]; i += MR){
        auto ib = std::min(MR,n[0]-i);
        auto op = out.data() + i * n[1];
        auto in_ptr = in.data() + i * w[1];
        amt::pack(op, ib, in_ptr, w.data(), ib, n[1],amt::tag::trans{});
    }

    REQUIRE(in[0] == out[0]);
    REQUIRE(in[1] == out[2]);
    REQUIRE(in[2] == out[4]);
    REQUIRE(in[3] == out[1]);
    REQUIRE(in[4] == out[3]);
    REQUIRE(in[5] == out[5]);
    REQUIRE(in[6] == out[6]);
    REQUIRE(in[7] == out[7]);
    REQUIRE(in[8] == out[8]);

}