#include <iostream>
#include <numeric>
#include <algorithm>
#include <vector>
#include <cassert>
#include <timer.hpp>
#include <matplot/matplot.h>
#include <boost/numeric/ublas/tensor.hpp>
#include <cblas.h>

namespace plt = matplot;
namespace ub = boost::numeric::ublas;

void compare_mat(auto const& l, auto const& r){
    for(auto i = 0ul; i < l.size(0); ++i){
        for(auto j = 0ul; j < l.size(1); ++j){
            auto lv = l(i,j);
            auto rv = r(i,j);
            if(lv != rv){
                std::cerr<<l.extents()<<' '<<l.strides()<<'\n';
                std::cerr<<lv<<' '<<rv<<'\n';
                std::cerr<<l<<'\n';
                std::cerr<<r<<'\n';
                exit(0);
            }
        }
    }
}

int main(){
    constexpr auto max_sz = 128;
    constexpr double peak_performance = 2.3 * 8 * 32;
    std::vector<double> x(max_sz), y(max_sz);
    // std::iota(x.begin(), x.end(), 1.f);
    constexpr auto inc = 8.;
    x[0] = inc;
    for(auto i = 1u; i < max_sz; ++i) x[i] = (x[i - 1] + inc);
    // x[0] = 16;
    auto i = 0u;
    auto avg_flops = 0.;
    auto max_flops = 0.;
    auto min_flops = peak_performance;
    for(auto const& el : x)
    {   
        std::size_t M = static_cast<std::size_t>(el);
        std::size_t N = 1u;
        auto ops = 2. * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(1ul);
        ub::dynamic_tensor<float> a(ub::extents<>{M,K});
        std::iota(a.begin(), a.end(), 1.f);
        ub::dynamic_tensor<float> b(ub::extents<>{K,N});
        std::iota(b.begin(), b.end(), 2.f);
        ub::dynamic_tensor<float> c(ub::extents<>{M,N});
        ub::dynamic_tensor<float> ctemp(ub::extents<>{M,N});
        [[maybe_unused]] auto wa = a.extents().data();
        [[maybe_unused]] auto wb = b.extents().data();
        [[maybe_unused]] auto wc = c.extents().data();
        auto sa = a.strides().data();
        auto sb = b.strides().data();
        auto sc = c.strides().data();
        // std::cerr<<a<<'\n';
        // std::cerr<<b<<'\n';
        amt::timer t{};
        {
            amt::mat_mul_col_block(
                c.data(), sc, wc,
                a.data(), sa, wa,
                b.data(), sb, wb,
                amt::matrix_partition<float>{}
            );
            // cblas_sgemm(
            //     CblasColMajor, CblasNoTrans, CblasNoTrans, 
            //     static_cast<blasint>(M), static_cast<blasint>(N), static_cast<blasint>(K), 1.f, 
            //     a.data(), static_cast<blasint>(sa[1]), 
            //     b.data(), static_cast<blasint>(sb[1]), 0.f, 
            //     c.data(), static_cast<blasint>(sc[1]) );
        }
        auto st = t.stop() * 10.;
        y[i++] = ( ops / st ) * 10e-9;
        avg_flops += y[ i - 1 ];
        max_flops = std::max(max_flops, y[i-1]);
        min_flops = std::min(min_flops, y[i-1]);
        // ub::detail::recursive::mtm(
        //     ctemp.data(), ctemp.extents().data(), ctemp.strides().data(),
        //     a.data(), a.extents().data(), a.strides().data(),
        //     b.data(), b.extents().data(), b.strides().data()
        // );
        // compare_mat(c,ctemp);
        // if(M == 16) { std::cout<<c<<'\n'<<ctemp<<'\n'; exit(0); }
        // std::cerr<<c<<'\n';
        // std::cerr<<ctemp<<'\n';
        std::cerr<<"M: "<<M<<", N: "<<N<<", K: "<<K<<", GFLOPS: "<<y[i - 1]<<", Utilizing: "<<(y[i - 1] / peak_performance) * 100.<<'\n';
    }
    std::cerr<<"Avg Flops: "<<(avg_flops / max_sz)<<'\n';
    std::cerr<<"Max Flops: "<<max_flops<<'\n';
    std::cerr<<"Min Flops: "<<min_flops<<'\n';
    // plt::scatter(x,y,2);
    // plt::show();
    return 0;
}
