#include <iostream>
#include <numeric>
#include <algorithm>
#include <vector>
#include <cassert>
#include <timer.hpp>
#include <metric.hpp>
#include <boost/numeric/ublas/tensor.hpp>
#include <boost/numeric/ublas/vector.hpp>
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

int ublas_dot_same_layout(std::vector<double> const& x, amt::metric& m){
    float ret{};
    for(auto const& el : x){
        double const ops = 2. * el;
        auto sz = static_cast<std::size_t>(el);
        ub::vector<float> v1(sz,3.), v2(sz, 3.);
        amt::timer t{};
        {
            ret += ub::inner_prod(v1,v2);
        }
        auto st = t.stop();
        m.insert_or_update("ublas_dot_same_layout", (ops / st) * 10e-9);
    }
    return static_cast<int>(ret);
}

int blas_same_layout(std::vector<double> const& x, amt::metric& m){
    float ret{};
    for(auto const& el : x){
        double const ops = 2. * el;
        auto sz = static_cast<std::size_t>(el);
        std::vector<float> v1(sz,3.), v2(sz, 3.);
        amt::timer t{};
        {
            ret += cblas_sdot(static_cast<blasint>(sz),v1.data(), 1u, v2.data(), 1u);
        }
        auto st = t.stop();
        m.insert_or_update("blas_same_layout", (ops / st) * 10e-9);
    }
    return static_cast<int>(ret);
}

int main(){
    
    constexpr auto max_size = 1000ul;
    int res = 0;
    std::vector<double> x(max_size);
    std::iota(x.begin(), x.end(), 1.);
    auto m = amt::metric(max_size);
    res += ublas_dot_same_layout(x,m);
    res += blas_same_layout(x,m);
    std::cout<<m<<'\n';
    m.plot(x);
    return res;
}
