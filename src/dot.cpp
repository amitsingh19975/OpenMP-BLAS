#include <iostream>
#include <numeric>
#include <algorithm>
#include <vector>
#include <cassert>
#include <timer.hpp>
#include <metric.hpp>
#include <dot.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <cblas.h>
#include <complex>

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

template<typename ValueType>
int ublas_dot_same_layout(std::vector<double> const& x, amt::metric& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );

    ValueType ret{};
    for(auto const& el : x){
        double const ops = 2. * el;
        auto sz = static_cast<std::size_t>(el);
        ub::vector<ValueType> v1(sz,3.), v2(sz, 3.);
        amt::timer t{};
        {
            ret += ub::inner_prod(v1,v2);
        }
        auto st = t.stop();
        if constexpr(std::is_same_v<ValueType,float>){
            m.insert_or_update("ublas_dot_same_layout_float", (ops / st) * 10e-9);
        }else if constexpr(std::is_same_v<ValueType,double>){
            m.insert_or_update("ublas_dot_same_layout_double", (ops / st) * 10e-9);
        }
    }
    return static_cast<int>(ret);
}

template<typename ValueType>
int blas_same_layout(std::vector<double> const& x, amt::metric& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );

    ValueType ret{};
    for(auto const& el : x){
        double const ops = 2. * el;
        auto sz = static_cast<std::size_t>(el);
        std::vector<ValueType> v1(sz,3.), v2(sz, 3.);
        amt::timer t{};
        {   
            if constexpr(std::is_same_v<ValueType,float>){
                ret += cblas_sdot(static_cast<blasint>(sz),v1.data(), 1u, v2.data(), 1u);
            }else if constexpr(std::is_same_v<ValueType,double>){
                ret += cblas_ddot(static_cast<blasint>(sz),v1.data(), 1u, v2.data(), 1u);
            }
        }
        auto st = t.stop();
        if constexpr(std::is_same_v<ValueType,float>){
            m.insert_or_update("blas_same_layout_float", (ops / st) * 10e-9);
        }else if constexpr(std::is_same_v<ValueType,double>){
            m.insert_or_update("blas_same_layout_double", (ops / st) * 10e-9);
        }
    }
    return static_cast<int>(ret);
}

template<typename ValueType>
int tensor_same_layout(std::vector<double> const& x, amt::metric& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );

    ValueType ret{};
    for(auto const& el : x){
        double const ops = 2. * el;
        auto sz = static_cast<std::size_t>(el);
        ub::dynamic_tensor<ValueType> v1(ub::extents<>{1ul, sz},3.), v2(ub::extents<>{1ul, sz}, 3.);
        amt::timer t{};
        {   
            amt::dot_prod(ret, v1, v2);
        }
        auto st = t.stop();
        if constexpr(std::is_same_v<ValueType,float>){
            m.insert_or_update("tensor_same_layout_float", (ops / st) * 10e-9);
        }else if constexpr(std::is_same_v<ValueType,double>){
            m.insert_or_update("tensor_same_layout_double", (ops / st) * 10e-9);
        }
    }
    return static_cast<int>(ret);
}

int main(){
    using value_type = double;
    constexpr auto max_size = 1000ul;
    int res = 0;
    std::vector<double> x(max_size);
    std::iota(x.begin(), x.end(), 2.);
    auto m = amt::metric(max_size);
    res += ublas_dot_same_layout<value_type>(x,m);
    res += blas_same_layout<value_type>(x,m);
    res += tensor_same_layout<value_type>(x,m);
    std::cout<<m<<'\n';
    m.plot(x);
    return res;
}
