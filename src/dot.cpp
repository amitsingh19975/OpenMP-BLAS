#define BOOST_UBLAS_USE_SIMD
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

template<typename ValueType>
void compare_mat_helper(std::size_t sz){
    ub::dynamic_tensor<ValueType> v1(ub::extents<>{1ul, sz},3.), v2(ub::extents<>{1ul, sz}, 3.);
    auto lres = ValueType{};
    auto rres = ValueType{}; 
    amt::dot_prod(rres, v1, v2);
    if constexpr(std::is_same_v<ValueType,float>){
        lres = cblas_sdot(static_cast<blasint>(sz),v1.data(), 1u, v2.data(), 1u);
    }else if constexpr(std::is_same_v<ValueType,double>){
        lres = cblas_ddot(static_cast<blasint>(sz),v1.data(), 1u, v2.data(), 1u);
    }
    if(lres != rres){
        std::cerr<<"Incorrect Result: Tensor( " << rres << " ), BLAS( " << lres << " )\n";
        exit(1);
    }
}

template<typename ValueType>
void compare_mat(std::size_t start, std::size_t end, std::size_t inc = 1u){
    for(; start < end; start += inc) compare_mat_helper<ValueType>(start);
    std::cerr << "TEST PASSED!" << std::endl;
}

int compare_mati(auto const& l, auto const& r){
    return static_cast<int>(l + r);
}

template<typename ValueType>
int ublas_dot_same_layout(std::vector<double> const& x, amt::metric& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );

    std::string fn_name = std::string(__func__) 
        + (std::is_same_v<ValueType,float> ? "_float" : "_double");
    
    auto& metric_data = m[fn_name];
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

        metric_data.update((ops / st) * 10e-9);
    }
    return static_cast<int>(ret);
}

template<typename ValueType>
int blas_same_layout(std::vector<double> const& x, amt::metric& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    std::string fn_name = std::string(__func__) 
        + (std::is_same_v<ValueType,float> ? "_float" : "_double");
    
    auto& metric_data = m[fn_name];
    ValueType ret{};

    for(auto const& el : x){
        double const ops = 2. * el;
        auto sz = static_cast<std::size_t>(el);
        ub::dynamic_tensor<ValueType> v1(ub::extents<>{1ul, sz},3.), v2(ub::extents<>{1ul, sz}, 3.);
        amt::timer t{};
        {   
            if constexpr(std::is_same_v<ValueType,float>){
                ret = cblas_sdot(static_cast<blasint>(sz),v1.data(), 1u, v2.data(), 1u);
            }else if constexpr(std::is_same_v<ValueType,double>){
                ret = cblas_ddot(static_cast<blasint>(sz),v1.data(), 1u, v2.data(), 1u);
            }
        }

        auto st = t.stop();
        metric_data.update((ops / st) * 10e-9);
    }
    return static_cast<int>(ret);
}

template<typename ValueType>
int tensor_same_layout(std::vector<double> const& x, amt::metric& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    std::string fn_name = std::string(__func__) 
        + (std::is_same_v<ValueType,float> ? "_float" : "_double");
    
    auto& metric_data = m[fn_name];
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
        metric_data.update((ops / st) * 10e-9);
    }
    return static_cast<int>(ret);
}

// #define TEST_ON

int main(){
    using value_type = float;
    constexpr auto max_size = 4096ul;
    
#ifndef TEST_ON

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

#else
    compare_mat<value_type>(2,max_size);
    return 0;
#endif
}
