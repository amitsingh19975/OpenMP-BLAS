#define BOOST_UBLAS_USE_SIMD
#include <iostream>
#include <numeric>
#include <algorithm>
#include <vector>
#include <cassert>
#include <benchmark.hpp>
#include <metric.hpp>
#include <dot.hpp>
#include <boost/numeric/ublas/vector.hpp>
// #include <cblas.h>
#include <mkl_cblas.h>
#include <blis.h>
#include <Eigen/Dense>
#include <complex>
#include <boost/mp11/list.hpp>
#include <boost/mp11/algorithm.hpp>
#include <openblas.hpp>
#include <range.hpp>

namespace plt = matplot;
namespace ub = boost::numeric::ublas;


template<typename ValueType, std::size_t MaxIter = 100ul>
void ublas_same_layout(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );

    std::string fn_name = std::string(__func__) 
        + (std::is_same_v<ValueType,float> ? "_float" : "_double");
    
    auto& metric_data = m[fn_name];
    constexpr auto bench_fn = [](auto const& v1, auto const& v2){
        amt::no_opt(ub::inner_prod(v1,v2));
    };

    for(auto const& el : x){
        double const ops = 2. * el;
        auto sz = static_cast<std::size_t>(el);
        ub::vector<ValueType> v1(sz), v2(sz);
        double st = amt::benchmark<MaxIter>(bench_fn, v1, v2);
        metric_data.update((ops / st));
    }
}

template<typename ValueType, std::size_t MaxIter = 100ul>
void mkl_same_layout(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    std::string fn_name = std::string(__func__) 
        + (std::is_same_v<ValueType,float> ? "_float" : "_double");
    
    auto& metric_data = m[fn_name];
    constexpr auto bench_fn = [](auto&&... args){
        if constexpr(std::is_same_v<ValueType,float>){
            amt::no_opt(cblas_sdot(std::forward<decltype(args)>(args)...));
        }else if constexpr(std::is_same_v<ValueType,double>){
            amt::no_opt(cblas_ddot(std::forward<decltype(args)>(args)...));
        }
    };

    for(auto const& el : x){
        double const ops = 2. * el;
        auto sz = static_cast<std::size_t>(el);
        ub::dynamic_tensor<ValueType> v1(ub::extents<>{1ul, sz}), v2(ub::extents<>{1ul, sz});
        double st = amt::benchmark<MaxIter>(bench_fn, static_cast<MKL_INT>(sz),v1.data(), 1, v2.data(), 1);
        metric_data.update((ops / st));
    }
}

#ifdef AMT_BENCHMARK_OPENBLAS_HPP
template<typename ValueType, std::size_t MaxIter = 100ul>
void openblas_same_layout(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    std::string fn_name = std::string(__func__) 
        + (std::is_same_v<ValueType,float> ? "_float" : "_double");
    
    auto& metric_data = m[fn_name];
    constexpr auto bench_fn = [](auto&&... args){
        amt::no_opt(amt::blas::dot_prod(std::forward<decltype(args)>(args)...));
    };

    for(auto const& el : x){
        double const ops = 2. * el;
        auto sz = static_cast<std::size_t>(el);
        ub::dynamic_tensor<ValueType> v1(ub::extents<>{1ul, sz}), v2(ub::extents<>{1ul, sz});
        double st = amt::benchmark<MaxIter>(bench_fn, static_cast<blasint>(sz), v1.data(), 1, v2.data(), 1);
        metric_data.update((ops / st));
    }
}
#endif

template<typename ValueType, std::size_t MaxIter = 100ul>
void openmp_same_layout(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    std::string fn_name = std::string(__func__) 
        + (std::is_same_v<ValueType,float> ? "_float" : "_double");
    
    auto& metric_data = m[fn_name];
    ValueType ret{};

    constexpr auto bench_fn = [](ValueType& ret, auto&&... args){
        amt::dot_prod(ret, std::forward<decltype(args)>(args)...);
    };

    for(auto const& el : x){
        double const ops = 2. * el;
        auto sz = static_cast<std::size_t>(el);
        ub::dynamic_tensor<ValueType> v1(ub::extents<>{1ul, sz}), v2(ub::extents<>{1ul, sz});
        double st = amt::benchmark_timer_as_arg<MaxIter>(bench_fn, ret, v1, v2, std::nullopt);
        metric_data.update((ops / st));
    }
}

// template<std::size_t Start, std::size_t End, typename ValueType>
// int static_tensor_same_layout(amt::metric<ValueType>& m){
//     using namespace boost::mp11;
//     static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
//     std::string fn_name = std::string(__func__) 
//         + (std::is_same_v<ValueType,float> ? "_float" : "_double");
    
//     auto& metric_data = m[fn_name];
//     ValueType ret{};

//     using number_list = mp_iota_c<End>;
//     using range = mp_drop_c<number_list, std::max(1ul, Start) >;

//     mp_for_each<range>([&](auto I){
//         constexpr auto sz = decltype(I)::value;
//         double const ops = 2. * sz;
//         using extents_type = ub::static_extents<1ul, sz>;
//         using tensor_type = ub::static_tensor<ValueType,extents_type>;
//         tensor_type v1, v2;
//         double st{};
//         auto k = max_iter;
//         while(k--){
//             amt::timer t{};
//             {   
//                 amt::dot_prod(ret, v1, v2, t);
//             }
//             st += t();
//         }
//         st /= static_cast<double>(max_iter);
//         metric_data.update((ops / st));
//     });
//     return static_cast<int>(ret);
// }

template<typename ValueType, std::size_t MaxIter = 100ul>
void blis_same_layout(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    std::string fn_name = std::string(__func__) 
        + (std::is_same_v<ValueType,float> ? "_float" : "_double");
    
    auto& metric_data = m[fn_name];
    ValueType ret{};
    constexpr auto bench_fn = [](auto&&... args){
        if constexpr(std::is_same_v<ValueType,float>){
            bli_sdotv(std::forward<decltype(args)>(args)...);
        }else if constexpr(std::is_same_v<ValueType,double>){
            bli_ddotv(std::forward<decltype(args)>(args)...);
        }
    };

    for(auto const& el : x){
        double const ops = 2. * el;
        auto sz = static_cast<std::size_t>(el);
        ub::dynamic_tensor<ValueType> v1(ub::extents<>{1ul, sz}), v2(ub::extents<>{1ul, sz});
        double st = amt::benchmark<MaxIter>(bench_fn, BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, static_cast<dim_t>(sz), v1.data(), 1, v2.data(), 1, &ret);
        metric_data.update((ops / st));
    }
}

template<typename ValueType, std::size_t MaxIter = 100ul>
void eigen_same_layout(std::vector<double> const& x, amt::metric<ValueType>& m){
    using namespace Eigen;
    using vector_type = Matrix<ValueType,-1,1,ColMajor>;
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    std::string fn_name = std::string(__func__) 
        + (std::is_same_v<ValueType,float> ? "_float" : "_double");
    
    auto& metric_data = m[fn_name];

    constexpr auto bench_fn = [](auto const& v1, auto const& v2){
        amt::no_opt(v1.dot(v2));
    };

    for(auto const& el : x){
        double const ops = 2. * el;
        auto sz = static_cast<std::size_t>(el);
        vector_type v1(sz), v2(sz);
        double st = amt::benchmark<MaxIter>(bench_fn, v1, v2);
        metric_data.update((ops / st));
    }
}

// #define DIFFERENT_LAYOUT
// #define DISABLE_PLOT
// #define SPEEDUP_PLOT
#define PLOT_ALL

int main(){
    using value_type = float;
    // using value_type = double;
    std::vector<double> x;
    [[maybe_unused]]constexpr std::size_t max_iter = 100ul;
    [[maybe_unused]]constexpr double max_value = 16382;
    amt::range(x, 32., max_value, 32., std::plus<>{});
    // [[maybe_unused]]constexpr double max_value = (1u<<20);
    // amt::range(x, 2., max_value, 1024., std::plus<>{});
    // amt::range(x, 2., max_value, 2., std::multiplies<>{});

    auto m = amt::metric<value_type>(x.size());
    // exit(0);
    #ifndef DIFFERENT_LAYOUT
        ublas_same_layout<value_type,max_iter>(x,m);
        openblas_same_layout<value_type,max_iter>(x,m);
        blis_same_layout<value_type,max_iter>(x,m);
        eigen_same_layout<value_type,max_iter>(x,m);
        openmp_same_layout<value_type,max_iter>(x,m);
        mkl_same_layout<value_type,max_iter>(x,m);
        // std::cout<<m.tail();
    #else
        // res += ref_dot_diff_layout<value_type,ub::layout::first_order>(x,m);
        // res += openblas_dot_diff_layout<value_type,ub::layout::first_order>(x,m);
        // res += tensor_dot_diff_layout<value_type,ub::layout::first_order>(x,m);
        // res += blis_dot_diff_layout<value_type,ub::layout::first_order>(x,m);
        // res += eigen_dot_diff_layout<value_type,ub::layout::first_order>(x,m);

    #endif

    // std::cout<<m.str("openmp")<<'\n';
    #ifndef DISABLE_PLOT
        #if !defined(SPEEDUP_PLOT) || defined(PLOT_ALL)
            m.plot(x);
            m.plot_per();
        #endif
        
        #if defined(SPEEDUP_PLOT) || defined(PLOT_ALL)
            // m.plot_speedup("openmp",x);
            m.plot_speedup_per("openmp");
        #endif
    #endif
    // m.raw();
    return 0;

}
