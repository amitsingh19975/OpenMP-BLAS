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
#include <utils.hpp>

namespace plt = matplot;
namespace ub = boost::numeric::ublas;


template<typename ValueType, std::size_t MaxIter = 100ul>
void ublas_dot_prod(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );

    std::string_view fn_name = "Boost.ublas";
    
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
void mkl_dot_prod(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    std::string_view fn_name = "intel MKL";
    
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
void openblas_dot_prod(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    openblas_set_num_threads(omp_get_max_threads());
    std::string_view fn_name = "OpenBlas";
    
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
void openmp_dot_prod(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    std::string_view fn_name = "Boost.ublas.tensor";
    
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

template<typename ValueType, std::size_t MaxIter = 100ul>
void blis_dot_prod(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    std::string_view fn_name = "Blis";
    
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
void eigen_dot_prod(std::vector<double> const& x, amt::metric<ValueType>& m){
    using namespace Eigen;
    using vector_type = Matrix<ValueType,-1,1,ColMajor>;
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    std::string_view fn_name = "Eigen";
    Eigen::initParallel();
    Eigen::setNbThreads(omp_get_max_threads());
    
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

// #define DISABLE_PLOT
// #define SPEEDUP_PLOT
#define PLOT_ALL
#define SIZE_KiB

#ifdef SIZE_KiB
    #define SIZE_SUFFIX "[KiB]"
    constexpr double size_conv(double val) noexcept{ return val / 1024. ;}
#else
    #define SIZE_SUFFIX "[MiB]"
    constexpr double size_conv(double val) noexcept{ return val / ( 1024. * 1024. ); }
#endif

int main(){
    
    // using value_type = float;
    using value_type = double;
    std::vector<double> x;
    [[maybe_unused]]constexpr std::size_t max_iter = 100ul;
    // [[maybe_unused]]constexpr double max_value = 16382;
    // amt::range(x, 32., max_value, 32., std::plus<>{});
    [[maybe_unused]]constexpr double max_value = (1u<<20);
    amt::range(x, 2., max_value, 1024., std::plus<>{});
    // amt::range(x, 2., max_value, 2., std::multiplies<>{});

    auto m = amt::metric<value_type>(x.size());

    ublas_dot_prod<value_type,max_iter>(x,m);
    openblas_dot_prod<value_type,max_iter>(x,m);
    blis_dot_prod<value_type,max_iter>(x,m);
    eigen_dot_prod<value_type,max_iter>(x,m);
    openmp_dot_prod<value_type,max_iter>(x,m);
    mkl_dot_prod<value_type,max_iter>(x,m);
    // std::cout<<m.tail();

    constexpr std::string_view comp_name = "tensor";

    constexpr std::string_view plot_xlable = "Size " SIZE_SUFFIX;
    std::transform(x.begin(), x.end(), x.begin(), [](auto sz){
        return size_conv(sz);
    });

    std::cout<<m.str(comp_name)<<'\n';
    #ifndef DISABLE_PLOT
        #if !defined(SPEEDUP_PLOT) || defined(PLOT_ALL)
            m.plot(x, "Performance of Boost.uBLAS.Tensor for the dot-operation [iter=100]", plot_xlable);
            m.plot_per("Sorted performance of Boost.uBLAS.Tensor for the dot-operation [iter=100]");
        #endif
        
        #if defined(SPEEDUP_PLOT) || defined(PLOT_ALL)
            m.plot_speedup(comp_name,x,"Speedup of Boost.uBLAS.Tensor for the dot-operation [iter=100]", plot_xlable);
            auto inter_pts = m.plot_speedup_per<true>(comp_name,"Sorted speedup of Boost.uBLAS.Tensor for the dot-operation [iter=100]");
            m.plot_speedup_semilogy<true>(comp_name,x,"Semilogy speedup of Boost.uBLAS.Tensor for the dot-operation [iter=100]", plot_xlable);
            amt::show_intersection_pts(std::cout,inter_pts);
        #endif
    #endif
    // m.raw();
    return 0;

}
