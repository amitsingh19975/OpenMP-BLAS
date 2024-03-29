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
std::string_view ublas_dot_prod(std::vector<double> const& x, amt::metric<ValueType>& m){
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
    return "ublas.csv";
}

template<typename ValueType, std::size_t MaxIter = 100ul>
std::string_view mkl_dot_prod(std::vector<double> const& x, amt::metric<ValueType>& m){
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
        auto v1 = amt::make_tensor<ValueType>(1ul,sz,1.);
        auto v2 = amt::make_tensor<ValueType>(1ul,sz,1.);
        double st = amt::benchmark<MaxIter>(bench_fn, static_cast<MKL_INT>(sz),v1.data(), 1, v2.data(), 1);
        metric_data.update((ops / st));
    }
    return "mkl.csv";
}

#ifdef AMT_BENCHMARK_OPENBLAS_HPP
template<typename ValueType, std::size_t MaxIter = 100ul>
std::string_view openblas_dot_prod(std::vector<double> const& x, amt::metric<ValueType>& m){
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
        auto v1 = amt::make_tensor<ValueType>(1ul,sz,1.);
        auto v2 = amt::make_tensor<ValueType>(1ul,sz,1.);
        double st = amt::benchmark<MaxIter>(bench_fn, static_cast<blasint>(sz), v1.data(), 1, v2.data(), 1);
        metric_data.update((ops / st));
    }
    return "openblas.csv";
}
#endif

template<typename ValueType, std::size_t MaxIter = 100ul>
std::string_view openmp_dot_prod(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    std::string_view fn_name = "Boost.ublas.tensor";
    
    auto& metric_data = m[fn_name];
    ValueType ret{};

    for(auto const& el : x){
        double const ops = 2. * el;
        auto sz = static_cast<std::size_t>(el);
        auto v1 = amt::make_tensor<ValueType>(1ul,sz,1.);
        auto v2 = amt::make_tensor<ValueType>(1ul,sz,1.);
        auto bench_fn = amt::dot_prod(ret,v1,v2,std::nullopt);
        double st = amt::benchmark<MaxIter>(std::move(bench_fn));
        metric_data.update((ops / st));
    }
    return "tensor.csv";
}

template<typename ValueType, std::size_t MaxIter = 100ul>
std::string_view blis_dot_prod(std::vector<double> const& x, amt::metric<ValueType>& m){
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
        auto v1 = amt::make_tensor<ValueType>(1ul,sz,1.);
        auto v2 = amt::make_tensor<ValueType>(1ul,sz,1.);
        double st = amt::benchmark<MaxIter>(bench_fn, BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, static_cast<dim_t>(sz), v1.data(), 1, v2.data(), 1, &ret);
        metric_data.update((ops / st));
    }
    return "blis.csv";
}

template<typename ValueType, std::size_t MaxIter = 100ul>
std::string_view eigen_dot_prod(std::vector<double> const& x, amt::metric<ValueType>& m){
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
    return "eigen.csv";

}

#define DISABLE_PLOT
// #define SPEEDUP_PLOT
#define PLOT_ALL
#define SIZE_KiB true


#ifdef SIZE_KiB
    #define SIZE_SUFFIX "[KiB]"
#else
    #define SIZE_KiB false
    #define SIZE_SUFFIX "[MiB]"
#endif

constexpr double size_conv(double val) noexcept{ 
    if constexpr( SIZE_KiB )
        return KiB(val);
    else
        return MiB(val);
}

int main(){
    
    using value_type = float;
    // using value_type = double;
    std::vector<double> x;
    [[maybe_unused]]constexpr std::size_t max_iter = 100ul;
    // [[maybe_unused]]constexpr double max_value = 16382;
    // amt::range(x, 32., max_value, 32., std::plus<>{});
    [[maybe_unused]]constexpr double max_value = (1u<<20);
    amt::range(x, 2., max_value, 1024., std::plus<>{});
    // amt::range(x, 2., max_value, 2., std::multiplies<>{});

    auto m = amt::metric<value_type>(x.size());
    std::string_view fn_name;

    // fn_name = ublas_dot_prod<value_type,max_iter>(x,m);
    fn_name = openblas_dot_prod<value_type,max_iter>(x,m);
    // fn_name = blis_dot_prod<value_type,max_iter>(x,m);
    // fn_name = eigen_dot_prod<value_type,max_iter>(x,m);
    // fn_name = openmp_dot_prod<value_type,max_iter>(x,m);
    // fn_name = mkl_dot_prod<value_type,max_iter>(x,m);
    // std::cout<<m.tail();

    constexpr std::string_view comp_name = "tensor";

    std::transform(x.begin(), x.end(), x.begin(), [](auto sz){
        return size_conv(sz);
    });

    std::cout<<m.str(comp_name)<<'\n';
    m.csv(fn_name);
    #ifndef DISABLE_PLOT
        constexpr std::string_view plot_xlable = "Size " SIZE_SUFFIX;
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
