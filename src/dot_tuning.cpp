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

// template<typename ValueType, std::size_t MaxIter = 100ul>
// void mkl_dot_prod(std::size_t N, std::vector<double> const& x, amt::metric<ValueType>& m){
//     static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
//     std::string_view fn_name = "intel MKL";
    
//     auto& metric_data = m[fn_name];
//     constexpr auto bench_fn = [](auto&&... args){
//         if constexpr(std::is_same_v<ValueType,float>){
//             amt::no_opt(cblas_sdot(std::forward<decltype(args)>(args)...));
//         }else if constexpr(std::is_same_v<ValueType,double>){
//             amt::no_opt(cblas_ddot(std::forward<decltype(args)>(args)...));
//         }
//     };

//     for(auto const& el : x){
//         double const ops = 2. * el;
//         auto sz = static_cast<std::size_t>(el);
//         ub::dynamic_tensor<ValueType> v1(ub::extents<>{1ul, sz}), v2(ub::extents<>{1ul, sz});
//         double st = amt::benchmark<MaxIter>(bench_fn, static_cast<MKL_INT>(sz),v1.data(), 1, v2.data(), 1);
//         metric_data.update((ops / st));
//     }
// }

template<int BPos = 0, typename ValueType, std::size_t MaxIter = 100ul>
void openmp_dot_prod(std::size_t N, std::size_t B1, std::size_t B2, std::size_t B3, std::vector<double> const& x, amt::metric<ValueType>& m, std::string_view name){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    std::string_view fn_name = std::move(name);
    
    auto& metric_data = m[fn_name];
    ValueType ret{};

    constexpr auto bench_fn = [](ValueType& ret, auto&&... args){
        amt::dot_prod_tuning_param(ret,std::forward<decltype(args)>(args)...);
    };

    if constexpr(BPos < 0){
        for(auto const& el : x){
            auto sz = static_cast<std::size_t>(el);
            ub::dynamic_tensor<ValueType> v1(ub::extents<>{1ul, sz});
            auto v2 = v1;
            double const ops = 2. * el;
            double st = amt::benchmark_timer_as_arg<MaxIter>(bench_fn, ret, v1, v2, B1, B2, B3, std::nullopt);
            metric_data.update((ops / st));
        }
    }else{
        for(auto const& el : x){
            ub::dynamic_tensor<ValueType> v1(ub::extents<>{1ul, N});
            auto v2 = v1;
            double const ops = 2. * static_cast<double>(N);
            auto sz = static_cast<std::size_t>(el);
            double st{};
            if constexpr( BPos == 0 ){
                st = amt::benchmark_timer_as_arg<MaxIter>(bench_fn, ret, v1, v2, sz, B2, B3, std::nullopt);
            }else if constexpr( BPos == 1 ){
                st = amt::benchmark_timer_as_arg<MaxIter>(bench_fn, ret, v1, v2, B1, sz, B3, std::nullopt);
            }else{
                st = amt::benchmark_timer_as_arg<MaxIter>(bench_fn, ret, v1, v2, B1, B2, sz, std::nullopt);
            }
            metric_data.update((ops / st));
        }
    }
}


// #define DISABLE_PLOT
// #define SPEEDUP_PLOT
// #define PLOT_ALL
#define SIZE_KiB

#ifdef SIZE_KiB
    #define SIZE_SUFFIX "[K]"
    constexpr double size_conv(double val) noexcept{ return val / 1024. ;}
#else
    #define SIZE_SUFFIX "[M]"
    constexpr double size_conv(double val) noexcept{ return val / ( 1024. * 1024. ); }
#endif

int main(){
    
    using value_type = float;
    // using value_type = double;
    [[maybe_unused]] auto const number_of_el_l1 = amt::cache_manager::size(0) / sizeof(value_type);
    [[maybe_unused]] auto const number_of_el_l2 = amt::cache_manager::size(1) / sizeof(value_type);
    [[maybe_unused]] auto const number_of_el_l3 = amt::cache_manager::size(2) / sizeof(value_type);
    [[maybe_unused]] auto const block1 = (number_of_el_l1);
    [[maybe_unused]] auto const block2 = (number_of_el_l1 >> 1u);
    [[maybe_unused]] auto const block3 = (number_of_el_l2 >> 1u);
    std::size_t N = 1 << 27;
    
    std::vector<double> x;
    [[maybe_unused]]constexpr std::size_t max_iter = 2ul;
    [[maybe_unused]]double max_value = 64 * 1024;
    amt::range(x, 1024., max_value, 1024., std::plus<>{});

    auto m = amt::metric<value_type>(x.size());
    
    // // For Block 1
    // // =================
    // openmp_dot_prod<-1,value_type,max_iter>(N,1024,block2,block3,x,m,"Boost.ublas.tensor[1024]");
    // openmp_dot_prod<-1,value_type,max_iter>(N,2048,block2,block3,x,m,"Boost.ublas.tensor[2048]");
    // openmp_dot_prod<-1,value_type,max_iter>(N,8192,block2,block3,x,m,"Boost.ublas.tensor[8192]");
    // openmp_dot_prod<-1,value_type,max_iter>(N,16384,block2,block3,x,m,"Boost.ublas.tensor[16384]");
    // openmp_dot_prod<-1,value_type,max_iter>(N,32768,block2,block3,x,m,"Boost.ublas.tensor[32768]");
    // // =================

    openmp_dot_prod<2,value_type,max_iter>(N,block1,block2,block3,x,m,"Boost.ublas.tensor");

    constexpr std::string_view plot_xlable = "Block in " SIZE_SUFFIX;
    std::transform(x.begin(), x.end(), x.begin(), [](auto sz){
        return size_conv(sz);
    });

    // std::cout<<m.str(comp_name)<<'\n';
    #ifndef DISABLE_PLOT
        matplot::grid(matplot::on);
        m.plot<true,amt::PLOT_TYPE::LINE>(x, "Tuning Parameter for Block 3 [iter=2]", plot_xlable);
    #endif
    // m.raw();
    return 0;

}
