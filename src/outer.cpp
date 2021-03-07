#define BOOST_UBLAS_USE_SIMD
#include <iostream>
#include <numeric>
#include <algorithm>
#include <vector>
#include <cassert>
#include <benchmark.hpp>
#include <metric.hpp>
#include <outer.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
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
using shape_t = ub::extents<2u>;


template<typename ValueType, std::size_t MaxIter = 100ul>
void ublas_outer_prod(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );

    std::string_view fn_name = "Boost.ublas";
    
    auto& metric_data = m[fn_name];
    constexpr auto bench_fn = [](auto& res, auto const& v1, auto const& v2){
        ub::noalias(res) = ub::outer_prod(v1,v2);
    };

    for(auto const& el : x){
        double const ops = el * el;
        auto sz = static_cast<std::size_t>(el);
        ub::vector<ValueType> v1(sz);
        ub::vector<ValueType> v2(sz);
        ub::matrix<ValueType> res(sz,sz);

        double st = amt::benchmark<MaxIter>(bench_fn, res, v1, v2);
        metric_data.update((ops / st));
    }
}

template<typename ValueType, std::size_t MaxIter = 100ul>
void mkl_outer_prod(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    std::string_view fn_name = "intel MKL";
    
    auto& metric_data = m[fn_name];
    constexpr auto bench_fn = [](const CBLAS_LAYOUT Layout, const MKL_INT m, const MKL_INT n, 
        const auto alpha, auto const *x, const MKL_INT incx, auto const *y, const MKL_INT incy, 
        auto *a, const MKL_INT lda)
    {
        if constexpr(std::is_same_v<ValueType,float>){
            cblas_sger(Layout, m, n, alpha, x, incx, y, incy, a, lda);
        }else if constexpr(std::is_same_v<ValueType,double>){
            cblas_dger(Layout, m, n, alpha, x, incx, y, incy, a, lda);
        }
    };
    
    for(auto const& el : x){
        double const ops = el * el;
        auto sz = static_cast<std::size_t>(el);
        ub::dynamic_tensor<ValueType> v1(shape_t{1ul, sz},1.);
        ub::dynamic_tensor<ValueType> res(shape_t{sz, sz});
        auto inc = static_cast<MKL_INT>(1);
        auto M = static_cast<MKL_INT>(sz);
        auto N = static_cast<MKL_INT>(sz);
        auto const* aptr = v1.data();
        auto const* bptr = v1.data();
        auto cptr = res.data();
        auto one = ValueType(1);
        auto lda = M;
        double st = amt::benchmark<MaxIter>(bench_fn, CblasColMajor, M, N, one, aptr, inc, bptr, inc, cptr, lda);
        metric_data.update((ops / st));
    }
}

#ifdef AMT_BENCHMARK_OPENBLAS_HPP
template<typename ValueType, std::size_t MaxIter = 100ul>
void openblas_outer_prod(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    std::string_view fn_name = "OpenBlas";
    
    auto& metric_data = m[fn_name];
    constexpr auto bench_fn = [](auto&&... args){
        amt::blas::outer_prod(std::forward<decltype(args)>(args)...);
    };

    for(auto const& el : x){
        double const ops = el * el;
        auto sz = static_cast<std::size_t>(el);
        ub::dynamic_tensor<ValueType> v1(shape_t{1ul, sz});
        ub::dynamic_tensor<ValueType> res(shape_t{sz, sz});
        auto inc = static_cast<blasint>(1);
        auto M = static_cast<blasint>(sz);
        auto N = static_cast<blasint>(sz);
        auto lda = M;
        double st = amt::benchmark<MaxIter>(bench_fn, amt::blas::ColMajor, M, N, ValueType{1}, v1.data(), inc, v1.data(), inc, res.data(), lda);
        metric_data.update((ops / st));
    }
}
#endif

template<typename ValueType, std::size_t MaxIter = 100ul>
void openmp_outer_prod(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    std::string_view fn_name = "experimental::boost(OpenMP)";
    
    auto& metric_data = m[fn_name];

    constexpr auto bench_fn = [](auto&&... args){
        amt::outer_prod(std::forward<decltype(args)>(args)...);
    };

    for(auto const& el : x){
        double const ops = el * el;
        auto sz = static_cast<std::size_t>(el);
        ub::dynamic_tensor<ValueType> v1(shape_t{1ul, sz});
        ub::dynamic_tensor<ValueType> res(shape_t{sz, sz});

        double st = amt::benchmark_timer_as_arg<MaxIter>(bench_fn, res, v1, v1, std::nullopt);
        metric_data.update((ops / st));
    }
}

template<typename ValueType, std::size_t MaxIter = 100ul>
void blis_outer_prod(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    std::string_view fn_name = "Blis";
    
    auto& metric_data = m[fn_name];
    constexpr auto bench_fn = [](auto&&... args){
        if constexpr(std::is_same_v<ValueType,float>){
            bli_sger(std::forward<decltype(args)>(args)...);
        }else if constexpr(std::is_same_v<ValueType,double>){
            bli_dger(std::forward<decltype(args)>(args)...);
        }
    };

    for(auto const& el : x){
        double const ops = el * el;
        auto sz = static_cast<std::size_t>(el);
        ub::dynamic_tensor<ValueType> v1(shape_t{1ul, sz},1.);
        ub::dynamic_tensor<ValueType> res(shape_t{sz, sz});
        auto inc = static_cast<inc_t>(1);
        auto alpha = ValueType{1};
        auto rsa = static_cast<inc_t>(res.strides()[0]);
        auto csa = static_cast<inc_t>(res.strides()[1]);

        double st = amt::benchmark<MaxIter>(bench_fn, 
            BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, 
            static_cast<dim_t>(sz), static_cast<dim_t>(sz), 
            &alpha, 
            v1.data(), inc,
            v1.data(), inc, 
            res.data(), rsa, csa
        );
        metric_data.update((ops / st));
    }
}

template<typename ValueType, std::size_t MaxIter = 100ul>
void eigen_outer_prod(std::vector<double> const& x, amt::metric<ValueType>& m){
    using namespace Eigen;
    using vector_type = Matrix<ValueType,-1,1,ColMajor>;
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    std::string_view fn_name = "Eigen";
    // Eigen::setNbThreads(16);
    
    auto& metric_data = m[fn_name];

    constexpr auto bench_fn = [](auto& res, auto const& v1, auto const& v2){
        res.noalias() = v1 * v2.transpose();
    };

    for(auto const& el : x){
        double const ops = el * el;
        auto sz = static_cast<std::size_t>(el);
        vector_type v1(sz), v2(sz);
        Matrix<ValueType,-1,-1> res(sz,sz);
        double st = amt::benchmark<MaxIter>(bench_fn, res, v1, v2);
        metric_data.update((ops / st));
    }

}

// #define DISABLE_PLOT
// #define SPEEDUP_PLOT
#define PLOT_ALL

int main(){
    Eigen::initParallel();
    using value_type = float;
    // using value_type = double;
    std::vector<double> x;
    [[maybe_unused]]constexpr std::size_t max_iter = 1ul;
    [[maybe_unused]]constexpr double max_value = 4 * 1024;
    amt::range(x, 2., max_value, 32., std::plus<>{});
    // [[maybe_unused]]constexpr double max_value = (1u<<20);
    // amt::range(x, 2., max_value, 1024., std::plus<>{});
    // amt::range(x, 2., max_value, 2., std::multiplies<>{});

    auto m = amt::metric<value_type>(x.size());

    mkl_outer_prod<value_type,max_iter>(x,m);
    ublas_outer_prod<value_type,max_iter>(x,m);
    openblas_outer_prod<value_type,max_iter>(x,m);
    blis_outer_prod<value_type,max_iter>(x,m);
    eigen_outer_prod<value_type,max_iter>(x,m);
    openmp_outer_prod<value_type,max_iter>(x,m);
    // std::cout<<m.tail();

    std::cout<<m.str()<<'\n';
    #ifndef DISABLE_PLOT
        #if !defined(SPEEDUP_PLOT) || defined(PLOT_ALL)
            m.plot(x);
            m.plot_per();
        #endif
        
        #if defined(SPEEDUP_PLOT) || defined(PLOT_ALL)
            // m.plot_speedup("OpenMP",x);
            // m.plot_speedup_semilogy("OpenMP",x);
            // m.plot_speedup_per("OpenMP");
        #endif
    #endif
    // m.raw();
    return 0;

}
