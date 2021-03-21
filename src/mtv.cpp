#define BOOST_UBLAS_USE_SIMD
#include <iostream>
#include <numeric>
#include <algorithm>
#include <vector>
#include <cassert>
#include <benchmark.hpp>
#include <metric.hpp>
#include <mtv.hpp>
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
#include <utils.hpp>

namespace plt = matplot;
namespace ub = boost::numeric::ublas;
using shape_t = ub::extents<2u>;


static std::size_t fixed_size = 1024ul;
static bool is_lrect_matrix = false;
static bool is_rrect_matrix = false;

auto lset(std::size_t l) noexcept{
    return is_lrect_matrix ? fixed_size : l;
}

auto rset(std::size_t r) noexcept{
    return is_rrect_matrix ? fixed_size : r;
}

std::string xlable(std::string_view prefix = "Size"){
    std::stringstream ss;
    if(is_lrect_matrix){
        ss << prefix << "[n]( m = " << fixed_size << " )";
    }else if(is_rrect_matrix){
        ss << prefix << "[m]( n = " << fixed_size << " )";
    }else{
        ss << prefix << "( m = n )";
    }

    return ss.str();
}

void check(bool cond, std::string_view mess = "") noexcept{
    if(!cond){
        std::cerr<<mess<<std::endl;
        exit(1);
    }
}


template<typename ValueType, std::size_t MaxIter = 100ul>
void ublas_gemv(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );

    constexpr std::string_view fn_name = "Boost.ublas";
    
    auto& metric_data = m[fn_name];
    constexpr auto bench_fn = [](auto& res, auto const& v1, auto const& v2) -> auto&{
        ub::noalias(res) = ub::prod(v1,v2);
        return res;
    };

    auto t = amt::timer{};
    for(auto const& el : x){
        double const ops = el * el;
        auto sz = static_cast<std::size_t>(el);
        auto const M = lset(sz);
        auto const N = rset(sz);
        ub::matrix<ValueType> A(M,N);
        ub::vector<ValueType> v(N);
        ub::vector<ValueType> res(M);

        double st = amt::benchmark<MaxIter>(bench_fn, res, A, v);
        metric_data.update((ops / st));
    }
    t.stop();
    std::cerr<<fn_name<<" has completed! ( "<<t<<" )"<<std::endl;
}

template<typename ValueType, std::size_t MaxIter = 100ul>
void mkl_gemv(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    constexpr std::string_view fn_name = "intel MKL";
    
    auto& metric_data = m[fn_name];
    constexpr auto bench_fn = [](auto&&... args)
    {
        if constexpr(std::is_same_v<ValueType,float>){
            cblas_sgemv(std::forward<decltype(args)>(args)...);
        }else if constexpr(std::is_same_v<ValueType,double>){
            cblas_dgemv(std::forward<decltype(args)>(args)...);
        }
    };

    auto t = amt::timer{};  
    for(auto const& el : x){
        double const ops = el * el;
        auto sz = static_cast<std::size_t>(el);
        auto inc = static_cast<MKL_INT>(1);
        auto const M = static_cast<MKL_INT>(lset(sz));
        auto const N = static_cast<MKL_INT>(rset(sz));
        auto one = ValueType(1);
        ub::dynamic_tensor<ValueType> A(shape_t{static_cast<std::size_t>(M), static_cast<std::size_t>(N)}, one);
        ub::dynamic_tensor<ValueType> v(shape_t{1ul, static_cast<std::size_t>(N)}, one);
        ub::dynamic_tensor<ValueType> res(shape_t{1ul, static_cast<std::size_t>(M)});
        auto const* aptr = A.data();
        auto const* bptr = v.data();
        auto* cptr = res.data();
        auto lda = M;
        double st = amt::benchmark<MaxIter>(bench_fn, CblasColMajor, CblasNoTrans, M, N, one, aptr, lda, bptr, inc, one, cptr, inc);
        metric_data.update((ops / st));
    }
    t.stop();
    std::cerr<<fn_name<<" has completed! ( "<<t<<" )"<<std::endl;
}

#ifdef AMT_BENCHMARK_OPENBLAS_HPP
template<typename ValueType, std::size_t MaxIter = 100ul>
void openblas_gemv(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    constexpr std::string_view fn_name = "OpenBlas";
    
    auto& metric_data = m[fn_name];
    constexpr auto bench_fn = [](auto&&... args){
        amt::blas::mtv(std::forward<decltype(args)>(args)...);
    };

    auto t = amt::timer{};
    for(auto const& el : x){
        double const ops = el * el;
        auto sz = static_cast<std::size_t>(el);
        auto inc = static_cast<blasint>(1);
        auto const M = static_cast<blasint>(lset(sz));
        auto const N = static_cast<blasint>(rset(sz));
        auto one = ValueType(1);
        ub::dynamic_tensor<ValueType> A(shape_t{static_cast<std::size_t>(M), static_cast<std::size_t>(N)}, one);
        ub::dynamic_tensor<ValueType> v(shape_t{1ul, static_cast<std::size_t>(N)}, one);
        ub::dynamic_tensor<ValueType> res(shape_t{1ul, static_cast<std::size_t>(M)});
        auto const* aptr = A.data();
        auto const* bptr = v.data();
        auto cptr = res.data();
        auto lda = M;
        double st = amt::benchmark<MaxIter>(bench_fn, amt::blas::ColMajor, amt::blas::NoTrans, M, N, one, aptr, lda, bptr, inc, one, cptr, inc);
        metric_data.update((ops / st));
    }
    t.stop();
    std::cerr<<fn_name<<" has completed! ( "<<t<<" )"<<std::endl;
}
#endif

template<typename ValueType, std::size_t MaxIter = 100ul>
void openmp_gemv(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    constexpr std::string_view fn_name = "Boost.ublas.tensor";
    
    auto& metric_data = m[fn_name];

    constexpr auto bench_fn = [](auto&&... args){
        amt::mtv(std::forward<decltype(args)>(args)...);
    };

    auto t = amt::timer{};
    for(auto const& el : x){
        double const ops = el * el;
        auto sz = static_cast<std::size_t>(el);
        auto const M = lset(sz);
        auto const N = rset(sz);
        ub::dynamic_tensor<ValueType> A(shape_t{M, N},1.);
        ub::dynamic_tensor<ValueType> v(shape_t{1ul, N},1.);
        ub::dynamic_tensor<ValueType> res(shape_t{N});

        double st = amt::benchmark_timer_as_arg<MaxIter>(bench_fn, res, A, v, std::nullopt);
        amt::no_opt(res);
        metric_data.update((ops / st));
    }
    t.stop();
    std::cerr<<fn_name<<" has completed! ( "<<t<<" )"<<std::endl;
}

template<typename ValueType, std::size_t MaxIter = 100ul>
void blis_gemv(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    constexpr std::string_view fn_name = "Blis";
    
    auto& metric_data = m[fn_name];
    constexpr auto bench_fn = [](auto&&... args){
        if constexpr(std::is_same_v<ValueType,float>){
            bli_sgemv(std::forward<decltype(args)>(args)...);
        }else if constexpr(std::is_same_v<ValueType,double>){
            bli_dgemv(std::forward<decltype(args)>(args)...);
        }
    };

    auto t = amt::timer{};
    for(auto const& el : x){
        double const ops = el * el;
        auto sz = static_cast<std::size_t>(el);
        auto inc = static_cast<inc_t>(1);
        auto alpha = ValueType{1};
        auto one = ValueType(1);
        auto const M = static_cast<dim_t>(lset(sz));
        auto const N = static_cast<dim_t>(rset(sz));
        ub::dynamic_tensor<ValueType> A(shape_t{static_cast<std::size_t>(M), static_cast<std::size_t>(N)}, one);
        ub::dynamic_tensor<ValueType> v(shape_t{1ul, static_cast<std::size_t>(N)}, one);
        ub::dynamic_tensor<ValueType> res(shape_t{1ul, static_cast<std::size_t>(M)});
        auto* aptr = A.data();
        auto* bptr = v.data();
        auto* cptr = res.data();
        auto rsa = static_cast<inc_t>(A.strides()[0]);
        auto csa = static_cast<inc_t>(A.strides()[1]);

        double st = amt::benchmark<MaxIter>(bench_fn, 
            BLIS_NO_TRANSPOSE, BLIS_NO_CONJUGATE, 
            M, N, 
            &alpha, 
            aptr, rsa, csa,
            bptr, inc,
            &alpha,
            cptr, inc
        );
        metric_data.update((ops / st));
    }
    t.stop();
    std::cerr<<fn_name<<" has completed! ( "<<t<<" )"<<std::endl;
}

template<typename ValueType, std::size_t MaxIter = 100ul>
void eigen_gemv(std::vector<double> const& x, amt::metric<ValueType>& m){
    using namespace Eigen;
    using vector_type = Matrix<ValueType,-1,1,ColMajor>;
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    constexpr std::string_view fn_name = "Eigen";
    Eigen::initParallel();
    Eigen::setNbThreads(omp_get_max_threads());
    
    auto& metric_data = m[fn_name];

    constexpr auto bench_fn = [](auto& res, auto const& v1, auto const& v2){
        res.noalias() = v1 * v2;
    };

    auto t = amt::timer{};
    for(auto const& el : x){
        double const ops = el * el;
        auto sz = static_cast<std::size_t>(el);
        auto const M = lset(sz);
        auto const N = rset(sz);
        Matrix<ValueType,-1,-1> A(M,N);
        vector_type v(N), res(M);
        double st = amt::benchmark<MaxIter>(bench_fn, res, A, v);
        metric_data.update((ops / st));
    }
    t.stop();
    std::cerr<<fn_name<<" has completed! ( "<<t<<" )"<<std::endl;
}


void show_cache_info(std::ostream& os){
    os  << "\n\tL1: " << (amt::cache_manager::size(0) / 1024ul) << "KiB\n"
        << "\tL2: " << (amt::cache_manager::size(1) / 1024ul) << "KiB\n"
        << "\tL3: " << (amt::cache_manager::size(2) / (1024ul * 1024ul)) << "MiB\n";
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
    
    show_cache_info(std::cout);

    // using value_type = float;
    using value_type = double;
    
    std::vector<double> x;

    fixed_size = 1<<11;
    // is_lrect_matrix = true;
    // is_rrect_matrix = true;

    [[maybe_unused]]constexpr std::size_t max_iter = 4ul;
    // [[maybe_unused]]constexpr double max_value = 16382;//8 * 1024;
    // amt::range(x, 32., max_value, 32., std::plus<>{});
    [[maybe_unused]]constexpr double max_value = 1<<16;
    amt::range(x, 2., max_value, 2., std::multiplies<>{});

    auto m = amt::metric<value_type>(x.size());

    ublas_gemv<value_type,max_iter>(x,m);
    openblas_gemv<value_type,max_iter>(x,m);
    blis_gemv<value_type,max_iter>(x,m);
    mkl_gemv<value_type,max_iter>(x,m);
    openmp_gemv<value_type,max_iter>(x,m);
    eigen_gemv<value_type,max_iter>(x,m);
    // std::cout<<m.tail();

    constexpr std::string_view comp_name = "tensor";

    constexpr std::string_view plot_xlable = "Size [n = m]" SIZE_SUFFIX;
    std::transform(x.begin(), x.end(), x.begin(), [](auto sz){
        double lsz = static_cast<double>(lset(static_cast<std::size_t>(sz)));
        // double rsz = static_cast<double>(rset(static_cast<std::size_t>(sz)));
        return size_conv(lsz);
    });

    // std::cout<<m.tail()<<'\n';
    std::cout<<m.str(comp_name)<<'\n';
    #ifndef DISABLE_PLOT
        #if !defined(SPEEDUP_PLOT) || defined(PLOT_ALL)
            m.plot(x, "Performance of Boost.uBLAS.Tensor for the gemv-operation [iter=4]", plot_xlable);
            m.plot_per("Sorted performance of Boost.uBLAS.Tensor for the gemv-operation [iter=4]");
        #endif
        
        #if defined(SPEEDUP_PLOT) || defined(PLOT_ALL)
            m.plot_speedup(comp_name,x,"Speedup of Boost.uBLAS.Tensor for the gemv-operation [iter=4]", plot_xlable);
            auto inter_pts = m.plot_speedup_per<true>(comp_name,"Sorted speedup of Boost.uBLAS.Tensor for the gemv-operation [iter=4]");
            m.plot_speedup_semilogy<true>(comp_name,x,"Semilogy speedup of Boost.uBLAS.Tensor for the gemv-operation [iter=4]", plot_xlable);
            amt::show_intersection_pts(std::cout,inter_pts);
        #endif
    #endif
    // m.raw();
    return 0;

}
