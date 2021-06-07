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


template<typename ValueType, typename LayoutType, std::size_t MaxIter = 100ul>
std::string_view ublas_gemv(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );

    constexpr std::string_view fn_name = "Boost.ublas";
    
    auto& metric_data = m[fn_name];
    constexpr auto bench_fn = [](auto& res, auto const& v1, auto const& v2) -> auto&{
        ub::noalias(res) = ub::prod(v1,v2);
        return res;
    };
    
    auto t = amt::timer{};
    defer(t.start(),t.stop()){
        for(auto const& el : x){
            auto sz = static_cast<std::size_t>(el);
            auto const M = lset(sz);
            auto const N = rset(sz);
            double const ops = static_cast<double>(M * (2 * N - 1));
            ub::matrix<ValueType,LayoutType> A(M,N);
            ub::vector<ValueType> v(N);
            ub::vector<ValueType> res(M);

            double st = amt::benchmark<MaxIter>(bench_fn, res, A, v);
            metric_data.update((ops / st));
        }
    }
    std::cerr<<fn_name<<" has completed! ( "<<t<<" )"<<std::endl;
    return "ublas.csv";
}

template<typename ValueType, typename LayoutType, std::size_t MaxIter = 100ul>
std::string_view mkl_gemv(std::vector<double> const& x, amt::metric<ValueType>& m){
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

    constexpr auto mkl_layout = std::is_same_v<ub::layout::first_order,LayoutType> ? CblasColMajor : CblasRowMajor;

    auto t = amt::timer{};  
    defer(t.start(),t.stop()){
        for(auto const& el : x){
            auto sz = static_cast<std::size_t>(el);
            auto const M = static_cast<MKL_INT>(lset(sz));
            auto const N = static_cast<MKL_INT>(rset(sz));
            double const ops = static_cast<double>(M * (2 * N - 1));
            auto inc = static_cast<MKL_INT>(1);
            auto one = ValueType(1);
            auto A = amt::make_tensor<ValueType,LayoutType>(M,N,one);
            auto v = amt::make_tensor<ValueType,LayoutType>(1ul,N,one);
            auto res = amt::make_tensor<ValueType,LayoutType>(1ul,M);
            auto const* aptr = A.data();
            auto const* bptr = v.data();
            auto* cptr = res.data();
            auto lda = static_cast<MKL_INT>(std::max(A.strides()[0], A.strides()[1]));
            double st = amt::benchmark<MaxIter>(bench_fn, mkl_layout, CblasNoTrans, M, N, one, aptr, lda, bptr, inc, one, cptr, inc);
            metric_data.update((ops / st));
        }
    }
    std::cerr<<fn_name<<" has completed! ( "<<t<<" )"<<std::endl;
    return "mkl.csv";
}

#ifdef AMT_BENCHMARK_OPENBLAS_HPP
template<typename ValueType, typename LayoutType, std::size_t MaxIter = 100ul>
std::string_view openblas_gemv(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    constexpr std::string_view fn_name = "OpenBlas";
    
    auto& metric_data = m[fn_name];
    constexpr auto bench_fn = [](auto&&... args){
        amt::blas::mtv(std::forward<decltype(args)>(args)...);
    };

    openblas_set_num_threads(omp_get_max_threads());

    constexpr auto open_layout = std::is_same_v<ub::layout::first_order,LayoutType> ? amt::blas::ColMajor : amt::blas::RowMajor;

    auto t = amt::timer{};
    defer(t.start(),t.stop()){
        for(auto const& el : x){
            auto sz = static_cast<std::size_t>(el);
            auto const M = static_cast<blasint>(lset(sz));
            auto const N = static_cast<blasint>(rset(sz));
            double const ops = static_cast<double>(M * (2 * N - 1));
            auto inc = static_cast<blasint>(1);
            auto one = ValueType(1);
            auto A = amt::make_tensor<ValueType,LayoutType>(M,N,one);
            auto v = amt::make_tensor<ValueType,LayoutType>(1ul,N,one);
            auto res = amt::make_tensor<ValueType,LayoutType>(1ul,M);
            auto const* aptr = A.data();
            auto const* bptr = v.data();
            auto cptr = res.data();
            auto lda = static_cast<blasint>(std::max(A.strides()[0], A.strides()[1]));
            double st = amt::benchmark<MaxIter>(bench_fn, open_layout, amt::blas::NoTrans, M, N, one, aptr, lda, bptr, inc, one, cptr, inc);
            metric_data.update((ops / st));
        }
    }
    std::cerr<<fn_name<<" has completed! ( "<<t<<" )"<<std::endl;
    return "openblas.csv";
}
#endif

template<typename ValueType, typename LayoutType, std::size_t MaxIter = 100ul>
std::string_view openmp_gemv(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    constexpr std::string_view fn_name = "Boost.ublas.tensor";
    
    auto& metric_data = m[fn_name];

    auto t = amt::timer{};
    defer(t.start(),t.stop()){
        for(auto const& el : x){
            auto sz = static_cast<std::size_t>(el);
            auto const M = lset(sz);
            auto const N = rset(sz);
            double const ops = static_cast<double>(M * (2 * N - 1));
            auto A = amt::make_tensor<ValueType,LayoutType>(M,N,1.);
            auto v = amt::make_tensor<ValueType,LayoutType>(1ul,N,1.);
            auto res = amt::make_tensor<ValueType,LayoutType>(1ul,M);
            // std::iota(A.begin(), A.end(), 1.);
            // std::iota(v.begin(), v.end(), 1.);
            auto bench_fn = amt::mtv(res, A, v, std::nullopt);
            double st = amt::benchmark<MaxIter>(std::move(bench_fn));
            amt::no_opt(res);
            metric_data.update((ops / st));
        }
    }
    std::cerr<<fn_name<<" has completed! ( "<<t<<" )"<<std::endl;
    return "tensor.csv";
}

template<typename ValueType, typename LayoutType, std::size_t MaxIter = 100ul>
std::string_view blis_gemv(std::vector<double> const& x, amt::metric<ValueType>& m){
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
    defer(t.start(),t.stop()){
        for(auto const& el : x){
            auto sz = static_cast<std::size_t>(el);
            auto inc = static_cast<inc_t>(1);
            auto alpha = ValueType{1};
            auto one = ValueType(1);
            auto const M = static_cast<dim_t>(lset(sz));
            auto const N = static_cast<dim_t>(rset(sz));
            double const ops = static_cast<double>(M * (2 * N - 1));
            auto A = amt::make_tensor<ValueType,LayoutType>(M,N,one);
            auto v = amt::make_tensor<ValueType,LayoutType>(1ul,N,one);
            auto res = amt::make_tensor<ValueType,LayoutType>(1ul,M);
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
    }
    std::cerr<<fn_name<<" has completed! ( "<<t<<" )"<<std::endl;
    return "blis.csv";
}

template<typename ValueType, typename LayoutType, std::size_t MaxIter = 100ul>
std::string_view eigen_gemv(std::vector<double> const& x, amt::metric<ValueType>& m){
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

    constexpr auto eigen_layout = std::is_same_v<ub::layout::first_order,LayoutType> ?  ColMajor : RowMajor;

    auto t = amt::timer{};
    defer(t.start(),t.stop()){
        for(auto const& el : x){
            auto sz = static_cast<std::size_t>(el);
            auto const M = lset(sz);
            auto const N = rset(sz);
            double const ops = static_cast<double>(M * (2 * N - 1));
            Matrix<ValueType,-1,-1,eigen_layout> A(M,N);
            vector_type v(N), res(M);
            double st = amt::benchmark<MaxIter>(bench_fn, res, A, v);
            metric_data.update((ops / st));
        }
    }
    std::cerr<<fn_name<<" has completed! ( "<<t<<" )"<<std::endl;
    return "eigen.csv";
}


void show_cache_info(std::ostream& os){
    os  << "\n\tL1: " << (amt::cache_manager::size(0) / 1024ul) << "KiB\n"
        << "\tL2: " << (amt::cache_manager::size(1) / 1024ul) << "KiB\n"
        << "\tL3: " << (amt::cache_manager::size(2) / (1024ul * 1024ul)) << "MiB\n";
}

// #define DISABLE_PLOT
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
    
    show_cache_info(std::cout);

    // using layout_t = ub::layout::last_order;
    using layout_t = ub::layout::first_order;

    using value_type = float;
    // using value_type = double;
    
    std::vector<double> x;

    fixed_size = 1<<11;
    // is_lrect_matrix = true;
    // is_rrect_matrix = true;

    [[maybe_unused]]constexpr std::size_t max_iter = 10ul;
    // [[maybe_unused]]constexpr double max_value = 1<<12;//16382ul;
    // amt::range(x, 32., max_value, 32., std::plus<>{});
    [[maybe_unused]]constexpr double max_value = 1<<16;
    amt::range(x, 2., max_value, 2., std::multiplies<>{});

    auto m = amt::metric<value_type>(x.size());
    [[maybe_unused]] std::string_view fn_name;
    // fn_name = ublas_gemv<value_type,layout_t,1>(x,m);
    // fn_name = openblas_gemv<value_type,layout_t,max_iter>(x,m);
    // fn_name = blis_gemv<value_type,layout_t,max_iter>(x,m);
    fn_name = mkl_gemv<value_type,layout_t,max_iter>(x,m);
    fn_name = openmp_gemv<value_type,layout_t,max_iter>(x,m);
    // fn_name = eigen_gemv<value_type,layout_t,max_iter>(x,m);
    // std::cout<<m.tail();

    constexpr std::string_view comp_name = "tensor";

    std::transform(x.begin(), x.end(), x.begin(), [](auto sz){
        double lsz = static_cast<double>(lset(static_cast<std::size_t>(sz)));
        // double rsz = static_cast<double>(rset(static_cast<std::size_t>(sz)));
        return size_conv(lsz);
    });

    // std::cout<<m.tail()<<'\n';
    // m.csv(fn_name);
    std::cout<<m.str(comp_name)<<'\n';
    #ifndef DISABLE_PLOT
        constexpr std::string_view plot_xlable = "Size [n = m], " SIZE_SUFFIX " iterating";
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
    return 0;

}
