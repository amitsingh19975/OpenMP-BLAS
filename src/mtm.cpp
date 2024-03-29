#define BOOST_UBLAS_USE_SIMD
#include <iostream>
#include <numeric>
#include <algorithm>
#include <vector>
#include <cassert>
#include <benchmark.hpp>
#include <metric.hpp>
#include <mtm.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
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
static bool is_Mrect_matrix = false;
static bool is_Nrect_matrix = false;
static bool is_Krect_matrix = false;

auto Mset(std::size_t l) noexcept{
    return is_Mrect_matrix ? fixed_size : l;
}

auto Nset(std::size_t r) noexcept{
    return is_Nrect_matrix ? fixed_size : r;
}

auto Kset(std::size_t k) noexcept{
    return is_Krect_matrix ? fixed_size : k;
}

std::string xlable(std::string_view prefix = "Size"){
    std::stringstream ss;
    if(is_Mrect_matrix){
        ss << prefix << "[n,k]( m = " << fixed_size << " )";
    }else if(is_Nrect_matrix){
        ss << prefix << "[m,k]( n = " << fixed_size << " )";
    }else if(is_Krect_matrix){
        ss << prefix << "[m,n]( k = " << fixed_size << " )";
    }else{
        ss << prefix << "( m = n = k )";
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
void ublas_gemm(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );

    constexpr std::string_view fn_name = "Boost.ublas";
    
    auto& metric_data = m[fn_name];
    constexpr auto bench_fn = [](auto& res, auto const& A, auto const& B) -> auto&{
        ub::noalias(res) = ub::prod(A,B);
        return res;
    };
    
    auto t = amt::timer{};
    for(auto const& el : x){
        auto sz = static_cast<std::size_t>(el);
        auto const M = Mset(sz);
        auto const N = Nset(sz);
        auto const K = Kset(sz);
        double const ops = static_cast<double>(M) * static_cast<double>(N) * (2. * static_cast<double>(K) - 1.);
        ub::matrix<ValueType,LayoutType> A(M,K);
        ub::matrix<ValueType,LayoutType> B(K,N);
        ub::matrix<ValueType,LayoutType> res(M,N);

        double st = amt::benchmark<MaxIter>(bench_fn, res, A, A);
        metric_data.update((ops / st));
    }
    t.stop();
    std::cerr<<fn_name<<" has completed! ( "<<t<<" )"<<std::endl;
}

template<typename ValueType, typename LayoutType, std::size_t MaxIter = 100ul>
void mkl_gemm(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    constexpr std::string_view fn_name = "intel MKL";
    
    auto& metric_data = m[fn_name];
    constexpr auto bench_fn = [](auto&&... args)
    {
        if constexpr(std::is_same_v<ValueType,float>){
            cblas_sgemm(std::forward<decltype(args)>(args)...);
        }else if constexpr(std::is_same_v<ValueType,double>){
            cblas_dgemm(std::forward<decltype(args)>(args)...);
        }
    };

    constexpr auto mkl_layout = std::is_same_v<ub::layout::first_order,LayoutType> ? CblasColMajor : CblasRowMajor;

    auto t = amt::timer{};  
    for(auto const& el : x){
        auto sz = static_cast<std::size_t>(el);
        auto const M = static_cast<MKL_INT>(Mset(sz));
        auto const N = static_cast<MKL_INT>(Nset(sz));
        auto const K = static_cast<MKL_INT>(Kset(sz));
        double const ops = static_cast<double>(M) * static_cast<double>(N) * (2. * static_cast<double>(K) - 1.);
        auto one = ValueType(1);
        auto A = amt::make_tensor<ValueType,LayoutType>(M,K,1.);
        auto B = amt::make_tensor<ValueType,LayoutType>(K,N,1.);
        auto res = amt::make_tensor<ValueType,LayoutType>(M,N);
        auto* aptr = A.data();
        auto* bptr = B.data();
        auto* cptr = res.data();
        auto lda = static_cast<MKL_INT>(std::max(A.strides()[0], A.strides()[1]));
        auto ldb = static_cast<MKL_INT>(std::max(B.strides()[0], B.strides()[1]));
        auto ldc = static_cast<MKL_INT>(std::max(res.strides()[0], res.strides()[1]));
        double st = amt::benchmark<MaxIter>(bench_fn, 
            mkl_layout, CblasNoTrans, CblasNoTrans, 
            M, N, K, one, aptr, lda, bptr, ldb, ValueType(0), cptr, ldc
        );
        metric_data.update((ops / st));

    }
    t.stop();
    std::cerr<<fn_name<<" has completed! ( "<<t<<" )"<<std::endl;
    // exit(0);
}

#ifdef AMT_BENCHMARK_OPENBLAS_HPP
template<typename ValueType, typename LayoutType, std::size_t MaxIter = 100ul>
void openblas_gemm(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    constexpr std::string_view fn_name = "OpenBlas";
    
    auto& metric_data = m[fn_name];
    constexpr auto bench_fn = [](auto&&... args){
        amt::blas::mtm(std::forward<decltype(args)>(args)...);
    };

    openblas_set_num_threads(omp_get_max_threads());

    constexpr auto open_layout = std::is_same_v<ub::layout::first_order,LayoutType> ? amt::blas::ColMajor : amt::blas::RowMajor;

    auto t = amt::timer{};
    for(auto const& el : x){
        auto sz = static_cast<std::size_t>(el);
        auto const M = static_cast<blasint>(Mset(sz));
        auto const N = static_cast<blasint>(Nset(sz));
        auto const K = static_cast<blasint>(Kset(sz));
        double const ops = static_cast<double>(M) * static_cast<double>(N) * (2. * static_cast<double>(K) - 1.);
        auto one = ValueType(1);
        auto A = amt::make_tensor<ValueType,LayoutType>(M,K,1.);
        auto B = amt::make_tensor<ValueType,LayoutType>(K,N,1.);
        auto res = amt::make_tensor<ValueType,LayoutType>(M,N);

        auto const* aptr = A.data();
        auto const* bptr = B.data();
        auto cptr = res.data();
        auto lda = static_cast<blasint>(std::max(A.strides()[0], A.strides()[1]));
        auto ldb = static_cast<blasint>(std::max(B.strides()[0], B.strides()[1]));
        auto ldc = static_cast<blasint>(std::max(res.strides()[0], res.strides()[1]));
        // const ORDER Order, const TRANSPOSE TransA, const TRANSPOSE TransB, const blasint M, const blasint N, const blasint K,
        //     const ValueType alpha, const ValueType *A, const blasint lda, const ValueType *B, const blasint ldb, const ValueType beta, ValueType *C, const blasint ldc
        double st = amt::benchmark<MaxIter>(bench_fn, 
            open_layout, amt::blas::NoTrans, amt::blas::NoTrans,
            M, N, K, one, aptr, lda, bptr, ldb, ValueType(0), cptr, ldc
        );
        metric_data.update((ops / st));
    }
    t.stop();
    std::cerr<<fn_name<<" has completed! ( "<<t<<" )"<<std::endl;
}
#endif

template<typename ValueType, typename LayoutType, std::size_t MaxIter = 100ul>
void openmp_gemm(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    constexpr std::string_view fn_name = "Boost.ublas.tensor";
    
    auto& metric_data = m[fn_name];

    auto t = amt::timer{};
    for(auto const& el : x){
        auto sz = static_cast<std::size_t>(el);
        auto const M = Mset(sz);
        auto const N = Nset(sz);
        auto const K = Kset(sz);
        double const ops = static_cast<double>(M) * static_cast<double>(N) * (2. * static_cast<double>(K) - 1.);
        auto A = amt::make_tensor<ValueType,LayoutType>(M,K,1.);
        auto B = amt::make_tensor<ValueType,LayoutType>(K,N,1.);
        auto res = amt::make_tensor<ValueType,LayoutType>(M,N);
        auto bench_fn = amt::mtm(res, A, B, std::nullopt);
        double st = amt::benchmark<MaxIter>(std::move(bench_fn));
        amt::no_opt(res);
        metric_data.update((ops / st));
    }
    t.stop();
    std::cerr<<fn_name<<" has completed! ( "<<t<<" )"<<std::endl;
}

template<typename ValueType, typename LayoutType, std::size_t MaxIter = 100ul>
void openmp_gemm(std::vector<double> const& x, amt::metric<ValueType>& m, std::size_t M){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    constexpr std::string_view fn_name = "Boost.ublas.tensor";
    
    auto& metric_data = m[fn_name];

    auto t = amt::timer{};
    for(auto const& el : x){
        auto sz = static_cast<std::size_t>(el);
        // auto const M = Mset(sz);
        auto const N = M;//Nset(sz);
        auto const K = M;//Kset(sz);
        double const ops = static_cast<double>(M) * static_cast<double>(N) * (2. * static_cast<double>(K) - 1.);
        auto A = amt::make_tensor<ValueType,LayoutType>(M,K,1.);
        auto B = amt::make_tensor<ValueType,LayoutType>(K,N,1.);
        auto res = amt::make_tensor<ValueType,LayoutType>(M,N);
        auto bench_fn = amt::mtm(res, A, B, std::nullopt,sz);
        double st = amt::benchmark<MaxIter>(std::move(bench_fn));
        amt::no_opt(res);
        metric_data.update((ops / st));
    }
    t.stop();
    std::cerr<<fn_name<<" has completed! ( "<<t<<" )"<<std::endl;
}

template<typename ValueType, typename LayoutType, std::size_t MaxIter = 100ul>
void blis_gemm(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    constexpr std::string_view fn_name = "Blis";
    
    auto& metric_data = m[fn_name];
    constexpr auto bench_fn = [](auto&&... args){
        if constexpr(std::is_same_v<ValueType,float>){
            bli_sgemm(std::forward<decltype(args)>(args)...);
        }else if constexpr(std::is_same_v<ValueType,double>){
            bli_dgemm(std::forward<decltype(args)>(args)...);
        }
    };

    auto t = amt::timer{};
    for(auto const& el : x){
        auto sz = static_cast<std::size_t>(el);
        auto alpha = ValueType{1};
        auto const M = static_cast<dim_t>(Mset(sz));
        auto const N = static_cast<dim_t>(Nset(sz));
        auto const K = static_cast<dim_t>(Kset(sz));
        double const ops = static_cast<double>(M) * static_cast<double>(N) * (2. * static_cast<double>(K) - 1.);
        auto A = amt::make_tensor<ValueType,LayoutType>(M,K,1.);
        auto B = amt::make_tensor<ValueType,LayoutType>(K,N,1.);
        auto res = amt::make_tensor<ValueType,LayoutType>(M,N);
        auto* aptr = A.data();
        auto* bptr = B.data();
        auto* cptr = res.data();
        auto rsa = static_cast<inc_t>(A.strides()[0]);
        auto csa = static_cast<inc_t>(A.strides()[1]);
        auto rsb = static_cast<inc_t>(B.strides()[0]);
        auto csb = static_cast<inc_t>(B.strides()[1]);
        auto rsc = static_cast<inc_t>(res.strides()[0]);
        auto csc = static_cast<inc_t>(res.strides()[1]);

        double st = amt::benchmark<MaxIter>(bench_fn, 
            BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, 
            M, N, K,
            &alpha, 
            aptr, rsa, csa,
            bptr, rsb, csb,
            &alpha,
            cptr, rsc, csc
        );
        metric_data.update((ops / st));
    }
    t.stop();
    std::cerr<<fn_name<<" has completed! ( "<<t<<" )"<<std::endl;
}

template<typename ValueType, typename LayoutType, std::size_t MaxIter = 100ul>
void eigen_gemm(std::vector<double> const& x, amt::metric<ValueType>& m){
    using namespace Eigen;
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
    for(auto const& el : x){
        auto sz = static_cast<std::size_t>(el);
        auto const M = Mset(sz);
        auto const N = Nset(sz);
        auto const K = Kset(sz);
        double const ops = static_cast<double>(M) * static_cast<double>(N) * (2. * static_cast<double>(K) - 1.);
        Matrix<ValueType,-1,-1,eigen_layout> A(M,K);
        Matrix<ValueType,-1,-1,eigen_layout> B(K,N);
        A.setOnes();
        B.setOnes();
        Matrix<ValueType,-1,-1,eigen_layout> res(M,N);
        double st = amt::benchmark<MaxIter>(bench_fn, res, A, B);
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
#define SIZE_KiB true


#ifdef SIZE_KiB
    #define SIZE_SUFFIX "[Ki]"
#else
    #define SIZE_KiB false
    #define SIZE_SUFFIX "[Mi]"
#endif

constexpr double size_conv(double val) noexcept{ 
    if constexpr( SIZE_KiB )
        return KiB(val);
    else
        return MiB(val);
}

int main(){
    
    show_cache_info(std::cout);

    using layout_t = ub::layout::first_order;
    // using layout_t = ub::layout::last_order;

    using value_type = float;
    // using value_type = double;
    
    std::vector<double> x;

    fixed_size = 1<<11;
    // is_lrect_matrix = true;
    // is_rrect_matrix = true;

    [[maybe_unused]]constexpr std::size_t max_iter = 4ul;
    [[maybe_unused]]constexpr double max_value = 3 * 1024;
    constexpr double sz = 32;
    amt::range(x, 32., max_value, sz, std::plus<>{});
    // [[maybe_unused]]constexpr double max_value = 1<<14;
    // amt::range(x, 2., max_value, 2., std::multiplies<>{});

    auto m = amt::metric<value_type>(x.size());

    // ublas_gemm<value_type,layout_t,1ul>(x,m);
    // openblas_gemm<value_type,layout_t,max_iter>(x,m);
    // blis_gemm<value_type,layout_t,max_iter>(x,m);
    // mkl_gemm<value_type,layout_t,max_iter>(x,m);
    openmp_gemm<value_type,layout_t,max_iter>(x,m);
    sleep(5);
    eigen_gemm<value_type,layout_t,max_iter>(x,m);
    // std::cout<<m.tail();

    constexpr std::string_view comp_name = "tensor";

    // std::transform(x.begin(), x.end(), x.begin(), [](auto sz){
    //     double msz = static_cast<double>(Mset(static_cast<std::size_t>(sz)));
    //     // double rsz = static_cast<double>(rset(static_cast<std::size_t>(sz)));
    //     return size_conv(msz);
    // });

    // std::cout<<m.tail()<<'\n';
    std::cout<<m.str(comp_name)<<'\n';
    m.csv("tensor.csv");
    #ifndef DISABLE_PLOT
        constexpr std::string_view plot_xlable = "Size [n = m = n], " SIZE_SUFFIX " iterating";
        #if !defined(SPEEDUP_PLOT) || defined(PLOT_ALL)
            m.plot(x, "Performance of Boost.uBLAS.Tensor for the gemm-operation [iter=4]", plot_xlable);
            m.plot_per("Sorted performance of Boost.uBLAS.Tensor for the gemm-operation [iter=4]");
        #endif
        
        #if defined(SPEEDUP_PLOT) || defined(PLOT_ALL)
            m.plot_speedup(comp_name,x,"Speedup of Boost.uBLAS.Tensor for the gemm-operation [iter=4]", plot_xlable);
            auto inter_pts = m.plot_speedup_per<true>(comp_name,"Sorted speedup of Boost.uBLAS.Tensor for the gemm-operation [iter=4]");
            m.plot_speedup_semilogy<true>(comp_name,x,"Semilogy speedup of Boost.uBLAS.Tensor for the gemm-operation [iter=4]", plot_xlable);
            amt::show_intersection_pts(std::cout,inter_pts);
        #endif
    #endif
    return 0;

}
