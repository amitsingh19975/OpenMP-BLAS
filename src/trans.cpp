#define BOOST_UBLAS_USE_SIMD
#include <iostream>
#include <numeric>
#include <algorithm>
#include <vector>
#include <cassert>
#include <benchmark.hpp>
#include <metric.hpp>
#include <trans.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <mkl_trans.h>
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
template<typename T, typename L>
using tensor_t = ub::dynamic_tensor<T,L>;


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
void ublas_transpose(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );

    constexpr std::string_view fn_name = "Boost.ublas";
    
    auto& metric_data = m[fn_name];
    constexpr auto bench_fn = [](auto& res, auto const& A) -> auto&{
        ub::noalias(res) = ub::trans(A);
        return res;
    };
    
    auto t = amt::timer{};
    for(auto const& el : x){
        auto sz = static_cast<std::size_t>(el);
        auto const M = Mset(sz);
        auto const N = Nset(sz);
        double const ops = static_cast<double>(M) * static_cast<double>(N);
        ub::matrix<ValueType,LayoutType> A(M,N);
        ub::matrix<ValueType,LayoutType> res(N,M);

        double st = amt::benchmark<MaxIter>(bench_fn, res, A);
        metric_data.update((ops / st));
    }
    t.stop();
    std::cerr<<fn_name<<" has completed! ( "<<t<<" )"<<std::endl;
}

template<typename ValueType, typename LayoutType, std::size_t MaxIter = 100ul>
void mkl_transpose(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    constexpr std::string_view fn_name = "intel MKL";
    
    auto& metric_data = m[fn_name];
    constexpr auto bench_fn = [](
        const char ordering, const char trans,
        std::size_t rows, std::size_t cols,
        const auto alpha,
        auto* A, std::size_t lda, 
        auto* B, std::size_t ldb
    ){
        if constexpr(std::is_same_v<ValueType,float>){
            mkl_somatcopy(ordering, trans, rows, cols, alpha, A, lda, B, ldb);
        }else if constexpr(std::is_same_v<ValueType,double>){
            mkl_domatcopy(ordering, trans, rows, cols, alpha, A, lda, B, ldb);
        }
    };

    constexpr auto mkl_layout = std::is_same_v<ub::layout::first_order,LayoutType> ? 'C' : 'R';

    auto t = amt::timer{};  
    for(auto const& el : x){
        auto sz = static_cast<std::size_t>(el);
        auto one = ValueType(1);
        auto const M = Mset(sz);
        auto const N = Nset(sz);
        double const ops = static_cast<double>(M) * static_cast<double>(N);
        tensor_t<ValueType,LayoutType> A(shape_t{static_cast<std::size_t>(M), static_cast<std::size_t>(N)}, one);
        tensor_t<ValueType,LayoutType> res(shape_t{static_cast<std::size_t>(N), static_cast<std::size_t>(M)});
        // std::iota(A.begin(), A.end(),1);
        // std::cerr<<A<<'\n';
        auto* cptr = res.data();
        auto const* aptr = A.data();
        auto lda = std::max(A.strides()[0], A.strides()[1]);
        auto ldc = std::max(res.strides()[0], res.strides()[1]);
        double st = amt::benchmark<MaxIter>(bench_fn, mkl_layout, 'T', M, N, one, aptr, lda, cptr, ldc);
        amt::no_opt(res);
        // std::cerr<<res<<'\n'; exit(0);
        metric_data.update((ops / st));
    }
    t.stop();
    std::cerr<<fn_name<<" has completed! ( "<<t<<" )"<<std::endl;
    // exit(0);
}

template<typename ValueType, typename LayoutType, std::size_t MaxIter = 100ul>
void openmp_transpose(std::vector<double> const& x, amt::metric<ValueType>& m, amt::tag::outplace){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    constexpr std::string_view fn_name = "Boost.ublas.tensor";
    
    auto& metric_data = m[fn_name];

    auto t = amt::timer{};
    for(auto const& el : x){
        auto sz = static_cast<std::size_t>(el);
        auto const M = Mset(sz);
        auto const N = Nset(sz);
        double const ops = static_cast<double>(M) * static_cast<double>(N);
        tensor_t<ValueType,LayoutType> A(shape_t{M, N},1.);
        [[maybe_unused]] tensor_t<ValueType,LayoutType> res(shape_t{N, M});
        // std::iota(A.begin(), A.end(),1);
        // std::cerr<<A<<'\n';
        auto bench_fn = amt::transpose(res, A, std::nullopt);
        double st = amt::benchmark<MaxIter>(std::move(bench_fn));
        amt::no_opt(res);
        // std::cerr<<res<<'\n';exit(0);
        metric_data.update((ops / st));
    }
    t.stop();
    std::cerr<<fn_name<<" has completed! ( "<<t<<" )"<<std::endl;
}

template<typename ValueType, typename LayoutType, std::size_t MaxIter = 100ul>
void openmp_transpose(std::vector<double> const& x, amt::metric<ValueType>& m, amt::tag::inplace){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    constexpr std::string_view fn_name = "Boost.ublas.tensor";
    
    auto& metric_data = m[fn_name];

    auto t = amt::timer{};
    for(auto const& el : x){
        auto sz = static_cast<std::size_t>(el);
        auto const M = Mset(sz);
        auto const N = Nset(sz);
        double const ops = static_cast<double>(M) * static_cast<double>(N);
        tensor_t<ValueType,LayoutType> A(shape_t{M, N},1.);
        // std::iota(A.begin(), A.end(),1);
        // std::cerr<<A<<'\n';
        auto bench_fn = amt::transpose(A, std::nullopt);
        double st = amt::benchmark<MaxIter>(std::move(bench_fn));
        amt::no_opt(A);
        // std::cerr<<res<<'\n';exit(0);
        metric_data.update((ops / st));
    }
    t.stop();
    std::cerr<<fn_name<<" has completed! ( "<<t<<" )"<<std::endl;
}

template<typename ValueType, typename LayoutType, std::size_t MaxIter = 100ul>
void eigen_transpose(std::vector<double> const& x, amt::metric<ValueType>& m){
    using namespace Eigen;
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    constexpr std::string_view fn_name = "Eigen";
    Eigen::initParallel();
    Eigen::setNbThreads(omp_get_max_threads());
    
    auto& metric_data = m[fn_name];

    constexpr auto bench_fn = [](auto& res, auto const& A){
        res.noalias() = A.transpose();
    };

    constexpr auto eigen_layout = std::is_same_v<ub::layout::first_order,LayoutType> ?  ColMajor : RowMajor;

    auto t = amt::timer{};
    for(auto const& el : x){
        auto sz = static_cast<std::size_t>(el);
        auto const M = Mset(sz);
        auto const N = Nset(sz);
        double const ops = static_cast<double>(M) * static_cast<double>(N);
        Matrix<ValueType,-1,-1,eigen_layout> A(M,N);
        Matrix<ValueType,-1,-1,eigen_layout> res(N,M);
        A.setOnes();
        double st = amt::benchmark<MaxIter>(bench_fn, res, A);
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
    [[maybe_unused]]constexpr double max_value = 4 * 1024;
    constexpr double sz = 32;
    amt::range(x, 32., max_value, sz, std::plus<>{});
    // [[maybe_unused]]constexpr double max_value = 1<<14;
    // amt::range(x, 2., max_value, 2., std::multiplies<>{});

    auto m = amt::metric<value_type>(x.size());

    ublas_transpose<value_type,layout_t,max_iter>(x,m);
    openmp_transpose<value_type,layout_t,max_iter>(x,m,amt::tag::outplace{});
    mkl_transpose<value_type,layout_t,max_iter>(x,m);
    eigen_transpose<value_type,layout_t,max_iter>(x,m);
    // std::cout<<m.tail();

    constexpr std::string_view comp_name = "tensor";

    std::transform(x.begin(), x.end(), x.begin(), [](auto sz){
        double msz = static_cast<double>(Mset(static_cast<std::size_t>(sz)));
        // double rsz = static_cast<double>(rset(static_cast<std::size_t>(sz)));
        return size_conv(msz);
    });

    // std::cout<<m.tail()<<'\n';
    std::cout<<m.str(comp_name)<<'\n';
    // m.csv("tensor.csv");

    #ifndef DISABLE_PLOT
        constexpr std::string_view plot_xlable = "Size [n = m = n], " SIZE_SUFFIX " iterating";
        #if !defined(SPEEDUP_PLOT) || defined(PLOT_ALL)
            m.plot(x, "Performance of Boost.uBLAS.Tensor for the transpose-operation [iter=4]", plot_xlable);
            m.plot_per("Sorted performance of Boost.uBLAS.Tensor for the transpose-operation [iter=4]");
        #endif
        
        #if defined(SPEEDUP_PLOT) || defined(PLOT_ALL)
            m.plot_speedup(comp_name,x,"Speedup of Boost.uBLAS.Tensor for the transpose-operation [iter=4]", plot_xlable);
            auto inter_pts = m.plot_speedup_per<true>(comp_name,"Sorted speedup of Boost.uBLAS.Tensor for the transpose-operation [iter=4]");
            m.plot_speedup_semilogy<true>(comp_name,x,"Semilogy speedup of Boost.uBLAS.Tensor for the transpose-operation [iter=4]", plot_xlable);
            amt::show_intersection_pts(std::cout,inter_pts);
        #endif
    #endif
    return 0;

}
