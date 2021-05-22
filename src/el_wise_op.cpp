#define BOOST_UBLAS_USE_SIMD
#include <iostream>
#include <numeric>
#include <algorithm>
#include <vector>
#include <cassert>
#include <benchmark.hpp>
#include <metric.hpp>
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
#include <cache_manager.hpp>
#include <mtm.hpp>
#include <el_wise_op.hpp>

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

template<typename ValueType, typename LayoutType, std::size_t MaxIter = 100ul>
void ublas_el_wise_op(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );

    constexpr std::string_view fn_name = "uBLAS Expr Template";
    
    auto& metric_data = m[fn_name];
    constexpr auto bench_fn = [](auto& res, auto const& A, auto const& B){
        ub::noalias(res) = ub::element_prod(A,B) + A + B;
    };
    
    auto t = amt::timer{};
    for(auto const& el : x){
        auto sz = static_cast<std::size_t>(el);
        auto const M = Mset(sz);
        auto const N = Nset(sz);
        // double const ops = static_cast<double>(M) * static_cast<double>(N) * (2. * static_cast<double>(N) + 1);
        ub::matrix<ValueType,LayoutType> A(M,N);
        ub::matrix<ValueType,LayoutType> B(N,M);
        ub::matrix<ValueType,LayoutType> res(M,M);

        double st = amt::benchmark<MaxIter>(bench_fn, res, A, B);
        
        amt::no_opt(res);
        metric_data.update(st * 1e-6);
    }
    t.stop();
    std::cerr<<fn_name<<" has completed! ( "<<t<<" )"<<std::endl;
}

template<typename ValueType, typename LayoutType, std::size_t MaxIter = 100ul>
void tensor_el_wise_op(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );

    constexpr std::string_view fn_name = "Tensor Expr Template";
    
    auto& metric_data = m[fn_name];
    constexpr auto bench_fn = [](auto& res, auto const& A, auto const& B){
        auto temp = A * B + A + B;
        for(auto i = 0ul; i < res.size(); ++i) res[i] = temp(i);
    };
    
    auto t = amt::timer{};
    for(auto const& el : x){
        auto sz = static_cast<std::size_t>(el);
        auto const M = Mset(sz);
        auto const N = Nset(sz);
        // double const ops = static_cast<double>(M) * static_cast<double>(N) * (2. * static_cast<double>(N) + 1);
        tensor_t<ValueType,LayoutType> A(shape_t{M,N}, 1.);
        tensor_t<ValueType,LayoutType> B(shape_t{N,M}, 1.);
        tensor_t<ValueType,LayoutType> res(shape_t{M,M});

        double st = amt::benchmark<MaxIter>(bench_fn, res, A, B);
        
        amt::no_opt(res);
        metric_data.update(st * 1e-6);
    }
    t.stop();
    std::cerr<<fn_name<<" has completed! ( "<<t<<" )"<<std::endl;
}

template<typename ValueType, typename LayoutType, std::size_t MaxIter = 100ul>
void openmp_el_wise_op(std::vector<double> const& x, amt::metric<ValueType>& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    constexpr std::string_view fn_name = "With no Expr Template";
    
    auto& metric_data = m[fn_name];

    constexpr auto bench_fn = [](auto& res, auto const& A, auto const& B){
        // ( (A * B) + A ) + B;
        // amt::mtm(res, A, B, std::nullopt)();
        // amt::el_op::mul(res, A, B, std::nullopt);
        // amt::el_op::add(res, res, A, std::nullopt);
        // amt::el_op::add(res, res, B, std::nullopt);
        amt::el_op::apply_ops(res, A, B, std::nullopt, [] (auto&& a, auto&& b) {
            return a * b + a + b;
        });
    };

    auto t = amt::timer{};
    for(auto const& el : x){
        auto sz = static_cast<std::size_t>(el);
        auto const M = Mset(sz);
        auto const N = Nset(sz);
        // double const ops = static_cast<double>(M) * static_cast<double>(N) * (2. * static_cast<double>(N) + 1);
        tensor_t<ValueType,LayoutType> A(shape_t{M,N}, 1.);
        tensor_t<ValueType,LayoutType> B(shape_t{N,M}, 1.);
        tensor_t<ValueType,LayoutType> res(shape_t{M,M});
        double st = amt::benchmark<MaxIter>(bench_fn, res, A, B);
        amt::no_opt(res);
        metric_data.update(st * 1e-6);
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

    [[maybe_unused]]constexpr std::size_t max_iter = 4ul;
    [[maybe_unused]]constexpr double max_value = 1 * 1024;
    constexpr double sz = 32;
    amt::range(x, sz, max_value, sz, std::plus<>{});
    // [[maybe_unused]]constexpr double max_value = 1<<14;
    // amt::range(x, 2., max_value, 2., std::multiplies<>{});

    auto m = amt::metric<value_type>(x.size());

    ublas_el_wise_op<value_type,layout_t,max_iter>(x,m);
    tensor_el_wise_op<value_type,layout_t,max_iter>(x,m);
    openmp_el_wise_op<value_type,layout_t,max_iter>(x,m);
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
        constexpr std::string_view plot_xlable = "Size [n = m], " SIZE_SUFFIX " iterating";
        m.plot(x, "Performance of Boost.uBLAS.Tensor for the Expression Templates [iter=4]", plot_xlable, "Milliseconds");
        m.plot_speedup_semilogy<true>(x,"Semilogy for the Expression Templates [iter=4]", plot_xlable, "Milliseconds");
    #endif
    return 0;

}
