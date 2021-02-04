#define BOOST_UBLAS_USE_SIMD
#include <iostream>
#include <numeric>
#include <algorithm>
#include <vector>
#include <cassert>
#include <timer.hpp>
#include <metric.hpp>
#include <dot.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <cblas.h>
#include <blis.h>
#include <Eigen/Dense>
#include <complex>


namespace plt = matplot;
namespace ub = boost::numeric::ublas;

template<typename ValueType>
void compare_mat_helper(std::size_t sz){
    ub::dynamic_tensor<ValueType> v1(ub::extents<>{1ul, sz},3.), v2(ub::extents<>{1ul, sz}, 3.);
    auto lres = ValueType{};
    auto rres = ValueType{}; 
    amt::dot_prod(rres, v1, v2);
    if constexpr(std::is_same_v<ValueType,float>){
        lres = cblas_sdot(static_cast<blasint>(sz),v1.data(), 1u, v2.data(), 1u);
    }else if constexpr(std::is_same_v<ValueType,double>){
        lres = cblas_ddot(static_cast<blasint>(sz),v1.data(), 1u, v2.data(), 1u);
    }
    if(lres != rres){
        std::cerr<<"Incorrect Result: Tensor( " << rres << " ), BLAS( " << lres << " )\n";
        exit(1);
    }
}

template<typename ValueType, typename L>
void compare_diff_mat_helper(std::size_t sz){
    
    using other_layout = std::conditional_t<
        std::is_same_v<L,ub::layout::first_order>,
        ub::layout::last_order,
        ub::layout::first_order
    >;

    ub::dynamic_tensor<ValueType, L> v1(ub::extents<>{1ul, sz},3.);
    ub::dynamic_tensor<ValueType, other_layout> v2(ub::extents<>{1ul, sz}, 3.);
    auto lres = ValueType{};
    auto rres = ValueType{}; 
    amt::dot_prod(rres, v1, v2);
    
    auto w1 = static_cast<blasint>(v1.strides()[0] * v1.strides()[1]);
    auto w2 = static_cast<blasint>(v2.strides()[0] * v2.strides()[1]);

    if constexpr(std::is_same_v<ValueType,float>){
        lres = cblas_sdot(static_cast<blasint>(sz),v1.data(), w1, v2.data(), w2);
    }else if constexpr(std::is_same_v<ValueType,double>){
        lres = cblas_ddot(static_cast<blasint>(sz),v1.data(), w1, v2.data(), w2);
    }
    if(lres != rres){
        std::cerr<<"Incorrect Result: Tensor( " << rres << " ), BLAS( " << lres << " )\n";
        exit(1);
    }
}

template<typename ValueType>
void compare_mat(std::size_t start, std::size_t end, std::size_t inc = 1u){
    for(; start < end; start += inc) compare_mat_helper<ValueType>(start);
    std::cerr << "TEST PASSED!" << std::endl;
}

template<typename ValueType>
void compare_diff_mat(std::size_t start, std::size_t end, std::size_t inc = 1u){
    for(; start < end; start += inc) compare_diff_mat_helper<ValueType, ub::layout::first_order>(start);
    std::cerr << "TEST PASSED!" << std::endl;
}

template<typename ValueType>
int ublas_dot_same_layout(std::vector<double> const& x, amt::metric& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );

    std::string fn_name = std::string(__func__) 
        + (std::is_same_v<ValueType,float> ? "_float" : "_double");
    
    auto& metric_data = m[fn_name];
    ValueType ret{};

    for(auto const& el : x){
        double const ops = 2. * el;
        auto sz = static_cast<std::size_t>(el);
        ub::vector<ValueType> v1(sz,3.), v2(sz, 3.);
        amt::timer t{};
        {
            ret += ub::inner_prod(v1,v2);
        }
        auto st = t.stop();

        metric_data.update((ops / st) * 10e-9);
    }
    return static_cast<int>(ret);
}

template<typename ValueType>
int blas_same_layout(std::vector<double> const& x, amt::metric& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    std::string fn_name = std::string(__func__) 
        + (std::is_same_v<ValueType,float> ? "_float" : "_double");
    
    auto& metric_data = m[fn_name];
    ValueType ret{};

    for(auto const& el : x){
        double const ops = 2. * el;
        auto sz = static_cast<std::size_t>(el);
        ub::dynamic_tensor<ValueType> v1(ub::extents<>{1ul, sz},3.), v2(ub::extents<>{1ul, sz}, 3.);
        amt::timer t{};
        {   
            if constexpr(std::is_same_v<ValueType,float>){
                ret = cblas_sdot(static_cast<blasint>(sz),v1.data(), 1u, v2.data(), 1u);
            }else if constexpr(std::is_same_v<ValueType,double>){
                ret = cblas_ddot(static_cast<blasint>(sz),v1.data(), 1u, v2.data(), 1u);
            }
        }

        auto st = t.stop();
        metric_data.update((ops / st) * 10e-9);
    }
    return static_cast<int>(ret);
}

template<typename ValueType>
int ref_same_layout(std::vector<double> const& x, amt::metric& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    ValueType ret{};

    for(auto const& el : x){
        double const ops = 2. * el;
        auto sz = static_cast<std::size_t>(el);
        ub::dynamic_tensor<ValueType> v1(ub::extents<>{1ul, sz},3.), v2(ub::extents<>{1ul, sz}, 3.);
        amt::timer t{};
        {   
            amt::dot_prod(ret, v1, v2);
        }
        auto st = t.stop();
        m.update_ref((ops / st) * 10e-9);
    }
    return static_cast<int>(ret);
}

template<typename ValueType>
int tensor_same_layout(std::vector<double> const& x, amt::metric& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    std::string fn_name = std::string(__func__) 
        + (std::is_same_v<ValueType,float> ? "_float" : "_double");
    
    auto& metric_data = m[fn_name];
    ValueType ret{};

    for(auto const& el : x){
        double const ops = 2. * el;
        auto sz = static_cast<std::size_t>(el);
        ub::dynamic_tensor<ValueType> v1(ub::extents<>{1ul, sz},3.), v2(ub::extents<>{1ul, sz}, 3.);
        amt::timer t{};
        {   
            amt::dot_prod(ret, v1, v2);
        }
        auto st = t.stop();
        metric_data.update((ops / st) * 10e-9);
    }
    return static_cast<int>(ret);
}

template<typename ValueType>
int blis_same_layout(std::vector<double> const& x, amt::metric& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    std::string fn_name = std::string(__func__) 
        + (std::is_same_v<ValueType,float> ? "_float" : "_double");
    
    auto& metric_data = m[fn_name];
    ValueType ret{};

    for(auto const& el : x){
        double const ops = 2. * el;
        auto sz = static_cast<std::size_t>(el);
        ub::dynamic_tensor<ValueType> v1(ub::extents<>{1ul, sz},3.), v2(ub::extents<>{1ul, sz}, 3.);
        amt::timer t{};
        {   
            if constexpr(std::is_same_v<ValueType,float>){
                bli_sdotv(BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, static_cast<dim_t>(sz), v1.data(), 1, v2.data(), 1, &ret);
            }else{
                bli_ddotv(BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, static_cast<dim_t>(sz), v1.data(), 1, v2.data(), 1, &ret);
            }
        }
        auto st = t.stop();
        metric_data.update((ops / st) * 10e-9);
    }
    return static_cast<int>(ret);
}

template<typename ValueType>
int eigen_same_layout(std::vector<double> const& x, amt::metric& m){
    using namespace Eigen;
    using vector_type = Matrix<ValueType,-1,1,ColMajor>;
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    std::string fn_name = std::string(__func__) 
        + (std::is_same_v<ValueType,float> ? "_float" : "_double");
    
    auto& metric_data = m[fn_name];
    ValueType ret{};

    for(auto const& el : x){
        double const ops = 2. * el;
        auto sz = static_cast<std::size_t>(el);
        vector_type v1(sz), v2(sz);
        v1.fill(3.);
        v2.fill(3.);
        amt::timer t{};
        {   
            ret += v1.dot(v2);
        }
        auto st = t.stop();
        metric_data.update((ops / st) * 10e-9);
    }
    return static_cast<int>(ret);
}

template<typename ValueType, typename L>
int ref_dot_diff_layout(std::vector<double> const& x, amt::metric& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    using other_layout = std::conditional_t<
        std::is_same_v<L,ub::layout::first_order>,
        ub::layout::last_order,
        ub::layout::first_order
    >;

    ValueType ret{};

    for(auto const& el : x){
        double const ops = 2. * el;
        auto sz = static_cast<std::size_t>(el);
        ub::dynamic_tensor<ValueType, L> v1(ub::extents<>{1ul, sz},3.);
        ub::dynamic_tensor<ValueType, other_layout> v2(ub::extents<>{1ul, sz}, 3.);
        amt::timer t{};
        {   
            amt::dot_prod_ref(ret, v1, v2);
        }
        auto st = t.stop();
        m.update_ref((ops / st) * 10e-9);
    }
    return static_cast<int>(ret);
}

template<typename ValueType, typename L>
int tensor_dot_diff_layout(std::vector<double> const& x, amt::metric& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    using other_layout = std::conditional_t<
        std::is_same_v<L,ub::layout::first_order>,
        ub::layout::last_order,
        ub::layout::first_order
    >;

    std::string fn_name = std::string(__func__) 
        + (std::is_same_v<L,ub::layout::first_order> ? "_first_last" : "_last_first")
        + (std::is_same_v<ValueType,float> ? "_float" : "_double");
    
    auto& metric_data = m[fn_name];
    ValueType ret{};

    for(auto const& el : x){
        double const ops = 2. * el;
        auto sz = static_cast<std::size_t>(el);
        ub::dynamic_tensor<ValueType, L> v1(ub::extents<>{1ul, sz},3.);
        ub::dynamic_tensor<ValueType, other_layout> v2(ub::extents<>{1ul, sz}, 3.);
        amt::timer t{};
        {   
            amt::dot_prod(ret, v1, v2);
        }
        auto st = t.stop();
        metric_data.update((ops / st) * 10e-9);
    }
    return static_cast<int>(ret);
}

template<typename ValueType, typename L>
int blas_dot_diff_layout(std::vector<double> const& x, amt::metric& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    using other_layout = std::conditional_t<
        std::is_same_v<L,ub::layout::first_order>,
        ub::layout::last_order,
        ub::layout::first_order
    >;

    std::string fn_name = std::string(__func__) 
        + (std::is_same_v<L,ub::layout::first_order> ? "_first_last" : "_last_first")
        + (std::is_same_v<ValueType,float> ? "_float" : "_double");
    
    auto& metric_data = m[fn_name];
    ValueType ret{};

    for(auto const& el : x){
        double const ops = 2. * el;
        auto sz = static_cast<std::size_t>(el);
        ub::dynamic_tensor<ValueType, L> v1(ub::extents<>{1ul, sz},3.);
        ub::dynamic_tensor<ValueType, other_layout> v2(ub::extents<>{1ul, sz}, 3.);
        auto w1 = static_cast<blasint>(v1.strides()[0] * v1.strides()[1]);
        auto w2 = static_cast<blasint>(v2.strides()[0] * v2.strides()[1]);
        amt::timer t{};
        {   
            if constexpr(std::is_same_v<ValueType,float>){
                ret = cblas_sdot(static_cast<blasint>(sz),v1.data(), w1, v2.data(), w2);
            }else if constexpr(std::is_same_v<ValueType,double>){
                ret = cblas_ddot(static_cast<blasint>(sz),v1.data(), w1, v2.data(), w2);
            }
        }

        auto st = t.stop();
        metric_data.update((ops / st) * 10e-9);
    }
    return static_cast<int>(ret);
}

template<typename ValueType, typename L>
int blis_dot_diff_layout(std::vector<double> const& x, amt::metric& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    using other_layout = std::conditional_t<
        std::is_same_v<L,ub::layout::first_order>,
        ub::layout::last_order,
        ub::layout::first_order
    >;

    std::string fn_name = std::string(__func__) 
        + (std::is_same_v<L,ub::layout::first_order> ? "_first_last" : "_last_first")
        + (std::is_same_v<ValueType,float> ? "_float" : "_double");
    
    auto& metric_data = m[fn_name];
    ValueType ret{};

    for(auto const& el : x){
        double const ops = 2. * el;
        auto sz = static_cast<std::size_t>(el);
        ub::dynamic_tensor<ValueType, L> v1(ub::extents<>{1ul, sz},3.);
        ub::dynamic_tensor<ValueType, other_layout> v2(ub::extents<>{1ul, sz}, 3.);
        auto w1 = static_cast<dim_t>(v1.strides()[0] * v1.strides()[1]);
        auto w2 = static_cast<dim_t>(v2.strides()[0] * v2.strides()[1]);
        amt::timer t{};
        {   
            if constexpr(std::is_same_v<ValueType,float>){
                bli_sdotv(BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, static_cast<dim_t>(sz), v1.data(), w1, v2.data(), w2, &ret);
            }else if constexpr(std::is_same_v<ValueType,double>){
                bli_ddotv(BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, static_cast<dim_t>(sz), v1.data(), w1, v2.data(), w2, &ret);
            }
        }

        auto st = t.stop();
        metric_data.update((ops / st) * 10e-9);
    }
    return static_cast<int>(ret);
}

template<typename ValueType, typename L>
int eigen_dot_diff_layout(std::vector<double> const& x, amt::metric& m){
    using namespace Eigen;
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    constexpr auto layout = (std::is_same_v<L,ub::layout::first_order> ? ColMajor : RowMajor);
    constexpr auto other_layout = (std::is_same_v<L,ub::layout::first_order> ? RowMajor : ColMajor);

    std::string fn_name = std::string(__func__) 
        + (std::is_same_v<L,ub::layout::first_order> ? "_first_last" : "_last_first")
        + (std::is_same_v<ValueType,float> ? "_float" : "_double");
    
    auto& metric_data = m[fn_name];
    ValueType ret{};

    for(auto const& el : x){
        double const ops = 2. * el;
        auto sz = static_cast<std::size_t>(el);
        
        Matrix<ValueType,-1,1,layout> v1(sz);
        Matrix<ValueType,1,-1,other_layout> v2(sz);
        v1.fill(3.);
        v2.fill(3.);
        amt::timer t{};
        {   
            ret += v1.dot(v2);
        }

        auto st = t.stop();
        metric_data.update((ops / st) * 10e-9);
    }
    return static_cast<int>(ret);
}

// #define TEST_ON
// #define DIFFERENT_LAYOUT
// #define SPEEDUP_PLOT

int main(){
    using value_type = float;
    constexpr auto max_size = 4096ul;
    
#ifndef TEST_ON

    int res = 0;
    std::vector<double> x(max_size);
    std::iota(x.begin(), x.end(), 2.);
    auto m = amt::metric(max_size);
    #ifndef DIFFERENT_LAYOUT
        res += ref_same_layout<value_type>(x,m);
        res += ublas_dot_same_layout<value_type>(x,m);
        res += blas_same_layout<value_type>(x,m);
        res += tensor_same_layout<value_type>(x,m);
        res += blis_same_layout<value_type>(x,m);
        res += eigen_same_layout<value_type>(x,m);
    #else
        res += ref_dot_diff_layout<value_type,ub::layout::first_order>(x,m);
        res += blas_dot_diff_layout<value_type,ub::layout::first_order>(x,m);
        res += tensor_dot_diff_layout<value_type,ub::layout::first_order>(x,m);
        res += blis_dot_diff_layout<value_type,ub::layout::first_order>(x,m);
        res += eigen_dot_diff_layout<value_type,ub::layout::first_order>(x,m);

    #endif

    std::cout<<m<<'\n';
    #ifndef SPEEDUP_PLOT
        m.plot(x);
    #else
        m.plot_speedup(x);
    #endif
    return res;
#else
    
    #ifndef DIFFERENT_LAYOUT
        compare_mat<value_type>(2,max_size);
    #else
        compare_diff_mat<value_type>(2,max_size);
    #endif
    return 0;
#endif
}
