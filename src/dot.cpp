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
// #include <cblas.h>
#include <mkl_cblas.h>
#include <blis.h>
#include <Eigen/Dense>
#include <complex>
#include <boost/mp11/list.hpp>
#include <boost/mp11/algorithm.hpp>
#include <openblas.hpp>
#include <range.hpp>

constexpr static std::size_t max_iter = 100ul;
constexpr static float EPSILON = std::numeric_limits<float>::epsilon() * 10.f;
namespace plt = matplot;
namespace ub = boost::numeric::ublas;

template<typename T, std::enable_if_t< std::is_floating_point_v<T>, void >* = nullptr >
constexpr bool float_compare(T a, T b, T const epsilon) noexcept{
    return fabs(a - b) <= ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon );
}

template<typename ValueType>
void compare_mat_helper(std::size_t sz){
    ub::dynamic_tensor<ValueType> v1(ub::extents<>{1ul, sz});
    std::iota(v1.begin(), v1.end(), 1.);
    auto v2 = v1;
    auto rres = ValueType{}; 
    amt::dot_prod(rres, v1, v2, std::nullopt);
    auto lres = amt::blas::dot_prod(static_cast<blasint>(sz),v1.data(), 1u, v2.data(), 1u);

    if(!float_compare(lres,rres, static_cast<ValueType>(EPSILON))){
        std::cerr<<"Incorrect Result: Tensor( " << rres << " ), BLAS( " << lres << " ), N: "<<sz<<'\n';
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

    ub::dynamic_tensor<ValueType, L> v1(ub::extents<>{1ul, sz});
    ub::dynamic_tensor<ValueType, other_layout> v2(ub::extents<>{1ul, sz});
    std::iota(v1.begin(), v1.end(), 1.);
    std::iota(v2.begin(), v2.end(), 1.);
    auto w1 = static_cast<blasint>(v1.strides()[0] * v1.strides()[1]);
    auto w2 = static_cast<blasint>(v2.strides()[0] * v2.strides()[1]);

    auto rres = ValueType{}; 
    amt::dot_prod(rres, v1, v2, std::nullopt);
    auto lres = amt::blas::dot_prod(static_cast<blasint>(sz),v1.data(), w1, v2.data(), w2);
    
    if(!float_compare(lres,rres, static_cast<ValueType>(EPSILON))){
        std::cerr<<"Incorrect Result: Tensor( " << rres << " ), BLAS( " << lres << " ), N: "<<sz<<'\n';
        exit(1);
    }
}

template<typename ValueType>
void compare_mat(std::vector<double> const& x){
    for(auto const& el: x) compare_mat_helper<ValueType>(static_cast<std::size_t>(el));
    std::cerr << "TEST PASSED!" << std::endl;
}

template<typename ValueType>
void compare_diff_mat(std::vector<double> const& x){
    for(auto const& el: x) compare_diff_mat_helper<ValueType, ub::layout::first_order>(static_cast<std::size_t>(el));
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
                double st{};
        auto k = max_iter;
        while(k--){
            amt::timer t{};
            {   
                ret += ub::inner_prod(v1,v2);
            }
            st += t();
        }
        st /= static_cast<double>(max_iter);
        metric_data.update((ops / st) * 10e-9);
    }
    return static_cast<int>(ret);
}

template<typename ValueType>
int mkl_same_layout(std::vector<double> const& x, amt::metric& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    std::string fn_name = std::string(__func__) 
        + (std::is_same_v<ValueType,float> ? "_float" : "_double");
    
    auto& metric_data = m[fn_name];
    ValueType ret{};

    for(auto const& el : x){
        double const ops = 2. * el;
        auto sz = static_cast<std::size_t>(el);
        ub::dynamic_tensor<ValueType> v1(ub::extents<>{1ul, sz},3.), v2(ub::extents<>{1ul, sz}, 3.);
        double st{};
        auto k = max_iter;
        while(k--){
            amt::timer t{};
            {   
                if constexpr(std::is_same_v<ValueType,float>){
                    ret = cblas_sdot(static_cast<MKL_INT>(sz),v1.data(), 1u, v2.data(), 1u);
                }else if constexpr(std::is_same_v<ValueType,double>){
                    ret = cblas_ddot(static_cast<MKL_INT>(sz),v1.data(), 1u, v2.data(), 1u);
                }
            }
            st += t();
        }
        st /= static_cast<double>(max_iter);
        metric_data.update((ops / st) * 10e-9);
    }
    return static_cast<int>(ret);
}

#ifdef AMT_BENCHMARK_OPENBLAS_HPP
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
        double st{};
        auto k = max_iter;
        while(k--){
            amt::timer t{};
            {   
                ret = amt::blas::dot_prod(static_cast<blasint>(sz),v1.data(), 1, v2.data(), 1);
            }
            st += t();
        }
        st /= static_cast<double>(max_iter);
        metric_data.update((ops / st) * 10e-9);
    }
    return static_cast<int>(ret);
}
#endif

template<typename ValueType>
int ref_same_layout(std::vector<double> const& x, amt::metric& m){
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    ValueType ret{};

    for(auto const& el : x){
        double const ops = 2. * el;
        auto sz = static_cast<std::size_t>(el);
        ub::dynamic_tensor<ValueType> v1(ub::extents<>{1ul, sz},3.), v2(ub::extents<>{1ul, sz}, 3.);
        double st{};
        auto k = max_iter;
        while(k--){
            amt::timer t{};
            {   
                amt::dot_prod_ref(ret, v1, v2, t);
            }
            st += t();
        }
        st /= static_cast<double>(max_iter);
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
        double st{};
        auto k = max_iter;
        while(k--){
            amt::timer t{};
            {   
                amt::dot_prod(ret, v1, v2, std::nullopt, t);
            }
            st += t();
        }
        st /= static_cast<double>(max_iter);
        metric_data.update((ops / st) * 10e-9);
    }
    return static_cast<int>(ret);
}

template<std::size_t Start, std::size_t End, typename ValueType>
int static_tensor_same_layout(amt::metric& m){
    using namespace boost::mp11;
    static_assert( std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType not supported" );
    
    std::string fn_name = std::string(__func__) 
        + (std::is_same_v<ValueType,float> ? "_float" : "_double");
    
    auto& metric_data = m[fn_name];
    ValueType ret{};

    using number_list = mp_iota_c<End>;
    using range = mp_drop_c<number_list, std::max(1ul, Start) >;

    mp_for_each<range>([&](auto I){
        constexpr auto sz = decltype(I)::value;
        double const ops = 2. * sz;
        using extents_type = ub::static_extents<1ul, sz>;
        using tensor_type = ub::static_tensor<ValueType,extents_type>;
        tensor_type v1(3.), v2( 3.);
        double st{};
        auto k = max_iter;
        while(k--){
            amt::timer t{};
            {   
                amt::dot_prod(ret, v1, v2, t);
            }
            st += t();
        }
        st /= static_cast<double>(max_iter);
        metric_data.update((ops / st) * 10e-9);
    });
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
        double st{};
        auto k = max_iter;
        while(k--){
            amt::timer t{};
            {   
                if constexpr(std::is_same_v<ValueType,float>){
                    bli_sdotv(BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, static_cast<dim_t>(sz), v1.data(), 1, v2.data(), 1, &ret);
                }else{
                    bli_ddotv(BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, static_cast<dim_t>(sz), v1.data(), 1, v2.data(), 1, &ret);
                }
            }
            st += t();
        }
        st /= static_cast<double>(max_iter);
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
        double st{};
        auto k = max_iter;
        while(k--){
            amt::timer t{};
            {   
                ret += v1.dot(v2);
            }
            st += t();
        }
        st /= static_cast<double>(max_iter);
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
        double st{};
        auto k = max_iter;
        while(k--){
            amt::timer t{};
            {   
                amt::dot_prod_ref(ret, v1, v2, t);
                ret += v1.dot(v2);
            }
            st += t();
        }
        st /= static_cast<double>(max_iter);
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
        double st{};
        auto k = max_iter;
        while(k--){
            amt::timer t{};
            {   
                amt::dot_prod(ret, v1, v2, std::nullopt, t);
            }
            st += t();
        }
        st /= static_cast<double>(max_iter);
        metric_data.update((ops / st) * 10e-9);
    }
    return static_cast<int>(ret);
}

#ifdef AMT_BENCHMARK_OPENBLAS_HPP
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
        double st{};
        auto k = max_iter;
        while(k--){
            amt::timer t{};
            {   
                ret = amt::blas::dot_prod(static_cast<blasint>(sz),v1.data(), w1, v2.data(), w2);
                ret += v1.dot(v2);
            }
            st += t();
        }
        st /= static_cast<double>(max_iter);
        metric_data.update((ops / st) * 10e-9);
    }
    return static_cast<int>(ret);
}
#endif

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
        double st{};
        auto k = max_iter;
        while(k--){
            amt::timer t{};
            {   
                if constexpr(std::is_same_v<ValueType,float>){
                    bli_sdotv(BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, static_cast<dim_t>(sz), v1.data(), w1, v2.data(), w2, &ret);
                }else if constexpr(std::is_same_v<ValueType,double>){
                    bli_ddotv(BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, static_cast<dim_t>(sz), v1.data(), w1, v2.data(), w2, &ret);
                }
            }
            st += t();
        }
        st /= static_cast<double>(max_iter);
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
        double st{};
        auto k = max_iter;
        while(k--){
            amt::timer t{};
            {   
                ret += v1.dot(v2);
            }
            st += t();
        }
        st /= static_cast<double>(max_iter);
        metric_data.update((ops / st) * 10e-9);
    }
    return static_cast<int>(ret);
}

// #define ENABLE_TEST
// #define DIFFERENT_LAYOUT
// #define DISABLE_PLOT
// #define SPEEDUP_PLOT

int main(){
    using value_type = float;
    // using value_type = double;
    amt::OpenBlasFnLoader::init();
    std::vector<double> x;
    [[maybe_unused]]constexpr double max_value = (1u<<20);
    amt::range(x, 2., max_value, 1024., std::plus<>{});
    
#ifndef ENABLE_TEST

    int res = 0;
    auto m = amt::metric(x.size());
    // exit(0);
    #ifndef DIFFERENT_LAYOUT
        // res += ref_same_layout<value_type>(x,m);
        // res += ublas_dot_same_layout<value_type>(x,m);
        // res += blas_same_layout<value_type>(x,m);
        // // res += static_tensor_same_layout<2ul, max_size, value_type>(m);
        // res += blis_same_layout<value_type>(x,m);
        // res += eigen_same_layout<value_type>(x,m);
        res += tensor_same_layout<value_type>(x,m);
        res += mkl_same_layout<value_type>(x,m);
    #else
        res += ref_dot_diff_layout<value_type,ub::layout::first_order>(x,m);
        res += blas_dot_diff_layout<value_type,ub::layout::first_order>(x,m);
        res += tensor_dot_diff_layout<value_type,ub::layout::first_order>(x,m);
        res += blis_dot_diff_layout<value_type,ub::layout::first_order>(x,m);
        res += eigen_dot_diff_layout<value_type,ub::layout::first_order>(x,m);

    #endif

    std::cout<<m<<'\n';
    #ifndef DISABLE_PLOT
        #ifndef SPEEDUP_PLOT
            m.plot(x);
        #else
            m.plot_speedup(x);
        #endif
    #endif
    
    return res;
#else
    
    #ifndef DIFFERENT_LAYOUT
        compare_mat<value_type>(x);
    #else
        compare_diff_mat<value_type>(x);
    #endif
    return 0;
#endif
}
