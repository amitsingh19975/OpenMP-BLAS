#if !defined(AMT_BENCHMARK_OPENBLAS_HPP)
#define AMT_BENCHMARK_OPENBLAS_HPP

#include <dlfcn.h>
#include <string>
#include <memory>
#include <functional>
#include <boost/dll/shared_library.hpp>
#include <boost/core/demangle.hpp>

using blasint = int;

namespace amt{

    namespace blas{
        enum ORDER     {RowMajor=101, ColMajor=102};
        enum TRANSPOSE {NoTrans=111, Trans=112, ConjTrans=113, ConjNoTrans=114};
        enum UPLO      {Upper=121, Lower=122};
        enum DIAG      {NonUnit=131, Unit=132};
        enum SIDE      {Left=141, Right=142};
        typedef ORDER LAYOUT;

        namespace detail{
            // vector times vector inner product
            std::function<float(blasint,const float*,blasint,const float*,blasint)> sdot = nullptr;
            std::function<double(blasint,const double*,blasint,const double*,blasint)> ddot = nullptr;
            
            // vector times vector outer product
            std::function<void(const ORDER order, const blasint M, const blasint N, const float  alpha, const float *X, const blasint incX, const float *Y, const blasint incY, float *A, const blasint lda)> sger = nullptr;
            std::function<void(const ORDER order, const blasint M, const blasint N, const double  alpha, const double *X, const blasint incX, const double *Y, const blasint incY, double *A, const blasint lda)> dger = nullptr;

            // matrix times vector
            std::function<void (const ORDER order,  const TRANSPOSE trans,  const blasint m, const blasint n,
            const float alpha, const float  *a, const blasint lda,  const float  *x, const blasint incx,  const float beta,  float  *y, const blasint incy)> sgemv = nullptr;
            std::function<void (const ORDER order,  const TRANSPOSE trans,  const blasint m, const blasint n,
            const double alpha, const double  *a, const blasint lda,  const double  *x, const blasint incx,  const double beta,  double  *y, const blasint incy)> dgemv = nullptr;

            // matrix times matrix
            std::function<void (const ORDER Order, const TRANSPOSE TransA, const TRANSPOSE TransB, const blasint M, const blasint N, const blasint K,
            const float alpha, const float *A, const blasint lda, const float *B, const blasint ldb, const float beta, float *C, const blasint ldc)> sgemm = nullptr;
            std::function<void (const ORDER Order, const TRANSPOSE TransA, const TRANSPOSE TransB, const blasint M, const blasint N, const blasint K,
            const double alpha, const double *A, const blasint lda, const double *B, const blasint ldb, const double beta, double *C, const blasint ldc)> dgemm = nullptr;
        }

        template<typename ValueType>
        ValueType dot_prod(blasint n,const ValueType* x, blasint incx, const ValueType* y, blasint incy){
            static_assert(std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType must be of type float or double");
            
            if constexpr( std::is_same_v<ValueType,float> ){
                return detail::sdot(n, x, incx, y, incy);
            }else{
                return detail::ddot(n, x, incx, y, incy);
            }
        }

        template<typename ValueType>
        void outer_prod(
            const ORDER order, const blasint M, 
            const blasint N, const ValueType  alpha, 
            const ValueType *X, const blasint incX, 
            const ValueType *Y, const blasint incY, 
            ValueType *A, const blasint lda
        ){
            static_assert(std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType must be of type float or double");
            
            if constexpr( std::is_same_v<ValueType,float> ){
                return detail::sger(order, M, N, alpha, X, incX, Y, incY, A, lda);
            }else{
                return detail::dger(order, M, N, alpha, X, incX, Y, incY, A, lda);
            }
        }

        template<typename ValueType>
        void mtv(const ORDER order,  const TRANSPOSE trans,  const blasint m, const blasint n,
            const ValueType alpha, const ValueType  *a, const blasint lda,  
            const ValueType  *x, const blasint incx,  const ValueType beta,  ValueType  *y, const blasint incy
        ){
            static_assert(std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType must be of type float or double");
            
            if constexpr( std::is_same_v<ValueType,float> ){
                return detail::sgemv(order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
            }else{
                return detail::dgemv(order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
            }
        }

        template<typename ValueType>
        void mtm(const ORDER Order, const TRANSPOSE TransA, const TRANSPOSE TransB, const blasint M, const blasint N, const blasint K,
            const ValueType alpha, const ValueType *A, const blasint lda, const ValueType *B, const blasint ldb, const ValueType beta, ValueType *C, const blasint ldc
        ){
            static_assert(std::is_same_v<ValueType,float> || std::is_same_v<ValueType,double>, "ValueType must be of type float or double");
            
            if constexpr( std::is_same_v<ValueType,float> ){
                return detail::sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
            }else{
                return detail::dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
            }
        }

    }

    namespace detail{
        void destroy_handle(void* handle){
            if(handle == nullptr) return;
            if(dlclose(handle) != 0){
                throw std::runtime_error(dlerror());
            }
        }
    }

    class OpenBlasFnLoader{
        using base_type = boost::dll::shared_library;
        OpenBlasFnLoader()
            : m_handle( m_path.data() )
        {
        }

        template<typename R, typename... Args>
        void get_fn(std::function<R(Args...)>& fn, std::string_view fn_name) const{
            if(m_handle.has(fn_name.data())){
                fn = m_handle.get<R(Args...)>(fn_name.data());
            }else
                throw std::runtime_error("amt::OpenBlasFnLoader::get(Fn&, std::string_view) : function does not exist");
        }

    public:
        OpenBlasFnLoader(OpenBlasFnLoader const& other) = delete;
        OpenBlasFnLoader(OpenBlasFnLoader&& other) = default;
        OpenBlasFnLoader& operator=(OpenBlasFnLoader const& other) = delete;
        OpenBlasFnLoader& operator=(OpenBlasFnLoader&& other) = default;


        static void init(){
            auto mod = OpenBlasFnLoader();
            mod.get_fn(blas::detail::sdot, "cblas_sdot");
            mod.get_fn(blas::detail::ddot, "cblas_ddot");
            mod.get_fn(blas::detail::sger, "cblas_sger");
            mod.get_fn(blas::detail::dger, "cblas_dger");
            mod.get_fn(blas::detail::sgemv, "cblas_sgemv");
            mod.get_fn(blas::detail::dgemv, "cblas_dgemv");
            mod.get_fn(blas::detail::sgemm, "cblas_sgemm");
            mod.get_fn(blas::detail::dgemm, "cblas_dgemm");
        }

        ~OpenBlasFnLoader(){
            m_handle.unload();
        }

    private:
        std::string_view m_path{OpenBLAS_LIB};
        base_type m_handle;
    };

} // namespace amt


#endif // AMT_BENCHMARK_OPENBLAS_HPP
