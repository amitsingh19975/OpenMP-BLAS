#if !defined(AMT_BENCHMARK_SIMD_LOOP_HPP)
#define AMT_BENCHMARK_SIMD_LOOP_HPP

#include <macros.hpp>
#include <utils.hpp>

namespace amt::impl{

    enum class SIMD_PROD_TYPE{
        NONE,
        INNER,
        OUTER,
        MTV,
        MTM
    };

    template<SIMD_PROD_TYPE OPType,std::size_t... Ns>
    struct simd_loop;

    template<>
    struct simd_loop<SIMD_PROD_TYPE::INNER>{
        constexpr static auto type = SIMD_PROD_TYPE::INNER;

        template<typename ValueType, typename SizeType>
        AMT_ALWAYS_INLINE ValueType operator()(ValueType const* a, ValueType const* b, SizeType const n) const noexcept{
            auto sum = ValueType{};
            #pragma omp simd reduction(+:sum)
            for(auto i = 0ul; i < n; ++i){
                sum += (a[i] * b[i]);
            }
            return sum;
        };

    };

    template<>
    struct simd_loop<SIMD_PROD_TYPE::OUTER>{
        constexpr static auto type = SIMD_PROD_TYPE::OUTER;
        
        template<typename ValueType, typename SizeType>
        AMT_ALWAYS_INLINE void operator()(ValueType* c, ValueType const* const a, ValueType const* const b, SizeType const n) const noexcept{
            auto const cst = *b;
            #pragma omp simd
            for(auto i = 0ul; i < n; ++i){
                c[i] += (a[i] * cst);
            }
        };

    };

    template<std::size_t N>
    struct simd_loop<SIMD_PROD_TYPE::MTV,N>{
        constexpr static auto type = SIMD_PROD_TYPE::MTV;
        constexpr static std::size_t size = N;
        static_assert(
            N && !(N & (N - 1)),
            "N should be a power of 2 and greater than 0"
        );

        template<typename ValueType, typename SizeType>
        AMT_ALWAYS_INLINE void operator()(ValueType* c, ValueType const* a, ValueType const* b, SizeType const n, SizeType const w) const noexcept{
            auto const cst = b[0];
            #pragma omp simd
            for(auto i = 0ul; i < n; ++i){
                c[i] += a[i] * cst;
                if constexpr(N > 1){
                    #pragma unroll(N-1)
                    for(auto j = 1ul; j < N; ++j){
                        c[i] += a[i + w * j] * b[j];
                    }
                }
            }
        }

    };

    template<std::size_t MR,std::size_t NR>
    struct simd_loop<SIMD_PROD_TYPE::MTM,MR,NR>{
        constexpr static auto type = SIMD_PROD_TYPE::MTM;
        
        template<typename ValueType, typename SizeType>
        auto operator()(
            ValueType* c, SizeType const ldc,
            ValueType const* const a,
            ValueType const* const b, SizeType const ldb, 
            SizeType const K,
            SizeType const nr
        ) const noexcept{
            helper(c,ldc,a,b,ldb,K,nr);
        }
        

    private:
        
        template<typename ValueType, typename SizeType>
        AMT_ALWAYS_INLINE static void helper(
            ValueType* c,
            ValueType const* const a,
            ValueType const* const b, SizeType const ldb,
            SizeType const K
        ) noexcept{

            #pragma omp simd reduction(+:c[0:NR])
            for(auto k = 0ul; k < K; ++k){
                auto aval = a[k];
                auto bk = b + k;
                for(auto i = 0ul; i < NR; ++i){
                    c[i] += aval * bk[i * ldb];
                }
            }

            // copy_vec(c,ldc,acc,nr);
        }

        template<typename ValueType, typename SizeType>
        AMT_ALWAYS_INLINE static void helper(
            ValueType* c, SizeType const ldc,
            ValueType const* const a,
            ValueType const* const b, SizeType const ldb,
            SizeType const K,
            SizeType const nr
        ) noexcept{
            ValueType acc[NR] = {0};
            helper(acc,a,b,ldb,K);;
            copy_vec(c,ldc,acc,nr);
        }

        template<typename ValueType, typename SizeType>
        AMT_ALWAYS_INLINE static void copy_vec(ValueType* out, SizeType const wo, ValueType const* in, SizeType const nr) noexcept{
            #pragma omp simd
            for(auto i = 0ul; i < nr; ++i){
                out[i * wo] += in[i];
            }
        }

    };


} // namespace amt::impl

#endif // AMT_BENCHMARK_SIMD_LOOP_HPP
