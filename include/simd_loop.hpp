#if !defined(AMT_BENCHMARK_SIMD_LOOP_HPP)
#define AMT_BENCHMARK_SIMD_LOOP_HPP

#include <macros.hpp>

namespace amt::impl{

    enum class SIMD_PROD_TYPE{
        NONE,
        INNER,
        OUTER,
        MTV
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
            (N < 1024ul) && N && !(N & (N - 1)),
            "N should be a power of 2 and less than 1024, but not 0"
        );

        template<typename ValueType, typename SizeType>
        AMT_ALWAYS_INLINE void operator()(ValueType* c, ValueType const* a, ValueType const* b, SizeType const n, SizeType const w) const noexcept{
            #define AMT_UNROLL_STATEMENT(OFFSET,C,A,B,W,I) C[I] += A[I + W * OFFSET] * b[OFFSET];
            auto const cst = b[0];
            #pragma omp simd
            for(auto i = 0ul; i < n; ++i){
                c[i] += a[i] * cst;
                if constexpr(N == 2){
                    AMT_UNROLL_LOOP1_IMPL(1,c,a,b,w,i);
                }else if constexpr(N == 4){
                    AMT_UNROLL_LOOP3_IMPL(1,c,a,b,w,i);
                }else if constexpr(N == 8){
                    AMT_UNROLL_LOOP7_IMPL(1,c,a,b,w,i);
                }else if constexpr(N == 16){
                    AMT_UNROLL_LOOP15_IMPL(1,c,a,b,w,i);
                }else if constexpr(N == 32){
                    AMT_UNROLL_LOOP31_IMPL(1,c,a,b,w,i);
                }else if constexpr(N == 64){
                    AMT_UNROLL_LOOP63_IMPL(1,c,a,b,w,i);
                }else if constexpr(N == 128){
                    AMT_UNROLL_LOOP127_IMPL(1,c,a,b,w,i);
                }else if constexpr(N == 256){
                    AMT_UNROLL_LOOP255_IMPL(1,c,a,b,w,i);
                }else if constexpr(N == 512){
                    AMT_UNROLL_LOOP511_IMPL(1,c,a,b,w,i);
                }
            }
            
            #undef AMT_UNROLL_STATEMENT
        }

    };

    template<std::size_t N, std::size_t K>
    struct simd_loop<SIMD_PROD_TYPE::MTV,N,K>{
        constexpr static auto type = SIMD_PROD_TYPE::MTV;
        constexpr static std::size_t size = N;
        template<typename ValueType, typename SizeType>
        AMT_ALWAYS_INLINE void operator()(ValueType* c, ValueType const* a, ValueType const* b, SizeType const n, SizeType const w) const noexcept{
            auto loop = simd_loop<type,K>{};
            constexpr auto M = size / K;
            for(auto i = 0ul; i < M; ++i){
                loop(c,a + i * w * K,b + i * K,n,w);
            }
        }

    };
    
} // namespace amt::impl


#undef AMT_UNROLL_LOOP1
#undef AMT_UNROLL_LOOP2
#undef AMT_UNROLL_LOOP4
#undef AMT_UNROLL_LOOP8
#undef AMT_UNROLL_LOOP16
#undef AMT_UNROLL_LOOP32
#undef AMT_UNROLL_LOOP64
#undef AMT_UNROLL_LOOP128
#undef AMT_UNROLL_LOOP256
#undef AMT_UNROLL_LOOP512
#undef AMT_UNROLL_LOOP1_IMPL
#undef AMT_UNROLL_LOOP2_IMPL
#undef AMT_UNROLL_LOOP3_IMPL
#undef AMT_UNROLL_LOOP4_IMPL
#undef AMT_UNROLL_LOOP7_IMPL
#undef AMT_UNROLL_LOOP8_IMPL
#undef AMT_UNROLL_LOOP15_IMPL
#undef AMT_UNROLL_LOOP16_IMPL
#undef AMT_UNROLL_LOOP31_IMPL
#undef AMT_UNROLL_LOOP32_IMPL
#undef AMT_UNROLL_LOOP63_IMPL
#undef AMT_UNROLL_LOOP64_IMPL
#undef AMT_UNROLL_LOOP127_IMPL
#undef AMT_UNROLL_LOOP128_IMPL
#undef AMT_UNROLL_LOOP255_IMPL
#undef AMT_UNROLL_LOOP256_IMPL
#undef AMT_UNROLL_LOOP511_IMPL
#undef AMT_UNROLL_LOOP512_IMPL

#endif // AMT_BENCHMARK_SIMD_LOOP_HPP
