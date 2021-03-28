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

    #define MTV_UNROLL_LOOP1(C,A,B,I) \
        C[I] += A[I] * b[0];
    
    #define MTV_UNROLL_LOOP8_IMPL(C,A,B,W,I,J) \
        C[I] += A[I + W * (0+J)] * b[0 + J]; \
        C[I] += A[I + W * (1+J)] * b[1 + J]; \
        C[I] += A[I + W * (2+J)] * b[2 + J]; \
        C[I] += A[I + W * (3+J)] * b[3 + J]; \
        C[I] += A[I + W * (4+J)] * b[4 + J]; \
        C[I] += A[I + W * (5+J)] * b[5 + J]; \
        C[I] += A[I + W * (6+J)] * b[6 + J]; \
        C[I] += A[I + W * (7+J)] * b[7 + J]; \
    
    #define MTV_UNROLL_LOOP16_IMPL(C,A,B,W,I,J) \
        MTV_UNROLL_LOOP8_IMPL(C,A,B,W,I,J) \
        MTV_UNROLL_LOOP8_IMPL(C,A,B,W,I,J + 8) \
    
    #define MTV_UNROLL_LOOP32_IMPL(C,A,B,W,I,J) \
        MTV_UNROLL_LOOP16_IMPL(C,A,B,W,I,J) \
        MTV_UNROLL_LOOP16_IMPL(C,A,B,W,I,J + 16) \
    
    #define MTV_UNROLL_LOOP64_IMPL(C,A,B,W,I,J) \
        MTV_UNROLL_LOOP32_IMPL(C,A,B,W,I,J) \
        MTV_UNROLL_LOOP32_IMPL(C,A,B,W,I,J + 32) \
    
    #define MTV_UNROLL_LOOP128_IMPL(C,A,B,W,I,J) \
        MTV_UNROLL_LOOP64_IMPL(C,A,B,W,I,J) \
        MTV_UNROLL_LOOP64_IMPL(C,A,B,W,I,J + 64) \
    
    #define MTV_UNROLL_LOOP256_IMPL(C,A,B,W,I,J) \
        MTV_UNROLL_LOOP128_IMPL(C,A,B,W,I,J) \
        MTV_UNROLL_LOOP128_IMPL(C,A,B,W,I,J + 128) \
    
    #define MTV_UNROLL_LOOP512_IMPL(C,A,B,W,I,J) \
        MTV_UNROLL_LOOP256_IMPL(C,A,B,W,I,J) \
        MTV_UNROLL_LOOP256_IMPL(C,A,B,W,I,J + 256) \

    #define MTV_UNROLL_LOOP1(C,A,B,I) C[I] += A[I] * b[0];
    
    #define MTV_UNROLL_LOOP8(C,A,B,W,I) MTV_UNROLL_LOOP8_IMPL(C,A,B,W,I,0)
    #define MTV_UNROLL_LOOP16(C,A,B,W,I) MTV_UNROLL_LOOP16_IMPL(C,A,B,W,I,0)
    #define MTV_UNROLL_LOOP32(C,A,B,W,I) MTV_UNROLL_LOOP32_IMPL(C,A,B,W,I,0)
    #define MTV_UNROLL_LOOP64(C,A,B,W,I) MTV_UNROLL_LOOP64_IMPL(C,A,B,W,I,0)
    #define MTV_UNROLL_LOOP128(C,A,B,W,I) MTV_UNROLL_LOOP128_IMPL(C,A,B,W,I,0)
    #define MTV_UNROLL_LOOP256(C,A,B,W,I) MTV_UNROLL_LOOP256_IMPL(C,A,B,W,I,0)
    #define MTV_UNROLL_LOOP512(C,A,B,W,I) MTV_UNROLL_LOOP512_IMPL(C,A,B,W,I,0)

    template<std::size_t N>
    struct simd_loop<SIMD_PROD_TYPE::MTV,N>{
        constexpr static auto type = SIMD_PROD_TYPE::MTV;
        constexpr static std::size_t size = N;
        static_assert(
            (N==1ul) || (N==8ul) || (N==16ul) || (N==32ul) ||
            (N==64ul) || (N==128ul) || (N==256ul) || (N==512ul)
        );

        template<typename ValueType, typename SizeType>
        AMT_ALWAYS_INLINE void operator()(ValueType* c, ValueType const* a, ValueType const* b, SizeType const n, SizeType const w) const noexcept{
            #pragma omp simd
            for(auto i = 0ul; i < n; ++i){
                if constexpr(N == 1){
                    MTV_UNROLL_LOOP1(c,a,b,i);
                }else if constexpr(N == 8){
                    MTV_UNROLL_LOOP8(c,a,b,w,i);
                }else if constexpr(N == 16){
                    MTV_UNROLL_LOOP16(c,a,b,w,i);
                }else if constexpr(N == 32){
                    MTV_UNROLL_LOOP32(c,a,b,w,i);
                }else if constexpr(N == 64){
                    MTV_UNROLL_LOOP64(c,a,b,w,i);
                }else if constexpr(N == 128){
                    MTV_UNROLL_LOOP128(c,a,b,w,i);
                }else if constexpr(N == 256){
                    MTV_UNROLL_LOOP256(c,a,b,w,i);
                }else{
                    MTV_UNROLL_LOOP512(c,a,b,w,i);
                }
            }
        }

    };

    
    #undef MTV_UNROLL_LOOP1
    #undef MTV_UNROLL_LOOP8
    #undef MTV_UNROLL_LOOP16
    #undef MTV_UNROLL_LOOP32
    #undef MTV_UNROLL_LOOP64
    #undef MTV_UNROLL_LOOP128
    #undef MTV_UNROLL_LOOP256
    #undef MTV_UNROLL_LOOP512
    #undef MTV_UNROLL_LOOP8_IMPL
    #undef MTV_UNROLL_LOOP16_IMPL
    #undef MTV_UNROLL_LOOP32_IMPL
    #undef MTV_UNROLL_LOOP64_IMPL
    #undef MTV_UNROLL_LOOP128_IMPL
    #undef MTV_UNROLL_LOOP256_IMPL
    #undef MTV_UNROLL_LOOP512_IMPL
    

} // namespace amt::impl


#endif // AMT_BENCHMARK_SIMD_LOOP_HPP
