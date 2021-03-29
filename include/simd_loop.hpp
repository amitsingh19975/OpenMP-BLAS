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
            (N==1ul) || (N==8ul) || (N==16ul) || (N==32ul) ||
            (N==64ul) || (N==128ul) || (N==256ul) || (N==512ul)
        );

        template<typename ValueType, typename SizeType>
        AMT_ALWAYS_INLINE void operator()(ValueType* c, ValueType const* a, ValueType const* b, SizeType const n, SizeType const w) const noexcept{
            #define AMT_UNROLL_STATEMENT(OFFSET,C,A,B,W,I) C[I] += A[I + W * OFFSET] * b[OFFSET];
            
            #pragma omp simd
            for(auto i = 0ul; i < n; ++i){
                if constexpr(N == 1){
                    AMT_UNROLL_LOOP1(c,a,b,w,i);
                }else if constexpr(N == 8){
                    AMT_UNROLL_LOOP8(c,a,b,w,i);
                }else if constexpr(N == 16){
                    AMT_UNROLL_LOOP16(c,a,b,w,i);
                }else if constexpr(N == 32){
                    AMT_UNROLL_LOOP32(c,a,b,w,i);
                }else if constexpr(N == 64){
                    AMT_UNROLL_LOOP64(c,a,b,w,i);
                }else if constexpr(N == 128){
                    AMT_UNROLL_LOOP128(c,a,b,w,i);
                }else if constexpr(N == 256){
                    AMT_UNROLL_LOOP256(c,a,b,w,i);
                }else{
                    AMT_UNROLL_LOOP512(c,a,b,w,i);
                }
            }
            
            #undef AMT_UNROLL_STATEMENT
        }

    };
    
} // namespace amt::impl


#undef AMT_UNROLL_LOOP1
#undef AMT_UNROLL_LOOP8
#undef AMT_UNROLL_LOOP16
#undef AMT_UNROLL_LOOP32
#undef AMT_UNROLL_LOOP64
#undef AMT_UNROLL_LOOP128
#undef AMT_UNROLL_LOOP256
#undef AMT_UNROLL_LOOP512
#undef AMT_UNROLL_LOOP1_IMPL
#undef AMT_UNROLL_LOOP8_IMPL
#undef AMT_UNROLL_LOOP16_IMPL
#undef AMT_UNROLL_LOOP32_IMPL
#undef AMT_UNROLL_LOOP64_IMPL
#undef AMT_UNROLL_LOOP128_IMPL
#undef AMT_UNROLL_LOOP256_IMPL
#undef AMT_UNROLL_LOOP512_IMPL

#endif // AMT_BENCHMARK_SIMD_LOOP_HPP
