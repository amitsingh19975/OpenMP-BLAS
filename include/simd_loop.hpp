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
        
        template<std::size_t I = 0ul, bool IsMR = true, std::size_t M = 0ul, typename ValueType, typename SizeType>
        constexpr void operator()(
            ValueType* c, SizeType const* wc,
            ValueType const* const a,
            ValueType const* const b,
            SizeType const K,
            SizeType const mr,
            SizeType const nr
        ) const noexcept{
            static_assert((I <= MR + 1ul) || (I <= NR + 1ul));
            if constexpr(IsMR){
                if constexpr(I <= MR){
                    switch(mr){
                        case I: this->operator()<0ul,false,I>(c,wc,a,b,K,mr,nr); return;
                        default: this->operator()<I + 1,true>(c,wc,a,b,K,mr,nr); return;
                    }
                }
            }else{
                if constexpr(I <= NR){
                    switch(nr){
                        case I: helper<M,I>(c,wc,a,b,K); return;
                        default: this->operator()<I + 1,false,M>(c,wc,a,b,K,mr,nr); return;
                    }
                }
            }
        }
        

    private:
        
        template<std::size_t M, std::size_t N, typename ValueType, typename SizeType>
        AMT_ALWAYS_INLINE void helper(
            ValueType* c, SizeType const* wc,
            ValueType const* const a,
            ValueType const* const b,
            SizeType const K
        ) const noexcept{
            ValueType buff[MR*NR] = {0};

            for(auto k = 0ul; k < K; ++k){
                auto ak = a + k * M;
                auto bk = b + k * N;
                #pragma omp simd
                for(auto j = 0ul; j < N; ++j){
                    auto ai = ak;
                    auto bval = bk[j];
                    auto ci = buff + j * MR;
                    for(auto i = 0ul; i < M; ++i){
                        ci[i] += ai[i] * bval;
                    }
                }
            }
            copy_vec<M,N>(c,wc,buff);
        }

        template<std::size_t M, std::size_t N, typename ValueType, typename SizeType>
        AMT_ALWAYS_INLINE void copy_vec(ValueType* out, SizeType const* wo, ValueType const* in) const noexcept{
            for(auto j = 0ul; j < N; ++j){
                auto ai = in + j * MR;
                auto ci = out + j * wo[1];
                #pragma omp simd
                for(auto i = 0ul; i < M; ++i){
                    ci[i * wo[0]] += ai[i];
                }
            }
        }

    };


} // namespace amt::impl

#endif // AMT_BENCHMARK_SIMD_LOOP_HPP
