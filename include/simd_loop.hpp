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
        MTM,
        TRANS
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
        using first_order = boost::numeric::ublas::layout::first_order;
        using last_order = boost::numeric::ublas::layout::last_order;
        
        template<typename OutLayout, typename ValueType, typename SizeType>
        AMT_ALWAYS_INLINE constexpr void operator()(
            ValueType* c, SizeType const ldc,
            ValueType const* a,
            ValueType const* b,
            SizeType const K,
            SizeType const mr,
            SizeType const nr,
            OutLayout
        ) const noexcept{
            if constexpr(std::is_same_v<ValueType,double> && is_last_order_v<OutLayout>){
                helper_double_last_order<OutLayout>(c,ldc,a,b,K,mr,nr);
            }else{
                helper<OutLayout>(c,ldc,a,b,K,mr,nr);
            }
        }
        

    private:
        
        template<typename OutLayout, typename ValueType, typename SizeType>
        AMT_ALWAYS_INLINE void helper(
            ValueType* c, SizeType const ldc,
            ValueType const* a,
            ValueType const* b,
            SizeType const K,
            SizeType const mr,
            SizeType const nr
        ) const noexcept{

            ValueType buff[MR * NR] = {0};
            auto ak = a;
            auto bk = b;
            constexpr auto halfNR = NR >> 1;
            for(auto k = 0ul; k < K; ++k, ak += mr, bk += nr){
                for(auto j = 0ul; j < halfNR; ++j){
                    #pragma omp simd aligned(ak)
                    for(auto i = 0ul; i < MR; ++i){
                        buff[(j + 0)        * MR + i] += bk[j + 0]        * ak[i];
                        buff[(j + halfNR)   * MR + i] += bk[j + halfNR]   * ak[i];
                    }
                }
            }
            copy_from_buff(c,ldc,buff,mr,nr,OutLayout{});
        }

        template<typename OutLayout, typename SizeType>
        AMT_ALWAYS_INLINE void helper_double_last_order(
            double* c, SizeType const ldc,
            double const* a,
            double const* b,
            SizeType const K,
            SizeType const mr,
            SizeType const nr
        ) const noexcept{
            
            double buff[MR * NR] = {0};

            auto ak = a;
            auto bk = b;
            constexpr auto halfMR = MR >> 1;
            for(auto k = 0ul; k < K; ++k, ak += mr, bk += nr){
                for(auto i = 0ul; i < halfMR; ++i){
                    #pragma omp simd aligned(bk)
                    for(auto j = 0ul; j < NR; ++j){
                        buff[j + ( i + 0 )      * NR] += bk[j] * ak[i + 0];
                        buff[j + ( i + halfMR ) * NR] += bk[j] * ak[i + halfMR];
                    }
                }
            }

            copy_from_buff(c,ldc,buff,mr,nr,OutLayout{});
        }

        template<typename ValueType, typename SizeType>
        AMT_ALWAYS_INLINE void copy_from_buff(ValueType* out, SizeType const wo, ValueType const* in, SizeType const mr, SizeType const nr, first_order) const noexcept{
            auto aj = in;
            auto cj = out;
            for(auto j = 0ul; j < nr; ++j, aj += MR, cj += wo){
                auto ai = aj;
                auto ci = cj;
                #pragma omp simd
                for(auto i = 0ul; i < mr; ++i){
                    ci[i] += ai[i];
                }
            }
        }

        template<typename ValueType, typename SizeType>
        AMT_ALWAYS_INLINE void copy_from_buff(ValueType* out, SizeType const wo, ValueType const* in, SizeType const mr, SizeType const nr, last_order) const noexcept{
            constexpr auto is_float = std::is_same_v<ValueType,float>;
            constexpr auto wa0 = ( is_float ? 1ul : NR );
            constexpr auto wa1 = ( is_float ? MR : 1ul );

            auto ai = in;
            auto ci = out;
            for(auto i = 0ul; i < mr; ++i, ai += wa0, ci += wo){
                auto aj = ai;
                auto cj = ci;
                #pragma omp simd
                for(auto j = 0ul; j < nr; ++j){
                    cj[j] += aj[j * wa1];
                }
            }
        }

    };
    template<>
    struct simd_loop<SIMD_PROD_TYPE::TRANS>{
        constexpr static auto type = SIMD_PROD_TYPE::TRANS;
        
        template<typename ValueType, typename SizeType>
        AMT_ALWAYS_INLINE void operator()(
            ValueType* c, SizeType const* wc,
            ValueType* a, SizeType const* wa,
            SizeType const mr,
            SizeType const nr,
            bool has_same_block,
            tag::inplace
        ) const noexcept{
            auto WC0 = wc[0];
            auto WC1 = wc[1];
            auto WA0 = wa[0];
            auto WA1 = wa[1];

            for(auto i = 0ul; i < mr && nr > (i * has_same_block); ++i){
                auto cj = c + WC1 * i;
                auto aj = a + WA0 * i;
                auto idx = i * has_same_block;
                #pragma omp simd
                for(auto j = idx; j < nr; ++j){
                    std::swap(cj[j * WC0],aj[j * WA1]);
                }
            }

        }

        template<typename ValueType, typename SizeType>
        AMT_ALWAYS_INLINE void operator()(
            ValueType* c, SizeType const* wc,
            ValueType const* a, SizeType const* wa,
            SizeType const mr,
            SizeType const nr,
            tag::outplace
        ) const noexcept{
            for(auto i = 0ul; i < mr && nr > 0ul; ++i){
                auto cj = c + wc[1] * i;
                auto aj = a + wa[0] * i;
                #pragma omp simd
                for(auto j = 0ul; j < nr; ++j){
                    cj[j * wc[0]] = aj[j * wa[1]];
                }
            }
        }
    };


} // namespace amt::impl

#endif // AMT_BENCHMARK_SIMD_LOOP_HPP
