#if !defined(AMT_BENCHMARK_MTM_HPP)
#define AMT_BENCHMARK_MTM_HPP

#include <cstdint>
#include <tuple>
#include <new>
#include <type_traits>
#include <boost/numeric/ublas/tensor.hpp>
#include <vector>
#include <omp.h>
#include <thread>

namespace amt{

template<typename T>
concept Pointer = std::is_pointer_v<T>;
    
template<Pointer DataType, Pointer Dim = std::size_t*>
constexpr void print( DataType __restrict in, const Dim __restrict w, const Dim __restrict n ){

    auto in0 = in;
    std::cerr<<'\n';
    for(auto i = 0ul; i < n[0]; ++i){
        auto in1 = in0 + w[0] * i;
        for(auto j = 0ul; j < n[1]; ++j){
            // printf("%03d ",static_cast<int>(in1[j * w[1]]) );
            std::cerr<<in1[j * w[1]] << ' ';
        }
        std::cerr<<'\n';
    }

}

template<Pointer PointerIn, Pointer PointerOut, Pointer Dim = std::size_t*>
void mat_mul_col(
        PointerOut __restrict c, const Dim __restrict sc, [[maybe_unused]] const Dim __restrict wc,
        PointerIn __restrict a, const Dim __restrict sa, const Dim __restrict wa,
        PointerIn __restrict b, const Dim __restrict sb, const Dim __restrict wb
    ){

    auto M = wa[0];
    auto K = wa[1];
    auto P = wb[0];
    auto N = wb[1];

    if(K != P) exit(1);
    [[maybe_unused]] constexpr auto cache_line = alignof(std::max_align_t) << 2u;
    using out_type = std::remove_pointer_t<PointerOut>;
    constexpr auto out_align = std::alignment_of_v<out_type>;

    constexpr auto Kb = 8ul;

    #pragma omp parallel firstprivate(a,b,c,sa,sb,sc,M,N,K)
    {
        auto K_iter = K / Kb;
        auto K_rem = K % Kb;

        #pragma omp for schedule(dynamic)
        for(auto j = 0ul; j < N; ++j){
            auto ci = c + j * sc[1];
            auto bk = b + j * sb[1];
            for(auto k = 0ul; k < K_iter; ++k){
                auto bkk = bk + k * sb[0] * Kb;
                auto akk = a + k * sa[1] * Kb;
                auto i = 0ul;
                #pragma omp simd safelen(cache_line) aligned(akk,bkk,ci:out_align)
                for(i = 0ul; i < M; ++i){
                    ci[i * sc[0]] += akk[i * sa[0] + 0 * sa[1]] * bkk[ 0 * sb[0] ];
                    ci[i * sc[0]] += akk[i * sa[0] + 1 * sa[1]] * bkk[ 1 * sb[0] ];
                    ci[i * sc[0]] += akk[i * sa[0] + 2 * sa[1]] * bkk[ 2 * sb[0] ];
                    ci[i * sc[0]] += akk[i * sa[0] + 3 * sa[1]] * bkk[ 3 * sb[0] ];
                    ci[i * sc[0]] += akk[i * sa[0] + 4 * sa[1]] * bkk[ 4 * sb[0] ];
                    ci[i * sc[0]] += akk[i * sa[0] + 5 * sa[1]] * bkk[ 5 * sb[0] ];
                    ci[i * sc[0]] += akk[i * sa[0] + 6 * sa[1]] * bkk[ 6 * sb[0] ];
                    ci[i * sc[0]] += akk[i * sa[0] + 7 * sa[1]] * bkk[ 7 * sb[0] ];
                }
            }
        }
        
        // #pragma omp barrier
        if(K_rem){
            a += K_iter * Kb * sa[1];
            b += K_iter * Kb * sb[0];

            #pragma omp for schedule(dynamic) nowait
            for(auto j = 0ul; j < N; ++j){
                auto ci = c + j * sc[1];
                auto bk = b + j * sb[1];
                for(auto k = 0ul; k < K_rem; ++k){
                    auto b_val = bk[k * sb[0]];
                    auto ai = a + k * sa[1];
                    #pragma omp simd safelen(cache_line) aligned(ai,ci:out_align)
                    for(auto i = 0ul; i < M; ++i){
                        ci[i * sc[0]] += ai[i * sa[0]] * b_val;
                    }
                }
            }
        }
    }


}

template<typename T>
struct matrix_partition{
    using size_type = std::size_t;
    static constexpr size_type L1 = 32 * 1024; // 32 KB
    static constexpr size_type L2 = 256 * 1024; // 256 KB
    static constexpr size_type L3 = 16 * 1024 * 1024; // 16 MB
    static constexpr size_type cache_line = alignof(std::max_align_t) << 1u; // 64B
    static constexpr size_type bits = CHAR_BIT;
    static constexpr size_type block_m = 8ul;
    static constexpr size_type block_n = 8ul;
    static constexpr size_type block_k = 8ul;
    
    constexpr size_type n() const noexcept{
        return 2048;//4 * 1024ul / ( sizeof(T) * bits);
    }

    constexpr size_type k() const noexcept{
        return 256;// / ( sizeof(T) * bits);
    }

    constexpr size_type m() const noexcept{
        return 96;//128ul / ( sizeof(T) * bits);
    }

};

template<typename PartitionType, Pointer DataType, Pointer Dim = std::size_t*>
__attribute__((always_inline))
void mat_mul_col_block_micro(
        DataType __restrict c, const Dim __restrict wc, const Dim __restrict nc,
        DataType __restrict a, const Dim __restrict wa, const Dim __restrict na,
        DataType __restrict b, const Dim __restrict wb, const Dim __restrict nb,
        PartitionType
    ){
    
    [[maybe_unused]] auto M = nc[0];
    [[maybe_unused]] auto N = nc[0];
    [[maybe_unused]] auto K = na[1];
    [[maybe_unused]] auto P = nb[1];

    [[maybe_unused]] auto a0 = a;
    [[maybe_unused]] auto b0 = b;
    [[maybe_unused]] auto c0 = c;

    
    // std::cerr<<"---------====---------\n";
    // print(a, wa, na);
    // print(b, wb, nb);
    // std::cerr<<"---------====---------\n";

    using data_type = std::remove_pointer_t<DataType>;
    [[maybe_unused]] constexpr auto alignment = alignof(data_type);

    #define INNER_LOOP(K) \
    { \
            c[0 * wc[0] + 0 * wc[1]] += a[0 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 0 * wb[0]];\
            c[1 * wc[0] + 0 * wc[1]] += a[1 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 0 * wb[0]];\
            c[2 * wc[0] + 0 * wc[1]] += a[2 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 0 * wb[0]];\
            c[3 * wc[0] + 0 * wc[1]] += a[3 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 0 * wb[0]];\
            c[4 * wc[0] + 0 * wc[1]] += a[4 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 0 * wb[0]];\
            c[5 * wc[0] + 0 * wc[1]] += a[5 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 0 * wb[0]];\
            c[6 * wc[0] + 0 * wc[1]] += a[6 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 0 * wb[0]];\
            c[7 * wc[0] + 0 * wc[1]] += a[7 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 0 * wb[0]];\
    }\
    { \
            c[0 * wc[0] + 1 * wc[1]] += a[0 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 1 * wb[0]];\
            c[1 * wc[0] + 1 * wc[1]] += a[1 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 1 * wb[0]];\
            c[2 * wc[0] + 1 * wc[1]] += a[2 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 1 * wb[0]];\
            c[3 * wc[0] + 1 * wc[1]] += a[3 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 1 * wb[0]];\
            c[4 * wc[0] + 1 * wc[1]] += a[4 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 1 * wb[0]];\
            c[5 * wc[0] + 1 * wc[1]] += a[5 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 1 * wb[0]];\
            c[6 * wc[0] + 1 * wc[1]] += a[6 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 1 * wb[0]];\
            c[7 * wc[0] + 1 * wc[1]] += a[7 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 1 * wb[0]];\
    }\
    { \
            c[0 * wc[0] + 2 * wc[1]] += a[0 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 2 * wb[0]];\
            c[1 * wc[0] + 2 * wc[1]] += a[1 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 2 * wb[0]];\
            c[2 * wc[0] + 2 * wc[1]] += a[2 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 2 * wb[0]];\
            c[3 * wc[0] + 2 * wc[1]] += a[3 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 2 * wb[0]];\
            c[4 * wc[0] + 2 * wc[1]] += a[4 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 2 * wb[0]];\
            c[5 * wc[0] + 2 * wc[1]] += a[5 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 2 * wb[0]];\
            c[6 * wc[0] + 2 * wc[1]] += a[6 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 2 * wb[0]];\
            c[7 * wc[0] + 2 * wc[1]] += a[7 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 2 * wb[0]];\
    }\
    { \
            c[0 * wc[0] + 3 * wc[1]] += a[0 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 3 * wb[0]];\
            c[1 * wc[0] + 3 * wc[1]] += a[1 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 3 * wb[0]];\
            c[2 * wc[0] + 3 * wc[1]] += a[2 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 3 * wb[0]];\
            c[3 * wc[0] + 3 * wc[1]] += a[3 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 3 * wb[0]];\
            c[4 * wc[0] + 3 * wc[1]] += a[4 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 3 * wb[0]];\
            c[5 * wc[0] + 3 * wc[1]] += a[5 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 3 * wb[0]];\
            c[6 * wc[0] + 3 * wc[1]] += a[6 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 3 * wb[0]];\
            c[7 * wc[0] + 3 * wc[1]] += a[7 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 3 * wb[0]];\
    }\
    { \
            c[0 * wc[0] + 4 * wc[1]] += a[0 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 4 * wb[0]];\
            c[1 * wc[0] + 4 * wc[1]] += a[1 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 4 * wb[0]];\
            c[2 * wc[0] + 4 * wc[1]] += a[2 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 4 * wb[0]];\
            c[3 * wc[0] + 4 * wc[1]] += a[3 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 4 * wb[0]];\
            c[4 * wc[0] + 4 * wc[1]] += a[4 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 4 * wb[0]];\
            c[5 * wc[0] + 4 * wc[1]] += a[5 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 4 * wb[0]];\
            c[6 * wc[0] + 4 * wc[1]] += a[6 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 4 * wb[0]];\
            c[7 * wc[0] + 4 * wc[1]] += a[7 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 4 * wb[0]];\
    }\
    { \
            c[0 * wc[0] + 5 * wc[1]] += a[0 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 5 * wb[0]];\
            c[1 * wc[0] + 5 * wc[1]] += a[1 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 5 * wb[0]];\
            c[2 * wc[0] + 5 * wc[1]] += a[2 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 5 * wb[0]];\
            c[3 * wc[0] + 5 * wc[1]] += a[3 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 5 * wb[0]];\
            c[4 * wc[0] + 5 * wc[1]] += a[4 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 5 * wb[0]];\
            c[5 * wc[0] + 5 * wc[1]] += a[5 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 5 * wb[0]];\
            c[6 * wc[0] + 5 * wc[1]] += a[6 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 5 * wb[0]];\
            c[7 * wc[0] + 5 * wc[1]] += a[7 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 5 * wb[0]];\
    }\
    { \
            c[0 * wc[0] + 6 * wc[1]] += a[0 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 6 * wb[0]];\
            c[1 * wc[0] + 6 * wc[1]] += a[1 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 6 * wb[0]];\
            c[2 * wc[0] + 6 * wc[1]] += a[2 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 6 * wb[0]];\
            c[3 * wc[0] + 6 * wc[1]] += a[3 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 6 * wb[0]];\
            c[4 * wc[0] + 6 * wc[1]] += a[4 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 6 * wb[0]];\
            c[5 * wc[0] + 6 * wc[1]] += a[5 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 6 * wb[0]];\
            c[6 * wc[0] + 6 * wc[1]] += a[6 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 6 * wb[0]];\
            c[7 * wc[0] + 6 * wc[1]] += a[7 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 6 * wb[0]];\
    }\
    { \
            c[0 * wc[0] + 7 * wc[1]] += a[0 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 7 * wb[0]];\
            c[1 * wc[0] + 7 * wc[1]] += a[1 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 7 * wb[0]];\
            c[2 * wc[0] + 7 * wc[1]] += a[2 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 7 * wb[0]];\
            c[3 * wc[0] + 7 * wc[1]] += a[3 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 7 * wb[0]];\
            c[4 * wc[0] + 7 * wc[1]] += a[4 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 7 * wb[0]];\
            c[5 * wc[0] + 7 * wc[1]] += a[5 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 7 * wb[0]];\
            c[6 * wc[0] + 7 * wc[1]] += a[6 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 7 * wb[0]];\
            c[7 * wc[0] + 7 * wc[1]] += a[7 * wa[0] + (K) * wa[1]] * b[(K) * wb[1] + 7 * wb[0]];\
    }\

    for(auto k = 0ul; k < K; k += PartitionType::block_k){
        INNER_LOOP(k);
        INNER_LOOP(k + 1);
        INNER_LOOP(k + 2);
        INNER_LOOP(k + 3);
        INNER_LOOP(k + 4);
        INNER_LOOP(k + 5);
        INNER_LOOP(k + 6);
        INNER_LOOP(k + 7);
    }

}

template<typename PartitionType, Pointer DataType, Pointer Dim = std::size_t*>
__attribute__((always_inline))
void mat_mul_col_block_macro(
        DataType __restrict c, const Dim __restrict wc, const Dim __restrict nc,
        DataType __restrict a, const Dim __restrict wa, const Dim __restrict na,
        DataType __restrict b, const Dim __restrict wb, const Dim __restrict nb,
        [[maybe_unused]] PartitionType par
    ){
    
    auto M = nc[0];
    auto N = nc[1];
    auto K = na[1];
    [[maybe_unused]] auto P = nb[0];

    using dim_type = std::remove_pointer_t<Dim>;

    [[maybe_unused]] constexpr auto MB = PartitionType::block_m;
    [[maybe_unused]] constexpr auto NB = PartitionType::block_n;
    [[maybe_unused]] constexpr auto KB = PartitionType::block_k;

    auto a0 = a;
    auto b0 = b;
    auto c0 = c;

    for(auto j = 0ul; j < N; j += NB){
        auto jb = std::min(N - j, NB);
        auto a1 = a0;
        auto b1 = b0 + j * wb[1];
        auto c1 = c0 + j * wc[1];
        for(auto i = 0ul; i < M; i += MB){
            auto ib = std::min(M - i, MB);
            auto a2 = a1 + i * wa[1];
            auto b2 = b1;
            auto c2 = c1 + i * wa[0];
            dim_type const nna[] = {ib, K};
            dim_type const nnb[] = {jb, K};
            dim_type const nnc[] = {ib, jb};
            dim_type const nwa[] = {1, MB};
            dim_type const nwb[] = {1, NB};
            mat_mul_col_block_micro(c2, wc, nnc, a2, nwa, nna, b2, nwb, nnb, par);
        }
        
    }

}

namespace tag{
    struct trans{};
}

template<Pointer DataType, Pointer Dim = std::size_t*>
constexpr void pack(
    DataType __restrict out, const Dim __restrict wo,
    const DataType __restrict in, const Dim __restrict wi, const Dim __restrict n
) noexcept{

    auto in0 = in;
    auto out0 = out;
    for(auto i = 0ul; i < n[1]; ++i){
        auto in1 = in0 + wi[1] * i;
        auto out1 = out0 + wo[1] * i;
        for(auto j = 0ul; j < n[0]; ++j){
            out1[j * wo[0]] = in1[j * wi[0]];
        }
    }

}

template<Pointer DataType, Pointer Dim = std::size_t*>
constexpr void pack(
    DataType __restrict out, const Dim __restrict wo,
    const DataType __restrict in, const Dim __restrict wi, const Dim __restrict n, tag::trans /*transpose*/
) noexcept{

    auto in0 = in;
    auto out0 = out;
    for(auto i = 0ul; i < n[1]; ++i){
        auto in1 = in0 + wi[0] * i;
        auto out1 = out0 + wo[1] * i;
        for(auto j = 0ul; j < n[0]; ++j){
            out1[j * wo[0]] = in1[j * wi[1]];
        }
    }

}

template<typename PartitionType, Pointer DataType, Pointer Dim = std::size_t*>
void mat_mul_col_block(
        DataType __restrict c, const Dim __restrict wc, const Dim __restrict nc,
        DataType __restrict a, const Dim __restrict wa, const Dim __restrict na,
        DataType __restrict b, const Dim __restrict wb, const Dim __restrict nb,
        PartitionType par
    ){
    
    auto M = nc[0];
    auto N = nc[1];
    auto K = na[1];
    auto P = nb[0];

    using dim_type = std::remove_pointer_t<Dim>;
    using data_type = std::remove_pointer_t<DataType>;

    if(K != P){
        std::cerr<<"Dimensions mismatch : K( " << K << " ) != P( " << P <<" )\n";
        exit(1);
    }

    constexpr auto MB = par.m();
    constexpr auto NB = par.n();
    constexpr auto KB = par.k();

    [[maybe_unused]]constexpr auto pMB = PartitionType::block_m;
    constexpr auto pNB = PartitionType::block_n;

    auto num_of_threads = std::thread::hardware_concurrency();

    std::vector<data_type> dataA( KB * ( MB + 1ul ) * num_of_threads );
    std::vector<data_type> dataB( KB * ( NB + 1ul ) );

    auto a0 = a;
    auto b0 = b;
    auto c0 = c;

    for(auto j = 0ul; j < N; j += NB){
        auto a1 = a0;
        auto b1 = b0 + j * wb[1];
        auto c1 = c0 + j * wc[1];
        auto jb = std::min(N - j, NB);
        for(auto k = 0ul; k < K; k += KB){
            auto a2 = a1 + k * wa[1];
            auto b2 = b1 + k * wb[0];
            auto c2 = c1;
            auto kb = std::min(K - k, KB);

            #pragma omp parallel for schedule(dynamic) num_threads(num_of_threads)
            for(auto jj = 0ul; jj < jb; jj += pNB){
                auto jjb = std::min(jb - jj, pNB);
                dim_type const nnb[] = {kb, jb};
                dim_type const wdB[] = {1ul, jjb};
                auto pB = dataB.data() + jj * kb;
                auto pb = b2 + jj * kb;
                pack(
                    pB,
                    wdB,
                    pb,
                    wb,
                    nnb,
                    tag::trans{}
                );
            }
            
            #pragma omp parallel for schedule(dynamic) num_threads(num_of_threads)
            for(auto i = 0ul; i < M; i += MB){
                auto a3 = a2 + i * wa[0];
                auto c3 = c2 + i * wb[0];
                auto ib = std::min(M - i, MB);
                auto tid = static_cast<std::size_t>(omp_get_thread_num());

                for(auto ii = 0ul; ii < ib; ii += pMB){
                    auto iib = std::min(ib - ii, pMB);
                    dim_type const nna[] = {ib, kb};
                    dim_type const wdA[] = {1ul, iib};
                    auto pA = dataA.data() + ( ii + tid * MB ) * kb;
                    auto pa = a3 + ii * wa[0];
                    pack(
                        pA,
                        wdA,
                        pa,
                        wa,
                        nna
                    );
                }

                // dim_type const nA[] = {kb, jb};
                // dim_type const wA[] = {1ul, jb};
                // print(dataB.data(), wA, nA);
                // dim_type const nA[] = {ib, kb};
                // dim_type const wA[] = {1ul, ib};
                // print(dataA.data() + tid * kb * MB, wA, nA);
                
                // if(false){

                dim_type const nna[] = {ib, kb};
                dim_type const nnb[] = {jb, kb};
                dim_type const nnc[] = {ib, jb};
                dim_type const nwa[] = {1ul, ib};
                dim_type const nwb[] = {1ul, jb};

                auto pc = c3;
                auto pa = dataA.data() + tid * kb * MB;
                auto pb = dataB.data();

                mat_mul_col_block_macro(
                    pc, wc, nnc,
                    pa, nwa, nna,
                    pb, nwb, nnb,
                    par
                );
                // }
            }
        }
    }

}

} // namespace amt


#endif // AMT_BENCHMARK_MTM_HPP
