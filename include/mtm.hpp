#if !defined(AMT_BENCHMARK_MTV_PRODUCT_HPP)
#define AMT_BENCHMARK_MTV_PRODUCT_HPP

#include <boost/numeric/ublas/tensor.hpp>
#include <optional>
#include <cache_manager.hpp>
#include <utils.hpp>
#include <array>
#include <thread_utils.hpp>
#include <simd_loop.hpp>
#include <sstream>
#include <aligned_buff.hpp>
#include <cpuinfo.hpp>

namespace amt {
    

    namespace impl{
        template<std::size_t VecLen, typename T>
        struct matrix_partition{
            using size_type = std::size_t;
            using value_type = T;
            constexpr static size_type data_size = sizeof(value_type);
            constexpr static size_type num_of_el_in_vec_reg = VecLen / ( data_size * CHAR_BIT);
            constexpr static size_type mr = calculate_mr<value_type,VecLen,CPUFamily::INTEL_SKYLAKE>();
            constexpr static size_type nr = calculate_nr<value_type,VecLen,CPUFamily::INTEL_SKYLAKE>() + 1;

            constexpr static size_type mc() noexcept{
                auto sz = cache_manager::size(1) / (data_size * 6);
                return nearest_power_of_two(sz / kc());
            }
            
            constexpr static size_type nc() noexcept{
                auto sz = cache_manager::size(2) / (data_size * 6);
                return nearest_power_of_two(sz / kc());
            }
            
            constexpr static size_type kc() noexcept{
                auto sz = cache_manager::size(0) / (data_size << 1);
                return nearest_power_of_two(sz / mr);
            }
        };
        
        template<typename OutLayout, typename PartitionType, typename ValueType, typename SizeType>
        AMT_ALWAYS_INLINE void mtm_kernel(
            ValueType* c,       SizeType const* wc,
            ValueType const* a,
            ValueType const* b,
            SizeType const M,   SizeType const N,   SizeType const K
        ) noexcept
        {
            constexpr auto MR = PartitionType::mr;
            constexpr auto NR = PartitionType::nr;
            constexpr auto simd_type = SIMD_PROD_TYPE::MTM;
            constexpr auto loop = simd_loop<simd_type,MR,NR>{};
            for(auto j = 0ul; j < N; j += NR){
                auto ai = a;
                auto bi = b + j * K;
                auto ci = c + wc[1] * j;
                auto jb = std::min(NR,N-j);
                for(auto i = 0ul; i < M; i += MR){
                    auto ak = ai + i * K;
                    auto bk = bi;
                    auto ck = ci + wc[0] * i;
                    auto ib = std::min(MR,M-i);
                    loop(ck,wc,ak,bk,K,ib,jb, OutLayout{});
                }
            }
        }


    } // namespace impl
    

    template<typename OutLayout, typename ValueType, typename SizeType>
    AMT_ALWAYS_INLINE void mtm_helper(
        ValueType* c, [[maybe_unused]] SizeType const* nc, [[maybe_unused]] SizeType const* wc,
        ValueType const* a, [[maybe_unused]] SizeType const* na, [[maybe_unused]] SizeType const* wa,
        ValueType const* b, [[maybe_unused]] SizeType const* nb, [[maybe_unused]] SizeType const* wb,
        OutLayout
    )
    {
        SizeType const WC0 = wc[0];
        SizeType const WC1 = wc[1];
        SizeType const WA0 = wa[0];
        SizeType const WA1 = wa[1];
        SizeType const WB0 = wb[0];
        SizeType const WB1 = wb[1];
        // TODO: Add a way to get CPU vector register length
        using partition_type = impl::matrix_partition<256ul,ValueType>;

        auto M = na[0];
        auto K = na[1];
        auto N = nb[1];

        auto aj = a;
        auto bj = b;
        auto cj = c;

        static auto const MB = partition_type::mc();
        static auto const NB = partition_type::nc();
        static auto const KB = partition_type::kc();
        constexpr static auto const NR = partition_type::nr;
        constexpr static auto const MR = partition_type::mr;

        static std::size_t const buffA_sz = KB * ( MB + 1ul ) * static_cast<std::size_t>(threads::get_max_threads());
        static std::size_t const buffB_sz = KB * ( NB + 1ul );

        static aligned_buff<ValueType> buffA(buffA_sz);
        static aligned_buff<ValueType> buffB(buffB_sz);

        auto const pA = buffA.data();
        auto const pB = buffB.data();

        #pragma omp parallel
        {
            for(auto j = 0ul; j < N; j += NB){
                auto const ak = aj;
                auto const bk = bj + WB1 * j;
                auto const ck = cj + WC1 * j;
                auto const jb = std::min(NB,N-j);
                for(auto k = 0ul; k < K; k += KB){
                    auto const ai = ak + WA1 * k;
                    auto const bi = bk + WB0 * k;
                    auto const ci = ck;
                    auto const kb = std::min(KB,K-k);
                    
                    #pragma omp for schedule(dynamic)
                    for(auto jj = 0ul; jj < jb; jj += NR){
                        auto jjb = std::min(jb-jj,NR);
                        auto const pBjj = pB + jj * kb;
                        auto const bijj = bi + jj * WB1;

                        pack(pBjj, jjb,
                            bijj, wb,
                            jjb, kb,
                            tag::trans{}
                        );
                    }

                    #pragma omp for schedule(dynamic)
                    for(auto i = 0ul; i < M; i += MB){
                        auto const tid = threads::get_thread_num<std::size_t>();
                        auto const ib = std::min(MB,M-i);
                        auto const aii = ai + WA0 * i;
                        auto const aptr = pA + tid * kb * MB;
                        auto const bptr = pB;
                        auto const cptr = ci + WC0 * i;

                        for(auto ii = 0ul; ii < ib; ii += MR){
                            auto iib = std::min(ib-ii,MR);
                            auto const pAii = aptr + ii * kb;
                            auto const apii = aii + ii * WA0;
                            pack(pAii, iib, 
                                apii, wa,
                                iib, kb
                            );
                        }
                        impl::mtm_kernel<OutLayout,partition_type>(cptr,wc,aptr,bptr,ib,jb,kb);
                    }
                } 
            }
        }

    }

    template<typename Out, typename E1, typename E2>
    constexpr auto mtm(
        boost::numeric::ublas::tensor_core<Out>& c,
        boost::numeric::ublas::tensor_core<E1> const& a,
        boost::numeric::ublas::tensor_core<E2> const& b,
        std::optional<std::size_t> num_threads
    )
    {
        using out_type          = boost::numeric::ublas::tensor_core<Out>;
        using tensor_type1      = boost::numeric::ublas::tensor_core<E1>;
        using tensor_type2      = boost::numeric::ublas::tensor_core<E2>;
        using value_type1       = typename tensor_type1::value_type;
        using value_type2       = typename tensor_type2::value_type;
        using out_value_type    = typename out_type::value_type;
        using out_layout_type   = typename out_type::layout_type;

        static_assert(
            std::is_same_v< value_type1, value_type2 > && 
            std::is_same_v< out_value_type, value_type2 >,
            "both tensor type and result type must be of same value_type"
        );

        auto const& na = a.extents();
        auto const& nb = b.extents();
        auto const& nc = c.extents();

        if( !( boost::numeric::ublas::is_matrix(na) && boost::numeric::ublas::is_matrix(nb) && boost::numeric::ublas::is_matrix(nc) ) ) {
            throw std::runtime_error(
                "amt::mtv(boost::numeric::ublas::tensor_core<Out>& c, boost::numeric::ublas::tensor_core<E1> const& a, boost::numeric::ublas::tensor_core<E2> const& b) : "
                "a, b, and c must be the matrices"
            );
        }
        
        threads::clip_num_threads(num_threads);
        
        bool has_no_dim_err = (na[0] == nc[0]) && (na[1] == nb[0]) && (nc[1] == nb[1]);

        if( !has_no_dim_err ){
            throw std::runtime_error(
                "amt::mtv(boost::numeric::ublas::tensor_core<Out>&, boost::numeric::ublas::tensor_core<E1> const&, boost::numeric::ublas::tensor_core<E2> const&) : "
                "dimension mismatch"
            );
        }

        auto* c_ptr = c.data();
        auto const* a_ptr = a.data();
        auto const* b_ptr = b.data();
        auto const* wc_ptr = c.strides().data();
        auto const* wa_ptr = a.strides().data();
        auto const* wb_ptr = b.strides().data();
        auto const* nc_ptr = nc.data();
        auto const* na_ptr = na.data();
        auto const* nb_ptr = nb.data();

        return [c_ptr,a_ptr,b_ptr,wc_ptr,wa_ptr,wb_ptr,nc_ptr,na_ptr,nb_ptr]{
            mtm_helper(c_ptr, nc_ptr, wc_ptr, a_ptr, na_ptr, wa_ptr, b_ptr, nb_ptr, wb_ptr, 
                out_layout_type{}
            );
        };
    }

} // namespace amt

#endif // AMT_BENCHMARK_MTV_PRODUCT_HPP
