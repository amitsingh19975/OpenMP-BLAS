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

namespace amt {

    namespace debug
    {   
        template<typename ValueType, typename SizeType>
        constexpr void show( ValueType* in, SizeType const* w, SizeType const* n ){

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
        
        template<typename ValueType, typename SizeType>
        void show( ValueType* in, SizeType const n, char del = '\n'){
            std::stringstream ss;
            ss<<'\n';
            for(auto i = 0ul; i < n; ++i){
                ss<<in[i] << del;
            }
            std::cerr<<ss.str();

        }
    } // namespace debug
    

    namespace impl{
        
        template<std::size_t... Ns>
        struct strides_factor{
            constexpr static std::size_t size = sizeof...(Ns);
            constexpr static std::array<std::size_t,size> factor = {Ns...};
        };

        template<typename LayoutType1, typename LayoutType2, typename OutLayoutType>
        constexpr auto get_strides_factor() noexcept{
            if constexpr(is_first_order_v<LayoutType1> && is_first_order_v<LayoutType2>){
                if constexpr(is_first_order_v<OutLayoutType>) 
                    return strides_factor<0,1,0,1,0,1>{};
                else 
                    return strides_factor<1,0,0,1,0,1>{};
            }else if constexpr(is_last_order_v<LayoutType1> && is_first_order_v<LayoutType2>){
                if constexpr(is_first_order_v<OutLayoutType>) 
                    return strides_factor<0,1,1,0,0,1>{};
                else 
                    return strides_factor<1,0,1,0,0,1>{};
            }else if constexpr(is_first_order_v<LayoutType1> && is_last_order_v<LayoutType2>){
                if constexpr(is_first_order_v<OutLayoutType>) 
                    return strides_factor<0,1,0,1,1,0>{};
                else 
                    return strides_factor<1,0,0,1,1,0>{};
            }else{
                if constexpr(is_first_order_v<OutLayoutType>) 
                    return strides_factor<0,1,1,0,1,0>{};
                else 
                    return strides_factor<1,0,1,0,1,0>{};
            }
        }

        template<std::size_t VecLen, typename T>
        struct matrix_partition{
            using size_type = std::size_t;
            using value_type = T;
            constexpr static size_type const data_size = sizeof(value_type);
            constexpr static size_type const nr = 4ul;//VecLen / ( data_size * CHAR_BIT);
            constexpr static size_type const mr = 32ul;

            // m = (L2/L1)nr
            constexpr static size_type mc() noexcept{
                return 32ul;//cache_manager::size(1) / (kc()*data_size);
            }
            
            // n = (L3/L1)nr
            constexpr static size_type nc() noexcept{
                return 1<<11;//cache_manager::size(2) / (kc()*data_size);
            }
            
            // k = L1/nr
            constexpr static size_type kc() noexcept{
                return 512ul;//cache_manager::size(0) / ( nr * data_size );
            }
        };
        
        template<typename PartitionType, typename ValueType, typename SizeType, typename StridesFactor>
        AMT_ALWAYS_INLINE void mtm_kernel(
            ValueType* c, [[maybe_unused]] SizeType const* nc, [[maybe_unused]] SizeType const* wc,
            ValueType const* a, [[maybe_unused]] SizeType const* na, [[maybe_unused]] SizeType const* wa,
            ValueType const* b, [[maybe_unused]] SizeType const* nb, [[maybe_unused]] SizeType const* wb,
            StridesFactor
            // boost::numeric::ublas::layout::last_order
        ) noexcept
        {
            [[maybe_unused]] SizeType const WC0 = (StridesFactor::factor[0] ? wc[0] : 1ul);
            [[maybe_unused]] SizeType const WC1 = (StridesFactor::factor[1] ? wc[1] : 1ul);
            [[maybe_unused]] SizeType const WA0 = (StridesFactor::factor[2] ? wa[0] : 1ul);
            [[maybe_unused]] SizeType const WA1 = (StridesFactor::factor[3] ? wa[1] : 1ul);
            [[maybe_unused]] SizeType const WB0 = (StridesFactor::factor[4] ? wb[0] : 1ul);
            [[maybe_unused]] SizeType const WB1 = (StridesFactor::factor[5] ? wb[1] : 1ul);
            [[maybe_unused]] constexpr auto MR = PartitionType::mr;
            [[maybe_unused]] constexpr auto NR = PartitionType::nr;
            constexpr auto simd_type = SIMD_PROD_TYPE::MTM;
            constexpr auto loop = simd_loop<simd_type,MR,NR>{};

            auto M = nc[0];
            auto N = nc[1];
            auto K = nb[0];

            for(auto j = 0ul; j < N; j+=NR){
                auto ai = a;
                auto bi = b + WB1 * j;
                auto ci = c + WC1 * j;
                auto jb = std::min(NR,N-j);
                for(auto i = 0ul; i < M; ++i){
                    auto ak = ai + WA1 * i;
                    auto bk = bi;
                    auto ck = ci + WC0 * i;
                    loop(ck,WC1,ak,bk,WB1,K,jb);
                }
            }
        }


    } // namespace impl
    

    template<typename ValueType, typename SizeType, typename StridesFactor>
    AMT_ALWAYS_INLINE void mtm_helper(
        ValueType* c, [[maybe_unused]] SizeType const* nc, [[maybe_unused]] SizeType const* wc,
        ValueType const* a, [[maybe_unused]] SizeType const* na, [[maybe_unused]] SizeType const* wa,
        ValueType const* b, [[maybe_unused]] SizeType const* nb, [[maybe_unused]] SizeType const* wb,
        [[maybe_unused]] int max_threads,
        StridesFactor,
        SizeType ={}
        // boost::numeric::ublas::layout::last_order
    ) noexcept
    {
        SizeType const WC0 = (StridesFactor::factor[0] ? wc[0] : 1ul);
        SizeType const WC1 = (StridesFactor::factor[1] ? wc[1] : 1ul);
        SizeType const WA0 = (StridesFactor::factor[2] ? wa[0] : 1ul);
        SizeType const WA1 = (StridesFactor::factor[3] ? wa[1] : 1ul);
        SizeType const WB0 = (StridesFactor::factor[4] ? wb[0] : 1ul);
        SizeType const WB1 = (StridesFactor::factor[5] ? wb[1] : 1ul);
        using partition_type = impl::matrix_partition<256ul,ValueType>;

        auto M = na[0];
        auto K = na[1];
        auto N = nb[1];


        auto aj = a;
        auto bj = b;
        auto cj = c;

        auto const MB = partition_type::mc();
        auto const NB = partition_type::nc();
        auto const KB = partition_type::kc();
        auto const NR = partition_type::nr;
        auto const MR = partition_type::mr;
        
        std::vector<ValueType> packed_A( KB * ( MB + 1ul ) * static_cast<std::size_t>(max_threads) );
        std::vector<ValueType> packed_B( KB * ( NB + 1ul ) );

        constexpr auto nfactor = impl::strides_factor<
            StridesFactor::factor[0],StridesFactor::factor[1],
            StridesFactor::factor[2], StridesFactor::factor[3], 
            StridesFactor::factor[4], StridesFactor::factor[5]
        >{};

        auto pA = packed_A.data();
        auto pB = packed_B.data();

        // std::cerr<< M<<' '<<N<<' '<<K<<'\n';

        #pragma omp parallel
        {
            for(auto j = 0ul; j < N; j += NB){
                auto ak = aj;
                auto bk = bj + WB1 * j;
                auto ck = cj + WC1 * j;
                auto jb = std::min(NB,N-j);
                for(auto k = 0ul; k < K; k += KB){
                    auto ai = ak + WA1 * k;
                    auto bi = bk + WB0 * k;
                    auto ci = ck;
                    auto kb = std::min(KB,K-k);
                    
                    #pragma omp for schedule(dynamic)
                    for(auto jj = 0ul; jj < jb; jj += NR){
                        auto pBjj = pB + jj * kb;
                        auto bijj = bi + jj * kb;
                        pack(pBjj, kb,
                            bijj, wb,
                            kb,std::min(jb-jj,NR)
                        );
                    }

                    #pragma omp for schedule(dynamic)
                    for(auto i = 0ul; i < M; i += MB){
                        auto aptr = ai + WA0 * i;
                        auto cptr = ci + WC0 * i;
                        auto ib = std::min(MB,M-i);
                        auto tid = static_cast<std::size_t>(omp_get_thread_num());
                        
                        for(auto ii = 0ul; ii < ib; ii += MR){
                            auto pAii = pA + ( ii + tid * MB ) * kb;
                            auto apii = aptr + ii * WA0;
                            pack(pAii, kb, 
                                apii, wa,
                                kb, std::min(ib-ii,MR),
                                tag::trans{}
                            );
                            // debug::show(pAii,WpA,NpA);
                            // debug::show(pAii,packed_A.size() );
                        }

                        SizeType const NNC[] = {ib,jb};
                        SizeType const NNA[] = {kb,ib};
                        SizeType const NNB[] = {kb,jb};
                        SizeType const NWA[] = {1ul,kb};
                        SizeType const NWB[] = {1ul,kb};

                        auto pc = cptr;
                        auto pa = pA + tid * kb * MB;
                        auto pb = pB;
                        // debug::show(pA,NWA,NNA);
                        // exit(0);
                        impl::mtm_kernel<partition_type>(pc,NNC,wc,pa,NNA,NWA,pb,NNB,NWB,nfactor);
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
        std::optional<std::size_t> num_threads,
        std::size_t Block = 64
    )
    {
        using out_type          = boost::numeric::ublas::tensor_core<Out>;
        using tensor_type1      = boost::numeric::ublas::tensor_core<E1>;
        using tensor_type2      = boost::numeric::ublas::tensor_core<E2>;
        using value_type1       = typename tensor_type1::value_type;
        using value_type2       = typename tensor_type2::value_type;
        using out_value_type    = typename out_type::value_type;
        using out_layout_type   = typename out_type::layout_type;
        using layout_type1      = typename tensor_type1::layout_type;
        using layout_type2      = typename tensor_type2::layout_type;

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
        
        threads::set_num_threads(num_threads);
        auto nths = threads::get_num_threads();
        
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

        return [c_ptr,a_ptr,b_ptr,wc_ptr,wa_ptr,wb_ptr,nc_ptr,na_ptr,nb_ptr,nths,Block]{
            mtm_helper(c_ptr, nc_ptr, wc_ptr, a_ptr, na_ptr, wa_ptr, b_ptr, nb_ptr, wb_ptr, nths,
                impl::get_strides_factor<layout_type1,layout_type2,out_layout_type>(),Block
            );
        };
    }

} // namespace amt

#endif // AMT_BENCHMARK_MTV_PRODUCT_HPP
