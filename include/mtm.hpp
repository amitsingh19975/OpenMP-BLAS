#if !defined(AMT_BENCHMARK_MTV_PRODUCT_HPP)
#define AMT_BENCHMARK_MTV_PRODUCT_HPP

#include <boost/numeric/ublas/tensor.hpp>
#include <optional>
#include <cache_manager.hpp>
#include <utils.hpp>
#include <array>
#include <thread_utils.hpp>
#include <simd_loop.hpp>

namespace amt {

    namespace impl{
        
        template<std::size_t CW0, std::size_t CW1, std::size_t AW0, std::size_t AW1, std::size_t BW0, std::size_t BW1>
        struct matrix_strides_traits{
            constexpr static std::size_t cw0 = CW0;
            constexpr static std::size_t cw1 = CW1;
            constexpr static std::size_t aw0 = AW0;
            constexpr static std::size_t aw1 = AW1;
            constexpr static std::size_t bw0 = BW0;
            constexpr static std::size_t bw1 = BW1;
        };

        template<typename LayoutType1, typename LayoutType2, typename OutLayoutType>
        constexpr auto get_strides_traits() noexcept{
            if constexpr(is_first_order_v<LayoutType1> && is_first_order_v<LayoutType2>){
                if constexpr(is_first_order_v<OutLayoutType>) 
                    return matrix_strides_traits<0,1,0,1,0,1>{};
                else 
                    return matrix_strides_traits<1,0,0,1,0,1>{};
            }else if constexpr(is_last_order_v<LayoutType1> && is_first_order_v<LayoutType2>){
                if constexpr(is_first_order_v<OutLayoutType>) 
                    return matrix_strides_traits<0,1,1,0,0,1>{};
                else 
                    return matrix_strides_traits<1,0,1,0,0,1>{};
            }else if constexpr(is_first_order_v<LayoutType1> && is_last_order_v<LayoutType2>){
                if constexpr(is_first_order_v<OutLayoutType>) 
                    return matrix_strides_traits<0,1,0,1,1,0>{};
                else 
                    return matrix_strides_traits<1,0,0,1,1,0>{};
            }else{
                if constexpr(is_first_order_v<OutLayoutType>) 
                    return matrix_strides_traits<0,1,1,0,1,0>{};
                else 
                    return matrix_strides_traits<1,0,1,0,1,0>{};
            }
        }

    } // namespace impl
    

    template<typename ValueType, typename SizeType, typename StridesTraits>
    AMT_ALWAYS_INLINE void mtm_helper(
        ValueType* c, [[maybe_unused]] SizeType const* nc, [[maybe_unused]] SizeType const* wc,
        ValueType const* a, [[maybe_unused]] SizeType const* na, [[maybe_unused]] SizeType const* wa,
        ValueType const* b, [[maybe_unused]] SizeType const* nb, [[maybe_unused]] SizeType const* wb,
        [[maybe_unused]] int max_threads,
        StridesTraits&&
        // boost::numeric::ublas::layout::last_order
    ) noexcept
    {
        (void)c;
        (void)a;
        (void)b;
        
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

        return [c_ptr,a_ptr,b_ptr,wc_ptr,wa_ptr,wb_ptr,nc_ptr,na_ptr,nb_ptr,nths]{
            mtm_helper(c_ptr, nc_ptr, wc_ptr, a_ptr, na_ptr, wa_ptr, b_ptr, nb_ptr, wb_ptr, nths,
                impl::get_strides_traits<layout_type1,layout_type2,out_layout_type>()
            );
        };
    }

} // namespace amt

#endif // AMT_BENCHMARK_MTV_PRODUCT_HPP
