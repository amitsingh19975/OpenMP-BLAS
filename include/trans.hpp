#if !defined(AMT_BENCHMARK_TRANSPOSE_PRODUCT_HPP)
#define AMT_BENCHMARK_TRANSPOSE_PRODUCT_HPP

#include <boost/numeric/ublas/tensor.hpp>
#include <optional>
#include <cache_manager.hpp>
#include <utils.hpp>
#include <thread_utils.hpp>
#include <simd_loop.hpp>

namespace amt {
    

    namespace impl{
        template<typename T>
        struct transpose_partition{
            using size_type = std::size_t;
            using value_type = T;

            constexpr static auto data_size = sizeof(value_type);

            constexpr static size_type block() noexcept{
                auto size = cache_manager::size(0) / ( data_size << 1 );
                auto const [root, _] = sqrt_pow_of_two(size);
                return root;
            }
            
        };

    } // namespace impl
    

    template<typename ValueType, typename SizeType>
    AMT_ALWAYS_INLINE void transpose_helper(
        ValueType* c, [[maybe_unused]] SizeType const* nc, [[maybe_unused]] SizeType const* wc,
        ValueType const* a, [[maybe_unused]] SizeType const* na, [[maybe_unused]] SizeType const* wa,
        tag::outplace
    )
    {
        static auto const block = impl::transpose_partition<ValueType>::block();
        constexpr auto simd_type = impl::SIMD_PROD_TYPE::TRANS;
        constexpr auto simd_loop = impl::simd_loop<simd_type>{};

        auto const num_threads = threads::get_num_threads();
        auto const max_blocks = std::max(1, static_cast<int>(na[0] / block));
        auto const max_threads = std::min(max_blocks, num_threads);
        
        #pragma omp parallel for num_threads(max_threads) schedule(dynamic) if(na[0] > block)
        for(auto i = 0ul; i < na[0]; i += block){
            auto ib = std::min(na[0] - i, block);
            auto aj = a + i * wa[0];
            auto cj = c + i * wc[1];
            for(auto j = 0ul; j < na[1]; j += block){
                auto jb = std::min(na[1] - j, block);
                auto ak = aj + j * wa[1];
                auto ck = cj + j * wc[0];
                simd_loop(ck, wc, ak, wa, ib, jb, tag::outplace{});
            }
        }

    }

    template<typename ValueType, typename SizeType>
    AMT_ALWAYS_INLINE void transpose_helper(
        ValueType* c, [[maybe_unused]] SizeType const* nc, [[maybe_unused]] SizeType const* wc,
        ValueType* a, [[maybe_unused]] SizeType const* na, [[maybe_unused]] SizeType const* wa,
        tag::inplace
    )
    {
        static auto const block = impl::transpose_partition<ValueType>::block();
        constexpr auto simd_type = impl::SIMD_PROD_TYPE::TRANS;
        constexpr auto simd_loop = impl::simd_loop<simd_type>{};
        
        auto const num_threads = threads::get_num_threads();
        auto const max_blocks = std::max(1, static_cast<int>(na[0] / block));
        auto const max_threads = std::min(max_blocks, num_threads);

        auto ai = a;
        auto ci = c;
        #pragma omp parallel for num_threads(max_threads) schedule(dynamic) if(na[0] > block)
        for(auto i = 0ul; i < na[0]; i += block){
            auto ib = std::min(na[0] - i, block);
            auto aj = ai + i * wa[0];
            auto cj = ci + i * wc[1];
            for(auto j = i; j < na[1]; j += block){
                auto jb = std::min(na[1] - j, block);
                auto ak = aj + j * wa[1];
                auto ck = cj + j * wc[0];
                simd_loop(ck, wc, ak, wa, ib, jb, i == j, tag::inplace{});
            }
        }

    }

    template<typename Out, typename E>
    constexpr auto transpose(
        boost::numeric::ublas::tensor_core<Out>& c,
        boost::numeric::ublas::tensor_core<E> const& a,
        std::optional<std::size_t> num_threads
    )
    {
        using out_type          = boost::numeric::ublas::tensor_core<Out>;
        using tensor_type       = boost::numeric::ublas::tensor_core<E>;
        using value_type        = typename tensor_type::value_type;
        using out_value_type    = typename out_type::value_type;

        static_assert(
            std::is_same_v< out_value_type, value_type >,
            "input value type and result value type must be of same value type"
        );

        auto const& na = a.extents();
        auto const& nc = c.extents();

        if( !( boost::numeric::ublas::is_matrix(na) && boost::numeric::ublas::is_matrix(nc) ) ) {
            throw std::runtime_error(
                "amt::transpose(boost::numeric::ublas::tensor_core<Out>& c, boost::numeric::ublas::tensor_core<E> const& a) : "
                "a and c must be the matrices"
            );
        }
        
        threads::set_num_threads(num_threads);

        if( !( (na[0] == nc[1]) && (na[1] == nc[0]) ) ){
            throw std::runtime_error(
                "amt::transpose(boost::numeric::ublas::tensor_core<Out>& c, boost::numeric::ublas::tensor_core<E> const& a) : "
                "dimension mismatch"
            );
        }

        auto* c_ptr = c.data();
        auto const* a_ptr = a.data();
        auto const* wc_ptr = c.strides().data();
        auto const* wa_ptr = a.strides().data();
        auto const* na_ptr = boost::numeric::ublas::data(na);
        auto const* nc_ptr = boost::numeric::ublas::data(na);

        return [c_ptr,a_ptr,wc_ptr,wa_ptr,na_ptr,nc_ptr]{
            transpose_helper(c_ptr, nc_ptr, wc_ptr, a_ptr, na_ptr, wa_ptr, tag::outplace{});
        };
    }

    template<typename E>
    constexpr auto transpose(
        boost::numeric::ublas::tensor_core<E>& a,
        std::optional<std::size_t> num_threads
    )
    {
        auto const& na = a.extents();

        if( !( boost::numeric::ublas::is_matrix(na) ) ) {
            throw std::runtime_error(
                "amt::transpose(boost::numeric::ublas::tensor_core<E> const& a) : "
                "a must be a matrix"
            );
        }
        
        threads::set_num_threads(num_threads);

        auto* a_ptr = a.data();
        auto const* na_ptr = boost::numeric::ublas::data(na);

        return [a_ptr,na_ptr]{
            using size_type = std::decay_t< std::remove_pointer_t<decltype(na_ptr)> >;
            size_type wc[2] = {1ul, na_ptr[0]};
            size_type wa[2] = {1ul, na_ptr[1]};
            transpose_helper(a_ptr, na_ptr, wc, a_ptr, na_ptr, wa, tag::inplace{});
        };
    }

} // namespace amt

#endif // AMT_BENCHMARK_TRANSPOSE_PRODUCT_HPP
