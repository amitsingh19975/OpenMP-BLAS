#if !defined(AMT_BENCHMARK_ALIGNED_BUFFER_HPP)
#define AMT_BENCHMARK_ALIGNED_BUFFER_HPP

#include <cstdlib>
#include <macros.hpp>
#include <type_traits>
#include <algorithm>
#include <new>

namespace amt{

    template<typename ValueType, std::size_t Alignment = 32ul>
    struct alinged_buff{
        using value_type    = ValueType;
        using size_type     = std::size_t;
        using alignment_type= std::align_val_t;
        using pointer       = std::add_pointer_t<value_type>;
        using const_pointer = std::add_const_t<pointer>;

        constexpr static alignment_type alignment = alignment_type{Alignment};
        
        constexpr alinged_buff(size_type sz)
            : m_ptr(new(alignment) value_type[sz])
            , m_size(sz)
        {
            if(!m_ptr){
                throw std::runtime_error(
                    "amt::alinged_buff(size_type): unable to allocate aligned buffer"
                );
            }
        }
        
        constexpr alinged_buff(size_type sz, value_type val)
            : m_ptr(new(alignment) value_type[sz])
            , m_size(sz)
        {
            auto rem = sz % static_cast<size_type>(alignment);

            std::fill_n(m_ptr,sz + rem,val);
            if(!m_ptr){
                throw std::runtime_error(
                    "amt::alinged_buff(size_type): unable to allocate aligned buffer"
                );
            }
        }

        constexpr pointer data() noexcept{ return m_ptr; }
        constexpr const_pointer data() const noexcept{ return m_ptr; }

        constexpr size_type size() const noexcept {return m_size;}

        constexpr operator bool() const noexcept{
            return m_ptr != nullptr;
        }

        ~alinged_buff() noexcept{
            delete[] m_ptr;
        }

    private:
        pointer     m_ptr{nullptr};
        size_type   m_size{0};
    };

} // namespace amt


#endif // AMT_BENCHMARK_ALIGNED_BUFFER_HPP
