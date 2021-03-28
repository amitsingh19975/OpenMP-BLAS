#if !defined(AMT_BENCHMARK_THREAD_UTILS_HPP)
#define AMT_BENCHMARK_THREAD_UTILS_HPP

#include <cstdlib>
#include <omp.h>
#include <optional>
#include <type_traits>

namespace amt{
    
    namespace detail{
        
        int get_default_num_threads() noexcept{
            if( char const* omp_env_var = std::getenv("OMP_NUM_THREADS"); omp_env_var != nullptr ){
                return std::atoi(omp_env_var);
            }else{
                return omp_get_max_threads();
            }
        }

        template<typename T>
        struct is_optional : std::false_type{};
        
        template<typename T>
        struct is_optional<std::optional<T>> : std::true_type{};

    } // namespace detail
    
    struct threads{
    private:
        template<typename SizeType>
        constexpr static void set_user_threads(std::optional<SizeType> ths) noexcept{
            m_user_threads = static_cast<int>(ths.value_or(m_default_num_threads));
        }

    public:
        template<typename SizeType>
        constexpr static void set_num_threads(SizeType const& ths) noexcept{
            if constexpr(detail::is_optional<SizeType>::value){
                set_user_threads(std::move(ths));
            }else{
                set_user_threads(std::optional<SizeType>(ths));
            }
            omp_set_num_threads(m_user_threads);
        }
        
        template<typename SizeType = int>
        constexpr static SizeType get_num_threads() noexcept{
            return static_cast<SizeType>(m_user_threads);
        }
        
        template<typename SizeType = int>
        constexpr static SizeType get_max_threads() noexcept{
            return static_cast<SizeType>(omp_get_max_threads());
        }

        static void reset_to_default() noexcept{
            set_num_threads(m_default_num_threads);
        }

    private:
        static int m_user_threads;
        static int m_default_num_threads;
    };

    int threads::m_user_threads = detail::get_default_num_threads();
    int threads::m_default_num_threads = detail::get_default_num_threads();

} // namespace amt


#endif // AMT_BENCHMARK_THREAD_UTILS_HPP
