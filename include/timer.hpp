#if !defined(AMT_BENCHMARK_TIMER_HPP)
#define AMT_BENCHMARK_TIMER_HPP

#include <chrono>
#include <ostream>

namespace amt{
    
    struct timer{
        
        using clock_type = std::chrono::high_resolution_clock;
        using base_type = decltype(clock_type::now());

        constexpr double operator()() noexcept{
            if(!m_stopped){
                m_end = clock_type::now();
                m_stopped = true;
            }
            auto diff = (m_end - m_start);
            return std::chrono::duration<double>(diff).count();
        }

        template<typename Unit = std::chrono::milliseconds>
        constexpr auto cast() const noexcept{
            auto diff = (m_end - m_start);
            return std::chrono::duration_cast<Unit>(diff).count();
        }

        constexpr auto milli() const noexcept{
            return cast();
        }

        constexpr auto micro() const noexcept{
            return cast<std::chrono::microseconds>();
        }

        constexpr auto sec() const noexcept{
            return cast<std::chrono::seconds>();
        }

        constexpr auto min() const noexcept{
            return cast<std::chrono::minutes>();
        }

        template<typename Unit = std::chrono::milliseconds>
        std::string str() const{
            if constexpr( std::is_same_v< Unit, std::chrono::microseconds > ){
                return std::to_string(micro()) + "us";
            }else if constexpr( std::is_same_v< Unit, std::chrono::milliseconds > ){
                return std::to_string(milli()) + "ms";
            }else if constexpr( std::is_same_v< Unit, std::chrono::seconds > ){
                return std::to_string(sec()) + "sec";
            }else{
                return std::to_string(min()) + "min";
            }
        }

        std::string micro_str() const{
            return str<std::chrono::microseconds>();
        }

        std::string milli_str() const{
            return str<std::chrono::milliseconds>();
        }

        std::string sec_str() const{
            return str<std::chrono::seconds>();
        }

        std::string min_str() const{
            return str<std::chrono::minutes>();
        }

        constexpr double stop() noexcept{
            return this->operator()();
        }

        constexpr void start() noexcept{
            m_start = clock_type::now();
            m_stopped = false;
        }

        constexpr operator double(){
            return this->operator()();
        }

        friend std::ostream& operator<<(std::ostream& os, timer& t){
            return os << t();
        }

    private:
        base_type m_start{clock_type::now()};
        base_type m_end;
        bool m_stopped{false};
    };
    

} // namespace amt


#endif // AMT_BENCHMARK_TIMER_HPP
