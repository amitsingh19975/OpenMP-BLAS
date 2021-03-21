#if !defined(AMT_BENCHMARK_TIMER_HPP)
#define AMT_BENCHMARK_TIMER_HPP

#include <chrono>
#include <ostream>

namespace amt{
    
    struct timer{
        
        using clock_type = std::chrono::steady_clock;
        using base_type = decltype(clock_type::now());

        constexpr double operator()() noexcept{
            if(!m_stopped){
                m_end = clock_type::now();
                m_stopped = true;
            }
            return nano();
        }

        template<typename Unit = std::chrono::nanoseconds>
        constexpr double cast() const noexcept{
            auto diff = (m_end - m_start);
            return static_cast<double>(std::chrono::duration_cast<Unit>(diff).count());
        }

        constexpr double nano() const noexcept{
            return cast<std::chrono::nanoseconds>();
        }

        constexpr double milli() const noexcept{
            return cast<std::chrono::milliseconds>();
        }

        constexpr double micro() const noexcept{
            return cast<std::chrono::microseconds>();
        }

        constexpr double sec() const noexcept{
            return cast<std::chrono::seconds>();
        }

        constexpr double min() const noexcept{
            return cast<std::chrono::minutes>();
        }

        template<typename Unit = std::chrono::nanoseconds>
        std::string str() const{
            if constexpr( std::is_same_v< Unit, std::chrono::nanoseconds > ){
                return std::to_string(static_cast<std::size_t>(nano())) + "ns";
            }else if constexpr( std::is_same_v< Unit, std::chrono::microseconds > ){
                return std::to_string(static_cast<std::size_t>(micro())) + "us";
            }else if constexpr( std::is_same_v< Unit, std::chrono::milliseconds > ){
                return std::to_string(static_cast<std::size_t>(milli())) + "ms";
            }else if constexpr( std::is_same_v< Unit, std::chrono::seconds > ){
                return std::to_string(static_cast<std::size_t>(sec())) + "sec";
            }else{
                return std::to_string(static_cast<std::size_t>(min())) + "min";
            }
        }

        std::string nano_str() const{
            return str<std::chrono::nanoseconds>();
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

        friend std::ostream& operator<<(std::ostream& os, timer const& t){
            std::size_t time = static_cast<std::size_t>(t.milli());
            std::size_t milli = time % 1000;
            std::size_t sec = time / 1000;
            std::size_t min = sec / 60;
            sec %= 60;
            
            return os << min << ":" << sec << ":" <<milli ;
        }

    private:
        base_type m_start{clock_type::now()};
        base_type m_end;
        bool m_stopped{false};
    };
    

} // namespace amt


#endif // AMT_BENCHMARK_TIMER_HPP
