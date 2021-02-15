#if !defined(AMT_BENCHMARK_CACHE_MANAGER_HPP)
#define AMT_BENCHMARK_CACHE_MANAGER_HPP

#include <boost/predef.h>
#include <cstdint>
#include <array>
#include <vector>

#if defined(BOOST_OS_MACOS_AVAILABLE)
    #include <sys/types.h>
    #include <sys/sysctl.h>
#elif defined(BOOST_OS_LINUX_AVAILABLE)
    #include <unistd.h>
#elif defined(BOOST_OS_WINDOWS_AVAILABLE)
    #include <windows.h>
#else
    #error Unrecognized platform
#endif

namespace amt{

    enum class cache_type : unsigned char{
        None    = 0,
        L1      = 1,
        L2      = 2,
        L3      = 3
    };

    struct cache_info{
        using byte_type = int8_t;
        using int_type  = int32_t;
        using size_type = std::size_t;

        cache_type  type{cache_type::None};
        byte_type   associativity{};
        int_type    line_size{};
        size_type   size{};
    };

    namespace detail{

        #if defined(BOOST_OS_MACOS_AVAILABLE)
            
            template<typename IntType>
            constexpr int sysctlbyname_helper(std::string_view name, IntType& out) noexcept{
                union{
                    int32_t i32;
                    int64_t i64;
                } n;
                auto sz = sizeof(n);
                if(sysctlbyname(name.data(), &n, &sz, nullptr, 0) < 0) return -1;
                switch(sz){
                    case sizeof(n.i32):
                        out = static_cast<IntType>(n.i32);
                        break;
                    case sizeof(n.i64):
                        out = static_cast<IntType>(n.i64);
                        break;
                    default:
                        return -1;
                }
                return 0;
            }

            constexpr auto get_cache_info() noexcept{
                std::array<cache_info,3ul> res;
                res[0].type = cache_type::L1;
                res[1].type = cache_type::L2;
                res[2].type = cache_type::L3;

                auto& line_size = res[0].line_size;
                if( sysctlbyname_helper("hw.cachelinesize", line_size) < 0 ){
                    res[1].line_size = res[2].line_size = line_size = 0u;
                }else{
                    res[1].line_size = res[2].line_size = line_size;
                }

                if( sysctlbyname_helper("machdep.cpu.cache.L2_associativity", res[1].associativity) < 0 ){
                    res[1].associativity = 0;
                }
                
                if( sysctlbyname_helper("machdep.cpu.cache.L1_associativity", res[0].associativity) < 0 ){
                    res[0].associativity = (res[1].associativity * 2);
                }

                if( sysctlbyname_helper("machdep.cpu.cache.L3_associativity", res[2].associativity) < 0 ){
                    res[2].associativity = 0;
                }

                if( sysctlbyname_helper("hw.l1dcachesize", res[0].size) < 0 ){
                    res[0].size = 0u;
                }

                if( sysctlbyname_helper("hw.l2cachesize", res[1].size) < 0 ){
                    res[1].size = 0u;
                }

                if( sysctlbyname_helper("hw.l3cachesize", res[2].size) < 0 ){
                    res[2].size = 0u;
                }

                return res;
            }

        #elif defined(BOOST_OS_LINUX_AVAILABLE)

            auto get_cache_info(){
                std::array<cache_info,3ul> res;
                res[0].type = cache_type::L1;
                res[1].type = cache_type::L2;
                res[2].type = cache_type::L3;

                if(auto temp = sysconf(_SC_LEVEL1_DCACHE_SIZE); temp > 0){
                    res[0].size = static_cast<cache_info::size_type>(temp);
                }

                if(auto temp = sysconf(_SC_LEVEL2_CACHE_SIZE); temp > 0){
                    res[1].size = static_cast<cache_info::size_type>(temp);
                }

                if(auto temp = sysconf(_SC_LEVEL3_CACHE_SIZE); temp > 0){
                    res[2].size = static_cast<cache_info::size_type>(temp);
                }

                if(auto temp = sysconf(_SC_LEVEL1_DCACHE_ASSOC); temp > 0){
                    res[0].associativity = static_cast<cache_info::byte_type>(temp);
                }

                if(auto temp = sysconf(_SC_LEVEL2_CACHE_ASSOC); temp > 0){
                    res[1].associativity = static_cast<cache_info::byte_type>(temp);
                }

                if(auto temp = sysconf(_SC_LEVEL3_CACHE_ASSOC); temp > 0){
                    res[1].associativity = static_cast<cache_info::byte_type>(temp);
                }

                if(auto temp = sysconf(_SC_LEVEL1_DCACHE_LINESIZE); temp > 0){
                    res[0].line_size = static_cast<cache_info::int_type>(temp);
                }

                if(auto temp = sysconf(_SC_LEVEL2_CACHE_LINESIZE); temp > 0){
                    res[1].line_size = static_cast<cache_info::int_type>(temp);
                }

                if(auto temp = sysconf(_SC_LEVEL3_CACHE_LINESIZE); temp > 0){
                    res[2].line_size = static_cast<cache_info::int_type>(temp);
                }

                return res;
            }

        #elif defined(BOOST_OS_WINDOWS_AVAILABLE)
            // TODO: Test the code on windows machine
            auto get_cache_info(){
                std::array<cache_info,3ul> res;
                res[0].type = cache_type::L1;
                res[1].type = cache_type::L2;
                res[2].type = cache_type::L3;

                DWORD buff_sz{};
                GetLogicalProcessorInformation(nullptr,&buff_sz);
                std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buff(buff_sz);
                GetLogicalProcessorInformation(buff.data(), &buff_sz);

                for(auto const& p : buff){
                    if(p.Relationship == RelationCache){
                        auto const& c = p.Cache;
                        auto pos = static_cast<std::size_t>(c.Level) - 1u;
                        if( ( pos == 0u && c.Type == CacheData ) || pos > 0){
                            res[pos].associativity = static_cast<cache_info::byte_type>(c.Associativity);
                            res[pos].line_size = static_cast<cache_info::int_type>(c.LineSize);
                            res[pos].size = static_cast<cache_info::size_type>(c.Size);
                        }
                    }
                }

                return res;
            }

        #endif
    } // namespace detail
    
    struct cache_manager{
        using base_type = std::array<cache_info,3ul>;
        using value_type = typename base_type::value_type;
        using size_type = typename base_type::size_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using iterator = typename base_type::iterator;
        using const_iterator = typename base_type::const_iterator;

        constexpr cache_manager() = default;
        constexpr cache_manager(cache_manager const&) = default;
        constexpr cache_manager(cache_manager &&) = default;
        constexpr cache_manager& operator=(cache_manager const&) = default;
        constexpr cache_manager& operator=(cache_manager &&) = default;
        ~cache_manager() = default;

        constexpr reference operator[](size_type k){ return m_data[k]; }
        constexpr const_reference operator[](size_type k) const { return m_data[k]; }

        constexpr size_type size(size_type k) const { return m_data[k].size; }
        constexpr size_type assoc(size_type k) const { return static_cast<size_type>(m_data[k].associativity); }
        constexpr size_type line_size(size_type k) const { return static_cast<size_type>(m_data[k].line_size); }
        constexpr size_type line_size() const { return line_size(0); }
        
        constexpr reference l1() noexcept{ return m_data[0ul]; }
        constexpr const_reference l1() const  noexcept{ return m_data[0ul]; }
        
        constexpr reference l2() noexcept{ return m_data[1ul]; }
        constexpr const_reference l2() const  noexcept{ return m_data[1ul]; }
        
        constexpr reference l3() noexcept{ return m_data[2ul]; }
        constexpr const_reference l3() const  noexcept{ return m_data[2ul]; }

        constexpr iterator begin() noexcept{ return m_data.begin(); }
        constexpr iterator end() noexcept{ return m_data.end(); }

        constexpr const_iterator begin() const noexcept{ return m_data.begin(); }
        constexpr const_iterator end() const noexcept{ return m_data.end(); }

    private:
        static base_type m_data;
    };

    cache_manager::base_type cache_manager::m_data = detail::get_cache_info();
} // namespace amt


#endif // AMT_BENCHMARK_CACHE_MANAGER_HPP
