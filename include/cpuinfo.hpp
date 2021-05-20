#if !defined(AMT_BENCHMARK_CPUINFO_HPP)
#define AMT_BENCHMARK_CPUINFO_HPP

#include <cstddef>
#include <utils.hpp>

namespace amt{

// TODO: Add more CPU support
enum class CPUFamily{
    NONE,
    INTEL_ICELAKE,
    INTEL_SKYLAKE,
    INTEL_BROADWELL,
    INTEL_HASWELL,
    INTEL_IVY_BRIDGE,
    INTEL_KNIGHTS_LANDING
};

template<typename ValueType, std::size_t VecLen, CPUFamily CPUType>
struct cpu_info;

template<std::size_t VecLen>
struct cpu_info<float, VecLen, CPUFamily::INTEL_ICELAKE>{
    constexpr static double fma_latency = 4;
    constexpr static double fma_throughput = ( VecLen < 512ul ? 0.5 : 1. );
    constexpr static double mul_latency = ( VecLen == 256ul ? 4. : 0. );
    constexpr static double mul_throughput = ( VecLen >= 256 ? ( static_cast<double>(VecLen)/512. ) : 0. );
    constexpr static double add_latency = ( VecLen == 256ul ? 4. : 0. );
    constexpr static double add_throughput = ( VecLen >= 256 ? ( static_cast<double>(VecLen)/512. ) : 0. );
    constexpr static double load_latency = ( VecLen == 256ul ? 7. : 0. );
    constexpr static double load_throughput = ( VecLen >= 256 ? ( static_cast<double>(VecLen)/512. ) : 0. );
};

template<std::size_t VecLen>
struct cpu_info<float, VecLen, CPUFamily::INTEL_SKYLAKE>{
    constexpr static double fma_latency = 4;
    constexpr static double fma_throughput = 0.5;
    constexpr static double mul_latency = 4;
    constexpr static double mul_throughput = 0.5;
    constexpr static double add_latency = 4;
    constexpr static double add_throughput = 0.5;
    constexpr static double load_latency = ( VecLen == 256ul ? 7. : 6. );
    constexpr static double load_throughput = 0.5;
};

template<std::size_t VecLen>
struct cpu_info<float, VecLen, CPUFamily::INTEL_BROADWELL>{
    constexpr static double fma_latency = 5;
    constexpr static double fma_throughput = 0.5;
    constexpr static double mul_latency = 3;
    constexpr static double mul_throughput = 0.5;
    constexpr static double add_latency = 3;
    constexpr static double add_throughput = 1;
    constexpr static double load_latency = 1;
    constexpr static double load_throughput = 0.5;
};

template<std::size_t VecLen>
struct cpu_info<float, VecLen, CPUFamily::INTEL_HASWELL>{
    constexpr static double fma_latency = 5;
    constexpr static double fma_throughput = 0.5;
    constexpr static double mul_latency = 5;
    constexpr static double mul_throughput = 0.5;
    constexpr static double add_latency = 3;
    constexpr static double add_throughput = 1;
    constexpr static double load_latency = 1;
    constexpr static double load_throughput = 0.5;
};

template<std::size_t VecLen>
struct cpu_info<float, VecLen, CPUFamily::INTEL_IVY_BRIDGE>{
    constexpr static double fma_latency = 0;
    constexpr static double fma_throughput = 0;
    constexpr static double mul_latency = 5;
    constexpr static double mul_throughput = 1;
    constexpr static double add_latency = 3;
    constexpr static double add_throughput = 1;
    constexpr static double load_latency = 1;
    constexpr static double load_throughput = 1;
};

template<std::size_t VecLen>
struct cpu_info<float, VecLen, CPUFamily::INTEL_KNIGHTS_LANDING>{
    constexpr static double fma_latency = 6;
    constexpr static double fma_throughput = 0.5;
    constexpr static double mul_latency = 0;
    constexpr static double mul_throughput = 0;
    constexpr static double add_latency = 0;
    constexpr static double add_throughput = 0;
    constexpr static double load_latency = 0;
    constexpr static double load_throughput = 0;
};

template<std::size_t VecLen>
struct cpu_info<double, VecLen, CPUFamily::INTEL_ICELAKE>{
    constexpr static double fma_latency = 4;
    constexpr static double fma_throughput = ( VecLen < 512ul ? 0.5 : 1. );
    constexpr static double mul_latency = ( VecLen == 256ul ? 4. : 0. );
    constexpr static double mul_throughput = ( VecLen >= 256 ? ( static_cast<double>(VecLen)/512. ) : 0. );
    constexpr static double add_latency = ( VecLen == 256ul ? 4. : 0. );
    constexpr static double add_throughput = ( VecLen >= 256 ? ( static_cast<double>(VecLen)/512. ) : 0. );
    constexpr static double load_latency = ( VecLen == 256ul ? 7. : 0. );
    constexpr static double load_throughput = ( VecLen >= 256 ? ( static_cast<double>(VecLen)/512. ) : 0. );
};

template<std::size_t VecLen>
struct cpu_info<double, VecLen, CPUFamily::INTEL_SKYLAKE>{
    constexpr static double fma_latency = 4;
    constexpr static double fma_throughput = 0.5;
    constexpr static double mul_latency = 4;
    constexpr static double mul_throughput = 0.5;
    constexpr static double add_latency = 4;
    constexpr static double add_throughput = 0.5;
    constexpr static double load_latency = ( VecLen == 256ul ? 7. : 6. );
    constexpr static double load_throughput = 0.5;
};

template<std::size_t VecLen>
struct cpu_info<double, VecLen, CPUFamily::INTEL_BROADWELL>{
    constexpr static double fma_latency = 5;
    constexpr static double fma_throughput = 0.5;
    constexpr static double mul_latency = ( VecLen == 256ul ? 3. : 5. );
    constexpr static double mul_throughput = 0.5;
    constexpr static double add_latency = 3;
    constexpr static double add_throughput = 1;
    constexpr static double load_latency = 1;
    constexpr static double load_throughput = 0.5;
};

template<std::size_t VecLen>
struct cpu_info<double, VecLen, CPUFamily::INTEL_HASWELL>{
    constexpr static double fma_latency = 5;
    constexpr static double fma_throughput = 0.5;
    constexpr static double mul_latency = 5;
    constexpr static double mul_throughput = 0.5;
    constexpr static double add_latency = 3;
    constexpr static double add_throughput = 1;
    constexpr static double load_latency = 1;
    constexpr static double load_throughput = 0.5;
};

template<std::size_t VecLen>
struct cpu_info<double, VecLen, CPUFamily::INTEL_IVY_BRIDGE>{
    constexpr static double fma_latency = 0;
    constexpr static double fma_throughput = 0;
    constexpr static double mul_latency = 5;
    constexpr static double mul_throughput = 1;
    constexpr static double add_latency = 3;
    constexpr static double add_throughput = 1;
    constexpr static double load_latency = 1;
    constexpr static double load_throughput = 1;
};

template<std::size_t VecLen>
struct cpu_info<double, VecLen, CPUFamily::INTEL_KNIGHTS_LANDING>{
    constexpr static double fma_latency = 6;
    constexpr static double fma_throughput = 0.5;
    constexpr static double mul_latency = 0;
    constexpr static double mul_throughput = 0;
    constexpr static double add_latency = 0;
    constexpr static double add_throughput = 0;
    constexpr static double load_latency = 0;
    constexpr static double load_throughput = 0;
};

template<typename ValueType, std::size_t VecLen, CPUFamily CPUType>
constexpr double fma_latency() noexcept{
    using cpu_type = cpu_info<ValueType,VecLen,CPUType>;
    if constexpr(cpu_type::mul_latency != 0. && cpu_type::add_latency != 0.){
        return cpu_type::fma_latency + cpu_type::load_latency;
    }else{
        return cpu_type::mul_latency + cpu_type::add_latency + cpu_type::load_latency;
    }
}

template<typename ValueType, std::size_t VecLen, CPUFamily CPUType>
constexpr double fma_throughput() noexcept{
    using cpu_type = cpu_info<ValueType,VecLen,CPUType>;
    if constexpr(cpu_type::mul_latency != 0. && cpu_type::add_latency != 0.){
        return cpu_type::fma_throughput + cpu_type::load_throughput;
    }else{
        return cpu_type::mul_throughput + cpu_type::add_throughput + cpu_type::load_throughput;
    }
}

template<typename T, typename U>
constexpr T ceil(U num) noexcept{
    auto int_part = static_cast<std::size_t>(num);
    return static_cast<T>( static_cast<std::size_t>(num >  static_cast<double>(int_part)? num + 1 : num) );
}

template<typename ValueType, std::size_t VecLen, CPUFamily CPUType>
constexpr std::size_t mrxnr() noexcept{
    constexpr auto lat = fma_latency<ValueType,VecLen,CPUType>();
    constexpr auto thr = fma_throughput<ValueType,VecLen,CPUType>();
    constexpr auto elements = VecLen / (sizeof(ValueType) * CHAR_BIT);
    return static_cast<std::size_t>(elements * lat * thr);
}

template<typename ValueType, std::size_t VecLen, CPUFamily CPUType>
constexpr std::size_t calculate_mr() noexcept{
    constexpr auto elements = VecLen / (sizeof(ValueType) * CHAR_BIT);
    return ceil<std::size_t>(ct_sqrt(mrxnr<ValueType,VecLen,CPUType>()) / elements) * elements;
}

template<typename ValueType, std::size_t VecLen, CPUFamily CPUType>
constexpr std::size_t calculate_nr() noexcept{
    constexpr auto lat = fma_latency<ValueType,VecLen,CPUType>();
    constexpr auto thr = fma_throughput<ValueType,VecLen,CPUType>();
    constexpr auto elements = static_cast<double>(VecLen / (sizeof(ValueType) * CHAR_BIT));
    constexpr auto mr = calculate_mr<ValueType,VecLen,CPUType>();
    return ceil<std::size_t>((elements * lat * thr) / mr);
}

} // namespace amt


#endif // AMT_BENCHMARK_CPUINFO_HPP
