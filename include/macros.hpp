#if !defined(AMT_BENCHMARK_MACROS_HPP)
#define AMT_BENCHMARK_MACROS_HPP

#if (!defined(__GNUC__) && !defined(__clang__))
    #define AMT_HAS_NO_INLINE_ASSEMBLY
#endif

#define AMT_ALWAYS_INLINE __attribute__((always_inline))
#define MiB(SIZE) SIZE / (1<<20)
#define KiB(SIZE) SIZE / (1<<10)

// See http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0154r1.html
// for more information.
#if defined(__GNUC__)
// Cache line alignment
#if defined(__i386__) || defined(__x86_64__)
    #define AMT_CACHELINE_SIZE 64
#elif defined(__powerpc64__)
    #define AMT_CACHELINE_SIZE 128
#elif defined(__aarch64__)
    // We would need to read special register ctr_el0 to find out L1 dcache size.
    // This value is a good estimate based on a real aarch64 machine.
    #define AMT_CACHELINE_SIZE 64
#elif defined(__arm__)
    // Cache line sizes for ARM: These values are not strictly correct since
    // cache line sizes depend on implementations, not architectures.  There
    // are even implementations with cache line sizes configurable at boot
    // time.
    #if defined(__ARM_ARCH_5T__)
        #define AMT_CACHELINE_SIZE 32
    #elif defined(__ARM_ARCH_7A__)
        #define AMT_CACHELINE_SIZE 64
    #endif
#endif

#ifndef AMT_CACHELINE_SIZE
    #define AMT_CACHELINE_SIZE 64
#endif

#endif // AMT_BENCHMARK_MACROS_HPP
