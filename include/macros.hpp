#if !defined(AMT_BENCHMARK_MACROS_HPP)
#define AMT_BENCHMARK_MACROS_HPP

#if (!defined(__GNUC__) && !defined(__clang__))
    #define AMT_HAS_NO_INLINE_ASSEMBLY
#endif

#define AMT_ALWAYS_INLINE __attribute__((always_inline))
#define MiB(SIZE) SIZE / (1<<20)
#define KiB(SIZE) SIZE / (1<<10)

#endif // AMT_BENCHMARK_MACROS_HPP
