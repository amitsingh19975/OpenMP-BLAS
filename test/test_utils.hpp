#if !defined(AMT_BENCHMARK_TEST_UTILS_HPP)
#define AMT_BENCHMARK_TEST_UTILS_HPP

template<typename TestType, typename Container>
void rand_gen(Container& c){
    std::generate(c.begin(), c.end(), [](){
        return static_cast<TestType>(rand() % 100);
    });
}


#endif // AMT_BENCHMARK_TEST_UTILS_HPP
