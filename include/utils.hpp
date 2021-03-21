#if !defined(AMT_BENCHMARK_UTILS_HPP)
#define AMT_BENCHMARK_UTILS_HPP

#include <unordered_map>
#include <ostream>
#include <utility>
#include <tuple>

namespace amt{

    struct speed_t{
        std::size_t one{};
        std::size_t two{};
    };
    
    void show_intersection_pts(std::ostream& os, 
        std::unordered_map<std::string_view, std::pair<speed_t,speed_t> > const& pts
    ){
        os <<"\n---------Intersection Points---------\n";
        for(auto const& [k,v] : pts){
            auto [up,down] = v;
            os << k << ": [ ( 1 => U: " << up.one<<", D: "<<down.one <<" ), ( 2 => U: "<< up.two<<", D: "<<down.two <<" ) ]\n";
        }
        os <<'\n';
    }

} // namespace amt


#endif // AMT_BENCHMARK_UTILS_HPP
