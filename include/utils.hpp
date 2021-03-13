#if !defined(AMT_BENCHMARK_UTILS_HPP)
#define AMT_BENCHMARK_UTILS_HPP

#include <unordered_map>
#include <ostream>
#include <utility>
#include <tuple>

namespace amt{
    
    void show_intersection_pts(std::ostream& os, 
        std::unordered_map<std::string_view, std::pair<std::size_t,std::size_t> > const& pts
    ){
        os <<"\n---------Intersection Points---------\n";
        for(auto const& [k,v] : pts){
            auto [one,two] = v;
            os << k << ": [ ( 1 => " << one <<" ), ( 2 => "<< two <<" ) ]\n";
        }
        os <<'\n';
    }

} // namespace amt


#endif // AMT_BENCHMARK_UTILS_HPP
