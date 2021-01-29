#if !defined(AMT_BENCHMARK_METRIC_HPP)
#define AMT_BENCHMARK_METRIC_HPP

#include <ostream>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <matplot/matplot.h>
#include <algorithm>
#include <functional>

namespace amt{

    class metric{
        // Get FLOP per cycle from https://en.wikipedia.org/wiki/FLOPS
        static constexpr double peak_performance = 2.3 * 8 * 32; // peak_performance = freq * cores * FLOP per cycle
        struct flops_data{
            std::vector<double> plot{};
            double min{ peak_performance };
            double max{ 0. };
            double agg{};
        };
    public:
        using base_type = std::unordered_map<std::string_view,flops_data>;
        using size_type = std::size_t;
        metric(size_type total)
            : m_total(total)
        {}
        
        void insert_or_update(std::string_view name, double gflops){
            if(auto it = m_data.find(name); it != m_data.end()){
                auto& data = it->second;
                data.agg += gflops;
                data.min += std::min(data.min,gflops);
                data.max += std::max(data.max,gflops);
                data.plot.push_back(gflops);
            }else{
                flops_data f;
                f.agg = gflops;
                f.plot.reserve(m_total);
                m_data[name] = std::move(f);
            }
        }
        
        void insert_or_update(std::string_view name, std::vector<double>&& gflops){
            if(auto it = m_data.find(name); it != m_data.end()){
                auto& data = it->second;
                data.agg = std::accumulate(gflops.begin(), gflops.end(), 0., std::plus<>{});
                auto [mi,ma] = std::minmax_element(gflops.begin(), gflops.end());
                data.min = *mi;
                data.max = *ma;
                data.plot = std::move(gflops);
            }else{
                flops_data f;
                f.agg = std::accumulate(gflops.begin(), gflops.end(), 0., std::plus<>{});
                auto [mi,ma] = std::minmax_element(gflops.begin(), gflops.end());
                f.min = *mi;
                f.max = *ma;
                f.plot = std::move(gflops);
                m_data[name] = std::move(f);
            }
        }

        flops_data& operator[](std::string_view name){
            if(auto it = m_data.find(name); it != m_data.end()){
                return it->second;
            }else{
                throw std::runtime_error("amt::metric::operator[](std::string_view): Key not found");
            }
        }

        flops_data const& operator[](std::string_view name) const{
            if(auto it = m_data.find(name); it != m_data.end()){
                return it->second;
            }else{
                throw std::runtime_error("amt::metric::operator[](std::string_view): Key not found");
            }
        }

        void plot(std::vector<double> const& x_coord, std::string_view xlabel = "Size", std::string_view ylabel = "GFlops") const{
            namespace plt = matplot;

            plt::xlabel(xlabel);
            plt::ylabel(ylabel);
            for(auto&& [k,v] : m_data){
                auto l = plt::scatter(x_coord, v.plot, 2);
                l->display_name(k);
                l->marker_face(true);
                plt::hold(plt::on);
            }

            plt::hold(plt::off);
            plt::legend();
            plt::show();
        }

        friend std::ostream& operator<<(std::ostream& os, metric const& m){
            os << "Peak Performance: "<< peak_performance << " GFlops\n";
            for(auto&& [k,v] : m.m_data){
                auto avg = (v.agg / static_cast<double>(m.m_total));
                os << "Name: "<< k << '\n';
                os << '\t' << "Min GFlops: "<<v.min<<'\n';
                os << '\t' << "Max GFlops: "<<v.max<<'\n';
                os << '\t' << "Avg GFlops: "<<avg<<'\n' << '\n';
                os << '\t' << "Peak Utilization in %: "<< (avg / peak_performance) * 100. <<'\n' << '\n';
            }
            return os;
        }

    private:
        base_type m_data{};
        size_type m_total{};
    };

} // namespace amt


#endif // AMT_BENCHMARK_METRIC_HPP
