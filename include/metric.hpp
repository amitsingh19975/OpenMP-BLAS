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

            void update(double f){
                agg += f;
                min = std::min(min,f);
                max = std::max(max,f);
                plot.push_back(f);
            }
        };
    public:
        using base_type = std::unordered_map<std::string,flops_data>;
        using size_type = std::size_t;
        metric(size_type total)
            : m_total(total)
        {}
        
        auto& insert_or_update(std::string const& name, double gflops){
            if(auto it = m_data.find(name); it != m_data.end()){
                auto& data = it->second;
                data.update(gflops);
                return data;
            }else{
                flops_data f;
                f.agg = gflops;
                f.plot.reserve(m_total);
                m_data[name] = std::move(f);
                return m_data[name];
            }
        }
        
        auto& operator[](std::string const& name){
            return m_data[name];
        }
        
        void insert_or_update(std::string const& name, std::vector<double>&& gflops){
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

        void update_ref(double f){
            m_ref.update(f);
        }

        void plot(std::vector<double> const& x_coord, std::string_view xlabel = "Size", std::string_view ylabel = "GFlops") const{
            namespace plt = matplot;

            auto norm = [](std::string name){
                std::transform(name.begin(), name.end(), name.begin(), [](auto c){
                    return c == '_' ? ' ' : c;
                });
                return name;
            };

            plt::xlabel(xlabel);
            plt::ylabel(ylabel);
            for(auto&& [k,v] : m_data){
                auto l = plt::scatter(x_coord, v.plot, 2);
                l->display_name(norm(k));
                l->marker_face(true);
                plt::hold(plt::on);
            }
            plt::hold(plt::off);
            plt::legend();
            plt::show();
        }

        void plot_speedup(std::vector<double> const& x_coord, std::string_view xlabel = "Size", std::string_view ylabel = "SpeedUP(%)") const{
            namespace plt = matplot;

            auto norm = [](std::string name){
                std::transform(name.begin(), name.end(), name.begin(), [](auto c){
                    return c == '_' ? ' ' : c;
                });
                return name;
            };

            plt::xlabel(xlabel);
            plt::ylabel(ylabel);
            plt::ylim({0.,1000.});
            for(auto&& [k,v] : m_data){
                std::vector<double> speed(m_total);
                std::transform(v.plot.begin(), v.plot.end(), m_ref.plot.begin(), speed.begin(), [](auto l, auto r){
                    return (l / r) * 100.;
                });
                auto l = plt::scatter(x_coord, speed, 2);
                l->display_name(norm(k));
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
                auto ref_avg = (m.m_ref.agg / static_cast<double>(m.m_total));
                os << "Name: "<< k << '\n';
                os << '\t' << "Min GFlops: "<<v.min<<'\n';
                os << '\t' << "Max GFlops: "<<v.max<<'\n';
                os << '\t' << "Max SpeedUp: "<<((v.max / m.m_ref.max) * 100.)<<'\n';
                os << '\t' << "Max Peak Utilization in %: "<< (v.max / peak_performance) * 100. <<'\n';
                os << '\t' << "Avg GFlops: "<<avg<<'\n';
                os << '\t' << "Avg SpeedUp: "<<((avg / ref_avg) * 100.)<<'\n';
                os << '\t' << "Avg Peak Utilization in %: "<< (avg / peak_performance) * 100. <<'\n' << '\n';
            }
            return os;
        }

    private:
        base_type m_data{};
        flops_data m_ref{};
        size_type m_total{};
    };

} // namespace amt


#endif // AMT_BENCHMARK_METRIC_HPP
