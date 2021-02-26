#if !defined(AMT_BENCHMARK_METRIC_HPP)
#define AMT_BENCHMARK_METRIC_HPP

#include <ostream>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <matplot/matplot.h>
#include <algorithm>
#include <functional>
#include <sstream>

namespace amt{

    template<typename T>
    class metric{
        // Get FLOP per cycle from https://en.wikipedia.org/wiki/FLOPS
        static constexpr double peak_performance = 2.3 * 8. * (std::is_same_v<T, double> ? 16. : 32.); // peak_performance = freq * cores * FLOP per cycle
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

        void moving_average(std::vector<double>& x, std::size_t n = 5ul) const noexcept{
            if(x.size() < n) return;
            for(auto i = n; i < x.size(); ++i){
                for(auto j = 1ul; j < n; ++j){
                    x[i] += x[i - j];
                }
                x[i] /= static_cast<double>(n);
            }
                
        }

        void plot(std::vector<double> const& x_coord, std::string_view xlabel = "Size", std::string_view ylabel = "GFlops"){
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
                moving_average(v.plot);
                auto l = plt::plot(x_coord, v.plot);
                l->line_width(2);
                l->display_name(norm(k));
                l->marker_face(true);
                plt::hold(plt::on);
            }
            plt::hold(plt::off);
            plt::legend();
            plt::show();
        }

        void plot_speedup(std::string pattern, std::vector<double> const& x_coord, std::string_view xlabel = "Size", std::string ylabel = "SpeedUP") const{
            namespace plt = matplot;
            flops_data const* pref = nullptr;

            auto sz = pattern.size();
            for(auto&& [k,v] : m_data){
                if(k.size() < sz) continue;
                else if(k.size() == sz && (k == pattern)){
                    pref = std::addressof(v);
                    break;
                }else if(k.substr(0,sz) == pattern) {
                    pref = std::addressof(v);
                    break;
                }
            }
            if(pref == nullptr){
                throw std::runtime_error(
                    "amt::metric::plot_speedup(std::string_view, std::vector<double> const&, std::string_view, std::string_view): "
                    "unable to find pattern"
                );
            }

            ylabel += "("+ pattern + " / existing implementation)";

            auto norm = [](std::string name){
                std::transform(name.begin(), name.end(), name.begin(), [](auto c){
                    return c == '_' ? ' ' : c;
                });
                return name;
            };

            plt::xlabel(xlabel);
            plt::ylabel(ylabel);
            plt::ylim({0.,3.});
            for(auto&& [k,v] : m_data){
                if(std::addressof(v) == pref) continue;
                std::vector<double> speed(m_total);
                std::transform(v.plot.begin(), v.plot.end(), pref->plot.begin(), speed.begin(), [](double l, double r){
                    return (r / l);
                });
                moving_average(speed);
                auto l = plt::scatter(x_coord, speed, 2);
                l->display_name(norm(k));
                l->marker_face(true);
                plt::hold(plt::on);
            }

            plt::hold(plt::off);
            plt::legend();
            plt::show();
        }

        std::string head(std::size_t n = 5) const {
            std::stringstream ss;
            for(auto&& [k,v] : m_data){
                ss << k << ": [ ";
                for(auto i = 0ul; i < n; ++i) ss << v.plot.at(i)<<", ";
                ss << "]\n";
            }
            return ss.str();
        }

        std::string tail(std::size_t n = 5) const {
            std::stringstream ss;
            for(auto&& [k,v] : m_data){
                auto last = v.plot.size() - 1 - n;
                ss << k << ": [ ";
                for(auto i = 0ul; i < n; ++i) ss << v.plot.at(last + i)<<", ";
                ss << "]\n";
            }
            return ss.str();
        }

        std::string str(std::optional<std::string_view> pattern = std::nullopt) const{
            std::stringstream ss;
            flops_data const* pref = nullptr;

            if (pattern.has_value()){
                auto& name = pattern.value();
                auto sz = name.size();
                for(auto&& [k,v] : m_data){
                    if(k.size() < sz) continue;
                    else if(k.size() == sz && (k == name)){
                        pref = std::addressof(v);
                        break;
                    }else if(k.substr(0,sz) == name) {
                        pref = std::addressof(v);
                        break;
                    }
                }
            }

            ss << "Peak Performance: "<< peak_performance << " GFlops\n";
            for(auto&& [k,v] : m_data){
                auto avg = (v.agg / static_cast<double>(m_total));
                ss << "Name: "<< k << '\n';
                ss << '\t' << "Min GFlops: "<<v.min<<'\n';
                ss << '\t' << "Max GFlops: "<<v.max<<'\n';
                if(pref){
                    auto patt_avg = (pref->agg / static_cast<double>(m_total));
                    ss << '\t' << "Max SpeedUp with respect to "<<pattern.value()<<": "<<((pref->max / v.max))<<'\n';
                    ss << '\t' << "Avg SpeedUp with respect to "<<pattern.value()<<": "<<((patt_avg / avg))<<'\n';
                }
                ss << '\t' << "Max Peak Utilization in %: "<< (v.max / peak_performance) * 100. <<'\n';
                ss << '\t' << "Avg GFlops: "<<avg<<'\n';
                ss << '\t' << "Avg Peak Utilization in %: "<< (avg / peak_performance) * 100. <<'\n' << '\n';
            }
            return ss.str();
        }

        friend std::ostream& operator<<(std::ostream& os, metric const& m){
            return os << m.str();
        }

    private:
        base_type m_data{};
        flops_data m_ref{};
        size_type m_total{};
    };

} // namespace amt


#endif // AMT_BENCHMARK_METRIC_HPP
