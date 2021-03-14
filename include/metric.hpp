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
#include <fstream>

namespace amt{

    enum class PLOT_TYPE{
        SCATTER = 0,
        LINE,
        SIZE
    };

    constexpr double clamp(double val, double cutoff = 6.) noexcept{
        return std::min(cutoff,val);
    }

    template<typename T>
    class metric{
        // Get FLOP per cycle from https://en.wikipedia.org/wiki/FLOPS
        static constexpr double peak_performance = 2.3 * 8. * (std::is_same_v<T, double> ? 16. : 32.); // peak_performance = freq * cores * FLOP per cycle
        std::string const type_name = std::is_same_v<T, double> ? "[Double-Precision]" : "[Single-Precision]";
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
        using base_type = std::unordered_map<std::string_view,flops_data>;
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
        
        auto& operator[](std::string_view name){
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

        template<bool Smooth = false, PLOT_TYPE PT = PLOT_TYPE::SCATTER, std::size_t Width = 2ul>
        void plot(std::vector<double> const& x_coord, std::string title, std::string_view xlabel = "Size", std::string_view ylabel = "GFlops"){
            namespace plt = matplot;
            static_assert(static_cast<int>(PT) >= 0 && static_cast<int>(PT) < static_cast<int>(PLOT_TYPE::SIZE));

            title += type_name ;
            
            plt::cla();
            plt::xlabel(xlabel);
            plt::ylabel(ylabel);
            plt::title(title);
            plt::hold(plt::on);
            
            for(auto&& [k,v] : m_data){
                auto speed = v.plot;
                if constexpr(Smooth) moving_average(speed);
                plt::line_handle l;
                if constexpr(PT == PLOT_TYPE::SCATTER){
                    l = plt::scatter(x_coord, speed, Width);
                }else if(PT == PLOT_TYPE::LINE){
                    l = plt::plot(x_coord, speed);
                    l->line_width(Width);
                }
                l->display_name(k);
                l->marker_face(true);
            }
            plt::hold(plt::off);
            plt::legend();
            plt::show();
        }

        template<PLOT_TYPE PT = PLOT_TYPE::SCATTER, std::size_t Width = 2ul>
        void plot_per(std::string title, std::string xlabel = "Percentage[%] of ", std::string_view ylabel = "GFlops"){
            static_assert(static_cast<int>(PT) >= 0 && static_cast<int>(PT) < static_cast<int>(PLOT_TYPE::SIZE));

            namespace plt = matplot;
            xlabel += std::to_string(m_total) + " tests";
            title += type_name;
            
            plt::cla();
            plt::xlabel(xlabel);
            plt::ylabel(ylabel);
            plt::title(title);
            plt::hold(plt::on);

            double size = static_cast<double>(m_total) / 100.;
            std::vector<double> x_coord = plt::iota(0,100);
            std::vector<double> y(100);
            

            for(auto&& [k,v] : m_data){

                for(auto i = 1ul; i < 100ul; i++){
                    auto p = std::min(static_cast<std::size_t>(std::ceil(size * static_cast<double>(i - 1))), m_total);
                    auto n = std::min(static_cast<std::size_t>(std::ceil(size * static_cast<double>(i))), m_total);
                    y[i] = v.plot[p];
                    for(auto j = p; j < n; ++j){
                        y[i] += v.plot[j];
                    }
                    y[i] /= static_cast<double>(size);
                }
                std::sort(y.begin(), y.end(), std::greater<>{});
                
                plt::line_handle l;
                if constexpr(PT == PLOT_TYPE::SCATTER){
                    l = plt::scatter(x_coord, y, 2);
                }else if(PT == PLOT_TYPE::LINE){
                    l = plt::plot(x_coord, y);
                    l->line_width(2);
                }

                l->display_name(k);
                l->marker_face(true);
            }
            plt::hold(plt::off);
            plt::legend();
            plt::show();
        }

        template<bool Smooth = true, PLOT_TYPE PT = PLOT_TYPE::SCATTER, std::size_t Width = 2ul>
        void plot_speedup(std::string_view pattern, std::vector<double> const& x_coord, std::string title, std::string_view xlabel = "Size", std::string_view ylabel = "SpeedUP") const{
            static_assert(static_cast<int>(PT) >= 0 && static_cast<int>(PT) < static_cast<int>(PLOT_TYPE::SIZE));

            namespace plt = matplot;
            flops_data const* pref = nullptr;

            title += type_name;
            plt::title(title);

            std::string name;
            for(auto&& [k,v] : m_data){
                if(auto it = k.find(pattern); it != std::string::npos){
                    pref = std::addressof(v);
                    name = std::string(k);
                    break;
                }
            }

            if(pref == nullptr){
                throw std::runtime_error(
                    "amt::metric::plot_speedup(std::string_view, std::vector<double> const&, std::string_view, std::string_view): "
                    "unable to find pattern"
                );
            }

            plt::cla();
            plt::xlabel(xlabel);
            plt::ylabel(ylabel);
            // plt::ylim({0.,5.});
            plt::hold(plt::on);

            for(auto&& [k,v] : m_data){
                if(std::addressof(v) == pref) continue;
                std::vector<double> speed(m_total);
                std::transform(v.plot.begin(), v.plot.end(), pref->plot.begin(), speed.begin(), [](double l, double r){
                    return clamp(r / l);
                });
                if constexpr(Smooth) moving_average(speed);
                plt::line_handle l;
                if constexpr(PT == PLOT_TYPE::SCATTER){
                    l = plt::scatter(x_coord, speed, Width);
                }else if(PT == PLOT_TYPE::LINE){
                    l = plt::plot(x_coord, speed);
                    l->line_width(Width);
                }
                l->display_name(k);
                l->marker_face(true);
            }
            
            plt::grid(plt::on);
            plt::hold(plt::off);
            plt::legend();
            plt::show();
        }

        template<bool Smooth = false>
        void plot_speedup_semilogy(std::string_view pattern, std::vector<double> const& x_coord, std::string title, std::string_view xlabel = "Size", std::string_view ylabel = "SpeedUP") const{

            namespace plt = matplot;
            flops_data const* pref = nullptr;

            title += type_name;
            plt::title(title);

            std::string name;
            for(auto&& [k,v] : m_data){
                if(auto it = k.find(pattern); it != std::string::npos){
                    pref = std::addressof(v);
                    name = std::string(k);
                    break;
                }
            }

            if(pref == nullptr){
                throw std::runtime_error(
                    "amt::metric::plot_speedup(std::string_view, std::vector<double> const&, std::string_view, std::string_view): "
                    "unable to find pattern"
                );
            }


            plt::cla();
            plt::xlabel(xlabel);
            plt::ylabel(ylabel);
            plt::hold(plt::on);

            for(auto&& [k,v] : m_data){
                if(std::addressof(v) == pref) continue;
                std::vector<double> speed(m_total);
                std::transform(v.plot.begin(), v.plot.end(), pref->plot.begin(), speed.begin(), [](double l, double r){
                    return clamp(r / l);
                });
                if constexpr(Smooth) moving_average(speed);
                auto l = plt::semilogy(x_coord, speed);
                l->display_name(k);
                l->marker_face(true);
            }
            plt::grid(plt::on);
            plt::hold(plt::off);
            plt::legend();
            plt::show();
        }

        template<bool InterPoint = false, PLOT_TYPE PT = PLOT_TYPE::SCATTER, std::size_t Width = 2ul>
        auto plot_speedup_per(std::string_view pattern, std::string title, std::string xlabel = "Percentage[%] of ", std::string_view ylabel = "SpeedUP") const
        {
            static_assert(static_cast<int>(PT) >= 0 && static_cast<int>(PT) < static_cast<int>(PLOT_TYPE::SIZE));

            namespace plt = matplot;
            flops_data const* pref = nullptr;

            plt::title(title + type_name);

            std::string name;
            for(auto&& [k,v] : m_data){
                if(auto it = k.find(pattern); it != std::string::npos){
                    pref = std::addressof(v);
                    name = std::string(k);
                    break;
                }
            }
            if(pref == nullptr){
                throw std::runtime_error(
                    "amt::metric::plot_speedup(std::string_view, std::vector<double> const&, std::string_view, std::string_view): "
                    "unable to find pattern"
                );
            }

            xlabel += std::to_string(m_total) + " tests";

            plt::cla();
            plt::xlabel(xlabel);
            plt::ylabel(ylabel);
            // plt::ylim({0.,10.});
            double size = static_cast<double>(m_total) / 100.;
            std::vector<double> x_coord = plt::iota(0,100);
            std::vector<double> y(100);
            plt::hold(plt::on);

            std::unordered_map<std::string_view, std::pair<std::size_t,std::size_t>> inter_res;

            for(auto&& [k,v] : m_data){
                if(std::addressof(v) == pref) continue;
                std::vector<double> speed(m_total);
                std::transform(v.plot.begin(), v.plot.end(), pref->plot.begin(), speed.begin(), [](double l, double r){
                    return clamp(r / l);
                });

                for(auto i = 1ul; i < 100ul; i++){
                    auto p = std::min(static_cast<std::size_t>(std::ceil(size * static_cast<double>(i - 1))), m_total);
                    auto n = std::min(static_cast<std::size_t>(std::ceil(size * static_cast<double>(i))), m_total);
                    y[i] = speed[p];
                    for(auto j = p; j < n; ++j){
                        y[i] += speed[j];
                    }
                    y[i] /= static_cast<double>(size);
                }
                
                std::sort(y.begin(), y.end(), std::greater<>{});

                if constexpr(InterPoint){
                    std::size_t val_1{};
                    std::size_t val_2{};

                    for(auto i = 0ul; i < y.size(); ++i){
                        auto el = y[i];
                        if(el >= 1.) val_1 = i;
                        if(el >= 2.) val_2 = i;
                    }
                    
                    inter_res[k] = {val_1, val_2};
                }

                plt::line_handle l;
                if constexpr(PT == PLOT_TYPE::SCATTER){
                    l = plt::scatter(x_coord, y, Width);
                }else if(PT == PLOT_TYPE::LINE){
                    l = plt::plot(x_coord, y);
                    l->line_width(Width);
                }
                l->display_name(k);
                l->marker_face(true);
            }

            plt::grid(plt::on);
            plt::hold(plt::off);
            plt::legend();
            plt::show();
            
            return inter_res;
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
                for(auto&& [k,v] : m_data){
                    if(auto it = k.find(name); it != std::string::npos){
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
                    ss << '\t' << "Max SpeedUp with respect to "<<pattern.value()<<": "<<(pref->max / v.max)<<'\n';
                    ss << '\t' << "Avg SpeedUp with respect to "<<pattern.value()<<": "<<(patt_avg / avg)<<'\n';
                }
                ss << '\t' << "Max Peak Utilization in %: "<< (v.max / peak_performance) * 100. <<'\n';
                ss << '\t' << "Avg GFlops: "<<avg<<'\n';
                ss << '\t' << "Avg Peak Utilization in %: "<< (avg / peak_performance) * 100. <<'\n' << '\n';
            }
            return ss.str();
        }

        void raw(std::string_view filename = "raw_data.txt") const{
            std::ofstream f(filename.data());
            for(auto&& [k,v] : m_data){
                f << k << " ";
                for(auto const& d : v.plot){
                    f << d << ' ';
                }
                f << '\n';
            }
            f.close();
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
