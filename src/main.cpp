#include <iostream>
#include <cache_manger.hpp>

int main(){
    auto temp = amt::cache_manager{};
    for(auto const& el : temp){
        std::cout<<"{\n";
        std::cout<<"\tCache Type: L"<<static_cast<int>(el.type)<<'\n';
        std::cout<<"\tAssociativity: "<<static_cast<int>(el.associativity)<<'\n';
        std::cout<<"\tLineSize: "<<el.line_size<<'\n';
        std::cout<<"\tCache Size: "<<el.size<<'\n';
        std::cout<<"}\n";
    }
    return 0;
}