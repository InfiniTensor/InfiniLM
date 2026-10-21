#pragma once
#include <array>
#include <cstddef>
#include <stdexcept>
namespace infinilm::models::glm_moe_dsa::PP_layout_r3854 {
inline std::array<size_t,5> boundaries(size_t pp){
 switch(pp){
 case 1:return{0,78,78,78,78};
 case 2:return{0,38,78,78,78};
 case 3:return{0,26,54,78,78};
 case 4:return{0,18,38,58,78};
 default:throw std::invalid_argument("3854 unsupported GLM PP size");
 }
}
}
