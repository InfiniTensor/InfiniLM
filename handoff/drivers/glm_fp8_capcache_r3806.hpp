#pragma once
#include <cstddef>
#include <cstdint>
extern "C" const char*glm_fp8_capcache_error_r3806();
extern "C" void*glm_fp8_capcache_create_r3806(int,void*,const void*);
extern "C" int glm_fp8_capcache_fits_r3806(size_t,size_t,size_t,size_t,size_t,size_t,size_t);
extern "C" int glm_fp8_capcache_run_r3806(void*,const void*,void*,size_t,size_t,size_t,size_t,size_t,size_t,size_t,void*,const void*,const void*,const void*,const void*);
extern "C" int glm_fp8_capcache_stats_r3806(void*,uint64_t*,size_t);
extern "C" int glm_fp8_capcache_clear_r3806(void*,const void*,void*);
extern "C" int glm_fp8_capcache_destroy_r3806(void*,const void*,void*);
