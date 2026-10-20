import pathlib,ast
r=pathlib.Path(__file__).resolve().parent;source=r/'latest-experimental-source-r3696/candidates/models-5f91ad7-r3675/GLM/helpers/src/glm_fp8_batched_r776.cu';s=source.read_text(encoding='utf8').replace('r776','r3806');s+='''
// Capacity is only storage. Every math/stride/chunk uses the actual M.
extern "C" int glm_fp8_run_m_r3806(void*raw,size_t actual_m,void*out,const void*x,const void*xs,const void*w,const void*ws){try{
 auto*p=static_cast<Plan*>(raw);if(!p||p->owner!=std::this_thread::get_id()||actual_m<32||actual_m>p->m)throw std::invalid_argument("3806 actualM/owner bounds");Plan frame=*p;frame.m=actual_m;frame.chunk=std::min(actual_m,size_t(256));return glm_fp8_run_r3806(&frame,1,out,x,xs,w,ws);
 }catch(const std::exception&e){error=e.what();return 1;}}
''';(r/'glm_fp8_capplan_r3806.cu').write_text(s,encoding='utf8',newline='\n');print('Exactlatest776 GPU math retained, only capacity-backed actualM frame added; cache bounded768MiB16entries.')
