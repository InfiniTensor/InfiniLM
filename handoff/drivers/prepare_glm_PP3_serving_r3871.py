import pathlib,ast,re
r=pathlib.Path(__file__).resolve().parent
s=(r/'modern_glm_large_r3721.py').read_text().replace('glm-scale-5f91ad7-r3694','glm-PP3-5f91ad7-r3854').replace('libmodern_glm_scale_r3694.so','libmodern_glm_PP3_r3854.so').replace('glm-scale-r3694-build.json','glm-PP3-r3854-build.json').replace('r3721','r3871')
s=s.replace('assert(ROWS,CAPACITY)in[(96,640),(128,640)]','assert(ROWS,CAPACITY)==(64,640)').replace('assert PP in[2,4]','assert PP==3').replace('BOUNDS=[0,38,78]if PP==2 else[0,18,38,58,78]','BOUNDS=[0,26,54,78]').replace('>10*1024**3','>8*1024**3')
s=s.replace('actual38/40or18/20/20/20layers, ctx16rows','actual26/28/24layers, ctx64rows').replace('TP8PP2/4','TP8PP3')
# The new library reports actual module bounds; keep this in the model evidence.
needle="    report['memory_after_create']=torch.cuda.mem_get_info(rank);"
assert s.count(needle)==1
s=s.replace(needle,"    lib.modern_glm_partition_report_r3854.argtypes=[ctypes.c_void_p];lib.modern_glm_partition_report_r3854.restype=ctypes.c_char_p;part=json.loads(lib.modern_glm_partition_report_r3854(ctx));assert[part['begin'],part['end']]==BOUNDS[stage:stage+2]and part['PP']==3;report['actual_partition']=part\n"+needle)
ast.parse(s);(r/'modern_glm_large_r3871.py').write_text(s,encoding='utf8',newline='\n')
s=(r/'glm_large_probe_r3721.py').read_text().replace('r3721','r3871').replace('TP8PP2/4','TP8PP3').replace("bounds=[0,38,78]if PP==2 else[0,18,38,58,78]","bounds=[0,26,54,78]").replace('for qa_batch in[4,8,16]','for qa_batch in[8,16]').replace('for m in[4,8,16]for b in[64,96,128]','for m in[8,16]for b in[32,64]').replace('== 18+2*len(configs)','== 12+2*len(configs)').replace('32servicecards;G64/G128/G192 withM4/M8/M16,ctx128or192rows','24servicecards;G32/G64 withM8/M16,ctx64rows')
ast.parse(s);(r/'glm_large_probe_r3871.py').write_text(s,encoding='utf8',newline='\n')
s=(r/'glm_large_r3721_inside.sh').read_text().replace('r3721','r3871').replace('glm-scale-5f91ad7-r3694','glm-PP3-5f91ad7-r3854');(r/'glm_large_r3871_inside.sh').write_text(s,encoding='utf8',newline='\n')
print('PP3 one-load 20phase QA/B32/B64 M8/M16 matrix; exact original full model loader and sixGiB serving guard.')
