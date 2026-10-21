import pathlib,ast
r=pathlib.Path(__file__).resolve().parent;s=(r/'audit_glm_profiles_r3797.py').read_text()
s=s.replace("ROWS=int(os.environ['GLM_LARGE_AUDIT_ROWS']);assert ROWS in[96,128]","ROWS=64").replace("f'glm-large-r3797-rows{ROWS}-audit.json'","'glm-PP3-r3873-audit.json'")
s=s.replace("next(j for j in load(r/'glm-capacity-r3721-launch.json')['jobs']if j['label']==f'glm-large-r3721-rows{ROWS}')","load(r/'glm-PP3-serving-r3871-launch.json')")
s=s.replace('r3721','r3871').replace('glm-scale-r3694-build.json','glm-PP3-r3854-build.json').replace('glm-scale-5f91ad7-r3694','glm-PP3-5f91ad7-r3854')
s=s.replace('range(4)','range(3)').replace('stage==3','stage==2').replace('bounds=[0,18,38,58,78]','bounds=[0,26,54,78]').replace('frozen_profile={4:True,8:True,16:True}','frozen_profile={8:True,16:True}')
s=s.replace('for m in[4,8,16]for b in[64,96,128]','for m in[8,16]for b in[32,64]').replace('18+2*len','12+2*len').replace("c['all_serving_cards']==32","c['all_serving_cards']==24").replace('normal_workers_closed=32','normal_workers_closed=24').replace('all32','all24').replace('All32','All24').replace('dec/32','dec/24').replace('inc/32','inc/24').replace('TP8PP4','TP8PP3').replace('large G64/96/128 M4/8/16','G32/64 M8/16')
needle="            assert q['fastpath_stage_layers']==layers and q['fastpath_full_indexer_layers']==full"
assert s.count(needle)==1
s=s.replace(needle,needle+"\n            assert[q['actual_partition']['begin'],q['actual_partition']['end']]==bounds[stage:stage+2]and q['actual_partition']['PP']==3")
ast.parse(s);(r/'audit_GLM_PP3_r3873.py').write_text(s,encoding='utf8',newline='\n')
print('All24 token/EOS/CPUargmax/owner/Slurm/IB audit; frozen reference remains historical PP4 stage3.')
