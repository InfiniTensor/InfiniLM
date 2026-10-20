import pathlib,ast
r=pathlib.Path(__file__).resolve().parent
for old,new in [('kimi_root_protocol_r3810.py','kimi_root_protocol_r3837.py'),('modern_kimi_root_r3810.py','modern_kimi_root_r3837.py'),('kimi_root_r3810_inside.sh','kimi_root_r3837_inside.sh')]:
 s=(r/old).read_text(encoding='utf8').replace('r3810','r3837')
 if new=='kimi_root_protocol_r3837.py':
  needle="assert after['generated_ids']==expected";assert s.count(needle)==1;s=s.replace(needle,"assert after['generated_ids'][0]==expected")
 ast.parse(s)if new.endswith('.py')else None
 (r/new).write_text(s,encoding='utf8',newline='\n')
print('Corrected r3837 harness prepared only; original failed r3810 source preserved. Merge next operator fix before reloading model.')
