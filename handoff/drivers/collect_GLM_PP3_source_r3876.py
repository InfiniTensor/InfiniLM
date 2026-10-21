import pathlib,json,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';base=r/'candidates/glm-PP3-5f91ad7-r3854';stage=json.loads((r/'results/glm-PP3-r3854-stage.json').read_text());assert stage['ready'];files={}
for p in base.rglob('*'):
 if p.is_file()and p.suffix in['.cpp','.hpp','.h','.cu','.cuh','.sh']:
  b=p.read_bytes();name=p.relative_to(base).as_posix();assert name not in stage['sources'] or hashlib.sha256(b).hexdigest()==stage['sources'][name],name;files[p.relative_to(r).as_posix()]={'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode('utf8')}
print(json.dumps({'files':files,'scope':'Actual isolated PP3 host topology source; model token/performance qualification separate.'}))
