import configparser,datetime,hashlib,io,json,pathlib,tarfile,time,urllib.request
r=pathlib.Path(__file__).resolve().parent;out=r/'upstream-c3deba4-r3477';assert not out.exists();out.mkdir()
manifest={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'root':{'repo':'InfiniTensor/InfiniLM','commit':'c3deba450a631d79e3a183048726b4a1f4acabdd'},'repositories':[],'scope':'Pinned current user-specified refactor branch; exact recursive gitlinks and Git blob SHA1 checks. GitSSH ls-remote3477 verifiedc3deba4 on2026-10-20; pinnednewCore14a8. Existing3e0 source/build snapshots and all running model experiments preserved. No new-version performance inheritance.'}
seen={};cached={}
for directory in['upstream-3e0d89fe-r2840','upstream-f3b27761-r2634','upstream-10eb4cae-r2552','upstream-b7b370-r2281']:
 root=r/directory;p=root/'manifest.json'
 if not p.exists():continue
 q=json.loads(p.read_text(encoding='utf-8'));assert q['completed']
 for v in q['repositories']:
  for candidate in[root,r/'upstream-10eb4cae-r2552',r/'upstream-b7b370-r2281']:
   archive=candidate/v['archive'];tree=candidate/v['tree_manifest']
   if archive.exists()and tree.exists()and hashlib.sha256(archive.read_bytes()).hexdigest()==v['archive_SHA256']:
    row=dict(v,archive_root=str(candidate));cached[v['repo']+'@'+v['commit']]=row;break
def save():(out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
def get(url,cap):
 for attempt in range(3):
  try:
   with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'InfiniLM-recursive-pinned-source-audit'}),timeout=25)as response:data=response.read(cap+1)
   assert len(data)<=cap;return data
  except Exception:
   if attempt==2:raise
   time.sleep(1)
def fetch(repo,commit):
 key=repo+'@'+commit
 if key in seen:return
 if key in cached:
  row=dict(cached[key],reuse_previously_verified_archive=True);assert row['all_git_blob_SHA1_exact'];seen[key]=row;manifest['repositories'].append(row);save()
  for child in row['gitlinks']:fetch(child['repo'],child['commit'])
  return
 assert len(commit)==40 and all(c in '0123456789abcdef'for c in commit)
 tree_raw=get(f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1',20*1024**2);tree=json.loads(tree_raw);assert not tree.get('truncated')
 expected={v['path']:v for v in tree['tree']if v['type']=='blob'};gitlinks={v['path']:v['sha']for v in tree['tree']if v['mode']=='160000'}
 data=get(f'https://codeload.github.com/{repo}/tar.gz/{commit}',100*1024**2);tag=repo.replace('/','-')+'-'+commit[:12];archive=out/(tag+'.tar.gz');archive.write_bytes(data);(out/(tag+'-tree.json')).write_bytes(tree_raw);found=set();modules=b''
 with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz')as tar:
  for m in tar:
   p=pathlib.PurePosixPath(m.name);assert not p.is_absolute()and'..'not in p.parts
   if m.isdir():continue
   rel='/'.join(p.parts[1:]);assert rel in expected
   if m.issym():raw=m.linkname.encode()
   else:assert m.isfile();raw=tar.extractfile(m).read()
   assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==expected[rel]['sha'],(repo,rel);found.add(rel)
   if rel=='.gitmodules':modules=raw
 assert found==set(expected)
 cfg=configparser.ConfigParser();cfg.read_string(modules.decode());children=[]
 for section in cfg.sections():
  path=cfg[section]['path'];url=cfg[section]['url'];assert path in gitlinks
  if url.startswith('git@github.com:'):child=url[len('git@github.com:'):]
  elif url.startswith('https://github.com/'):child=url[len('https://github.com/'):]
  elif url.startswith('../'):child=repo.split('/')[0]+'/'+url[3:]
  else:raise ValueError(url)
  children.append({'path':path,'repo':child.removesuffix('.git').rstrip('/'),'commit':gitlinks[path]})
 assert {v['path']for v in children}==set(gitlinks)
 row={'repo':repo,'commit':commit,'archive':archive.name,'archive_root':str(out),'archive_SHA256':hashlib.sha256(data).hexdigest(),'tree_manifest':tag+'-tree.json','all_git_blob_SHA1_exact':True,'blob_count':len(found),'gitlinks':children};seen[key]=row;manifest['repositories'].append(row);save()
 for child in children:fetch(child['repo'],child['commit'])
save()
try:
 fetch(**manifest['root']);manifest['completed']=True;save();print(json.dumps({'completed':True,'repositories':len(seen),'root':manifest['root'],'changed_primary':[{k:v[k]for k in['repo','commit','blob_count']}for v in manifest['repositories']if v['repo'].startswith('InfiniTensor/')]}))
except BaseException as exc:manifest['error']=repr(exc);save();raise
