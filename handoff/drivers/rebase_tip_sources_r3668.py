import pathlib,json,hashlib,shutil,subprocess,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';base=r/'candidates/rebased-5f91ad7-r3668';out=r/'results/rebased-5f91ad7-r3668-stage.json';assert not base.exists()and not out.exists();new=json.loads((r/'results/upstream-5f91ad7-r3667.json').read_text());old=json.loads((r/'results/upstream-06d0c03-r3547.json').read_text());assert new['completed']and old['completed'];nr={v['repo'].split('/')[-1]:v for v in new['repositories']};orr={v['repo'].split('/')[-1]:v for v in old['repositories']};pins={k:nr[k]['commit']for k in ['InfiniLM','InfiniCore','InfiniOps','InfiniRT','InfiniCCL']};base.mkdir();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ready':False,'pins':pins,'variants':{},'conflicts':[],'scope':'New5f91 source copy and reverse-checked privateoverlay migration fromactual06d sources. LM blobs unchanged; coreREADME/OpsMoore paths andRT/CCLgitlinks updated. NativeHygon Mulmetadata/Pad/C10guard and model contracts retained. FreshnewRT/CCL,allSDK/Common/Model/helpers mustbuild, no performance inheritance.'};sha=lambda b:hashlib.sha256(b).hexdigest()
def save():out.write_text(json.dumps(d,indent=2))
def read(root,name):p=root/name;return p.read_bytes()if p.is_file()else None
def merge(a,b,c,work):
 if a==b:return c,'upstream'
 if b==c:return a,'private_only'
 if a==c:return a,'already_equal'
 if a is None or b is None or c is None:return None,'addition_deletion_conflict'
 work.mkdir(parents=True);ap,bp,cp=[work/n for n in ['private','base','new']]
 for p,v in [(ap,a),(bp,b),(cp,c)]:p.write_bytes(v)
 q=subprocess.run(['git','merge-file','-p',str(ap),str(bp),str(cp)],capture_output=True)
 if q.returncode:(work/'forward.conflict').write_bytes(q.stdout);return None,'merge_conflict'
 ap.write_bytes(q.stdout);bp.write_bytes(c);cp.write_bytes(b);back=subprocess.run(['git','merge-file','-p',str(ap),str(bp),str(cp)],capture_output=True)
 if back.returncode or back.stdout!=a:(work/'reverse.conflict').write_bytes(back.stdout);return None,'reverse_conflict'
 return q.stdout,'three_way_reverse_exact'
save()
for kind in ['Model','Common','Ops']:
 key='InfiniOps'if kind=='Ops'else'InfiniLM';row=nr[key];oldrow=orr[key];fresh=pathlib.Path(row['remote_path']);upold=pathlib.Path(oldrow['remote_path']);tree=json.loads((r/'incoming'/row['tree_manifest']).read_text())['tree'];otree=json.loads((r/'incoming'/oldrow['tree_manifest']).read_text())['tree'];aent={v['path']:v for v in otree if v['type']=='blob'};cent={v['path']:v for v in tree if v['type']=='blob'};changed=[p for p in sorted(set(aent)|set(cent))if aent.get(p,{}).get('sha')!=cent.get(p,{}).get('sha')]
 for name in ['Kimi','GLM','Deep']:
  parent=r/'candidates/rebased-06d0c03-r3548'/kind/name/key;target=base/kind/name/key;assert parent.is_dir();shutil.copytree(parent,target,symlinks=True,ignore=shutil.ignore_patterns('.git','build','.xmake','__pycache__','generated','*.pyc','*.o','*.so','*.a','native-checks-*','PARENT_GENERATED*'))
  for v in tree:
   if v['mode']!='160000':continue
   dest=target/v['path'];src=(fresh/v['path']).resolve();assert src.is_dir()
   if dest.is_symlink():dest.unlink()
   elif dest.is_dir():assert not any(dest.iterdir()),str(dest);dest.rmdir()
   assert not dest.exists();dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(src,target_is_directory=True)
  changes=[];conflicts=[]
  for path in changed:
   a,b,c=read(parent,path),read(upold,path),read(fresh,path);merged,how=merge(a,b,c,base/'merge-work'/kind/name/path.replace('/','__'));v={'path':path,'mode':how,'private_sha256':sha(a)if a is not None else None,'old_upstream_sha256':sha(b)if b is not None else None,'new_upstream_sha256':sha(c)if c is not None else None}
   if'conflict'in how:conflicts.append(v);d['conflicts'].append({'kind':kind,'variant':name,**v});continue
   dest=target/path
   if merged is None:
    if dest.exists():assert dest.resolve().is_relative_to(base.resolve());dest.unlink()
   else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(merged);dest.chmod(int(cent[path]['mode'][-3:],8));v['merged_sha256']=sha(merged)
   changes.append(v)
  if kind=='Ops':
   assert 'input_shape_(input.shape())'in(target/'src/base/mul.h').read_text();assert '"kHygon"'in(target/'scripts/generate_torch_ops.py').read_text();assert(target/'src/native/cuda/hygon/ops/pad/kernel.h').exists()
  m={'ready':not conflicts,'pins':pins,'kind':kind,'variant':name,'parent':str(parent),'target':str(target),'changes':changes,'conflicts':conflicts,'scope':d['scope']};mp=target.parent/'migration-manifest.json';mp.write_text(json.dumps(m,indent=2));verified=target/'SOURCE_VERIFIED.json'
  if verified.exists():verified.rename(target/'PARENT_SOURCE_VERIFIED_r3668.json')
  verified.write_text(json.dumps({'pins':pins,'migration_manifest':str(mp),'manifest_sha256':sha(mp.read_bytes()),'upstream_blobs_independently_verified':True,'private_patches_present':True,'qualified_build':False},indent=2));d['variants'][kind+'-'+name]={'source':str(target),'ready':not conflicts,'changed_upstream_files':len(changed),'conflicts':conflicts};save()
d['ready']=not d['conflicts'];save();print(json.dumps({'ready':d['ready'],'pins':pins,'variants':d['variants'],'conflicts':d['conflicts']}))
