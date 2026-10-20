import datetime,hashlib,json,pathlib,shutil,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';base=r/'candidates/rebased-c3deba4-r3485';record=r/'results/rebased-c3deba4-r3485-stage.json';assert not base.exists()and not record.exists();source=json.loads((r/'results/upstream-c3deba4-r3479.json').read_text());assert source['completed']and source['all_recursive_pinned_sources_ready'];repos={v['repo'].split('/')[-1]:v for v in source['repositories']};newLM=pathlib.Path(repos['InfiniLM']['remote_path']);newOps=pathlib.Path(repos['InfiniOps']['remote_path']);oldsource=json.loads((r/'results/upstream-3e0d89fe-r2842.json').read_text());assert oldsource['completed'];oldrepos={v['repo'].split('/')[-1]:v for v in oldsource['repositories']};base.mkdir(parents=True)
d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ready':False,'status':'merging','prior_staging':'rebased-c3deba4-r3479 retained with explicit causal_conv1d conflict; this separate target preserves prior runtime validation contracts across assert-only upstream rewrites','pins':{k:repos[k]['commit']for k in ['InfiniLM','InfiniCore','InfiniOps','InfiniRT','InfiniCCL']},'variants':{},'conflicts':[],'scope':'Source-only reverse-checked three-way migration from3e0/030/c86 onto c3de/14a8/bcfe. ParentOps retains allmodel-specific extensions and qualifiedCast64; Kimi Common3393 F64 and Model3261 privatecommABI/TLS preserved. GLM batchcache/expertorder and Deep3443 overlays remain separate model-assembly work. No binary/performance inheritance; actual new Mul/Pad contracts and all3 models require rebuild and qualification. Removedgitlinks unlinked in newtargetonly. AllnewSDK/Common/model/GPUhelpers require rebuild/qualification; no inheritedperformance. Same pinnedRT/CCL may reuse onlyafterSHAcheck; allOps/Common/modelbinaries must newlybuild/test. No mathguard/capturemetadata relaxation and no inherited performance.'}
def sha(raw):return hashlib.sha256(raw).hexdigest()
def save():
 p=record.with_suffix('.tmp');p.write_text(json.dumps(d,indent=2));p.replace(record)
def read(root,path):
 p=root/path;return p.read_bytes()if p.is_file()else None
def merge(a,b,c,work):
 a=a.replace(b'\r\n',b'\n')if a is not None else None;b=b.replace(b'\r\n',b'\n')if b is not None else None;c=c.replace(b'\r\n',b'\n')if c is not None else None
 if a==b:return c,'upstream'
 if b==c:return a,'private_only'
 if a==c:return a,'already_equal'
 if a is None or b is None or c is None:return None,'addition_deletion_conflict'
 work.mkdir(parents=True,exist_ok=True);ap=work/'private';bp=work/'base';cp=work/'upstream';ap.write_bytes(a);bp.write_bytes(b);cp.write_bytes(c);p=subprocess.run(['git','merge-file','-p',str(ap),str(bp),str(cp)],capture_output=True)
 if p.returncode:(work/'merge.conflict').write_bytes(p.stdout);return None,'three_way_conflict'
 ap.write_bytes(p.stdout);bp.write_bytes(c);cp.write_bytes(b);reverse=subprocess.run(['git','merge-file','-p',str(ap),str(bp),str(cp)],capture_output=True)
 if reverse.returncode or reverse.stdout!=a:(work/'merged_for_review').write_bytes(p.stdout);return None,'reverse_merge_not_exact'
 return p.stdout,'three_way_reverse_exact'
def links(target,fresh,tree):
 third=target/'third_party'
 if third.is_symlink():third.unlink();third.mkdir()
 obsolete={v['path']for v in oldtree['tree']if v['mode']=='160000'}-{v['path']for v in tree['tree']if v['mode']=='160000'}
 for path in obsolete:
  dest=target/path
  if dest.is_symlink():dest.unlink()
  elif dest.is_dir():assert not any(dest.iterdir()),('Review nonempty removed submodule',dest);dest.rmdir()
 for v in tree['tree']:
  if v['mode']!='160000':continue
  dest=target/v['path'];src=(fresh/v['path']).resolve();assert src.is_dir()
  if dest.is_symlink():dest.unlink()
  elif dest.is_dir():
   if any(dest.iterdir()):
    backup=target.parent/('PARENT_SUBMODULE_'+v['path'].replace('/','__'));assert not backup.exists()and dest.resolve().is_relative_to(base.resolve());dest.rename(backup)
   else:dest.rmdir()
  assert not dest.exists();dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(src,target_is_directory=True)
save()
try:
 for kind in ['Ops','Model','Common']:
  fresh=newOps if kind=='Ops'else newLM;old=pathlib.Path(oldrepos['InfiniOps'if kind=='Ops'else'InfiniLM']['remote_path']);oldtree=json.loads((r/'incoming'/oldrepos['InfiniOps'if kind=='Ops'else'InfiniLM']['tree_manifest']).read_text());newtree=json.loads((r/'incoming'/repos['InfiniOps'if kind=='Ops'else'InfiniLM']['tree_manifest']).read_text());aentries={v['path']:v for v in oldtree['tree']if v['type']=='blob'};centries={v['path']:v for v in newtree['tree']if v['type']=='blob'};changed=[p for p in sorted(set(aentries)|set(centries))if aentries.get(p,{}).get('sha')!=centries.get(p,{}).get('sha')]
  for p in changed:
   for root,entry in [(old,aentries.get(p)),(fresh,centries.get(p))]:
    if entry is not None:
     raw=read(root,p);assert raw is not None and hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==entry['sha'],(root,p)
  for name in ['Kimi','GLM','Deep']:
   parent=(r/'candidates/cast64-3e0d89fe-r3315'/name/'InfiniOps'if kind=='Ops'else r/'candidates/kimi-reduce-model-3e0d89fe-r3261/model-src/InfiniLM'if kind=='Model'and name=='Kimi'else r/'candidates/kimi-wide-3e0d89fe-r3393/Common/InfiniLM'if kind=='Common'and name=='Kimi'else r/'candidates/rebased-3e0d89fe-r2842'/kind/name/'InfiniLM');target=base/kind/name/('InfiniOps'if kind=='Ops'else'InfiniLM');assert parent.is_dir();shutil.copytree(parent,target,symlinks=True,ignore=shutil.ignore_patterns('.git','build','.xmake','__pycache__','generated','*.pyc','*.o','*.so','*.a'));links(target,fresh,newtree);changes=[];conflicts=[]
   for path in changed:
    a=read(parent,path);b=read(old,path);c=read(fresh,path);work=base/'merge-work'/kind/name/path.replace('/','__');merged,mode=(a,'retain_private_runtime_guards_over_assert_only_upstream')if kind=='Ops'and path in {'src/base/causal_conv1d.h','src/base/chunk_gated_delta_rule.h','src/base/fused_gated_delta_net_gating.h','src/base/mrope.h','src/base/recurrent_gated_delta_rule.h','src/base/separate_table_rope.h'}else merge(a,b,c,work)
    row={'path':path,'mode':mode,'private_sha256':sha(a)if a is not None else None,'base_sha256':sha(b)if b is not None else None,'upstream_sha256':sha(c)if c is not None else None}
    if'conflict'in mode or mode=='reverse_merge_not_exact':conflicts.append(row);d['conflicts'].append({'kind':kind,'variant':name,**row,'work':str(work)});continue
    dest=target/path
    if merged is None:
     if dest.exists():dest.unlink()
    else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(merged);dest.chmod(int(centries[path]['mode'][-3:],8));row['merged_sha256']=sha(merged)
    changes.append(row)
   m={'kind':kind,'variant':name,'pins':d['pins'],'parent':str(parent),'target':str(target),'ready':not conflicts,'changes':changes,'conflicts':conflicts,'scope':d['scope']};manifest=target.parent/'migration-manifest.json';manifest.write_text(json.dumps(m,indent=2));provenance=target/'SOURCE_VERIFIED.json'
   if provenance.exists():provenance.rename(target/'PARENT_SOURCE_VERIFIED_r3485.json')
   provenance.write_text(json.dumps({'pins':d['pins'],'migration_manifest':str(manifest),'manifest_sha256':sha(manifest.read_bytes()),'upstream_blobs_independently_verified':True,'private_patches_present':True,'qualified_build':False},indent=2));d['variants'][kind+'-'+name]={'ready':not conflicts,'source':str(target),'changed_upstream_files':len(changed),'conflicts':conflicts};save()
 d['ready']=not d['conflicts'];d['status']='all9_variants_sources_rebased'if d['ready']else'explicit_conflict_review_required';save();print(json.dumps({'ready':d['ready'],'status':d['status'],'pins':d['pins'],'variants':d['variants'],'conflicts':d['conflicts']}))

except BaseException as e:
 d['status']='failed';d['error']=repr(e);save();raise
