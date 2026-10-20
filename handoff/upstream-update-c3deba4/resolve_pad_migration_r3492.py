import pathlib,json,datetime,hashlib,difflib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';base=r/'candidates/rebased-c3deba4-r3485';root=r/'results/rebased-c3deba4-r3485-stage.json';d=json.loads(root.read_text());out=r/'results/pad-migration-r3492-resolution.json';assert not out.exists();assert d['status']=='explicit_conflict_review_required'and len(d['conflicts'])==5;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();resolved=[]
for c in d['conflicts']:
 assert c['path']=='csrc/execution/tensor_expressions.hpp';target=base/c['kind']/c['variant']/'InfiniLM';p=target/c['path'];before=p.read_text();assert sha(p)==c['private_sha256'];s=before.replace('tensor_checks::pad::analyze(input, padding, "constant")','tensor_checks::pad::analyze(input, padding)')
 old='''#if defined(INFINILM_PAD_CONSTANT_OPS_UNAVAILABLE)
        throw std::runtime_error("PadConstant public API is unavailable in the installed SDK");
#else
        submit<infini::ops::PadConstant>(
            runtime, infini::ops::PadConstant::DefaultConfig(runtime.device().type()),
            input, std::vector<int64_t>(padding.begin(), padding.end()), value, output);
#endif'''
 new='''        submit<infini::ops::Pad>(
            runtime, infini::ops::Pad::DefaultConfig(runtime.device().type()),
            input, std::vector<int64_t>(padding.begin(), padding.end()),
            std::string{"constant"}, std::optional<double>{value}, output);'''
 assert s.count(old)==1;s=s.replace(old,new);assert 'PadConstant'not in s and 'INFINILM_PAD_CONSTANT_OPS_UNAVAILABLE'not in s;p.write_text(s)
 manifest=target.parent/'migration-manifest.json';m=json.loads(manifest.read_text());assert len(m['conflicts'])==1 and m['conflicts'][0]['path']==c['path'];row={**m['conflicts'][0],'mode':'explicit_new_Pad_API_preserve_shape_checks_remove_obsolete_unavailable_guard','merged_sha256':sha(p)};m['changes'].append(row);m['resolved_conflicts']=m['conflicts'];m['conflicts']=[];m['ready']=True;manifest.write_text(json.dumps(m,indent=2));v=target/'SOURCE_VERIFIED.json';z=json.loads(v.read_text());z.update(manifest_sha256=sha(manifest),qualified_build=False);v.write_text(json.dumps(z,indent=2));key=c['kind']+'-'+c['variant'];d['variants'][key].update(ready=True,conflicts=[]);resolved.append({**c,'new_sha256':sha(p),'diff':''.join(difflib.unified_diff(before.splitlines(True),s.splitlines(True)))})
d['resolved_conflicts_r3492']=d['conflicts'];d['conflicts']=[];d['ready']=all(v['ready']for v in d['variants'].values());assert d['ready']and len(d['variants'])==9;d['status']='all9_variants_sources_rebased';d['resolution_record']=out.name
for v in d['variants'].values():
 target=pathlib.Path(v['source']);m=json.loads((target.parent/'migration-manifest.json').read_text());assert m['ready']and not m['conflicts']
 for c in m['changes']:
  if 'merged_sha256'in c:assert sha(target/c['path'])==c['merged_sha256']
p=root.with_suffix('.tmp');p.write_text(json.dumps(d,indent=2));p.replace(root);out.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':True,'resolved':resolved,'scope':'Source-only c3de Pad interface port. Retains all prior shape checks, switches fixedconstant helper to public Pad string+optional signature, removes obsolete compile-time unavailable guard. No GPU/build/model/performance qualification.'},indent=2));print(json.dumps({'resolved':len(resolved),'all9_ready':d['ready']}))
