"""Restore a separately transferred immutable runtime bundle, never weights.

Run on Linux into a materialized, idle runtime. Existing different files are
rejected before writing. This does not qualify the target hardware or model.
"""
import argparse, datetime, hashlib, json, pathlib, tarfile

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def relative(name):
    p = pathlib.PurePosixPath(name)
    assert not p.is_absolute() and '..' not in p.parts and p.parts, name
    return p

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('archive', type=pathlib.Path)
p.add_argument('--profile', choices=['legacy', 'current'], required=True)
p.add_argument('--runtime', type=pathlib.Path, required=True)
p.add_argument('--manifest', type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1] / 'evidence/runtime-bundles.json')
a = p.parse_args()
root = a.runtime.resolve()
assert root.is_dir(), 'Materialize an isolated runtime first'
expected = json.loads(a.manifest.read_text())['profiles'][a.profile]
assert digest(a.archive) == expected['sha256'], 'Archive checksum mismatch'
proof = root / 'runtime-restore-proof.json'
assert not proof.exists(), 'Refuse to restore into an already restored runtime'
with tarfile.open(a.archive) as t:
    m = json.load(t.extractfile('RUNTIME_MANIFEST.json'))
    assert m['profile'] == a.profile and len(m['files']) == expected['files']
    members = {v.name: v for v in t.getmembers()}
    assert len(members) == len(m['files']) + 1
    for name, row in m['files'].items():
        rel = relative(name)
        assert name in members and members[name].isfile()
        assert any(name == v or name.startswith(v + '/') for v in expected['roots'])
        dst = root / rel
        assert dst.resolve().is_relative_to(root), name
        data = t.extractfile(members[name]).read()
        assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'], name
        if dst.exists():
            assert dst.is_file() and digest(dst) == row['sha256'], 'Existing file differs: ' + name
    for name, row in m.get('symlinks', {}).items():
        dst = root / relative(name)
        target = pathlib.Path(row['target'])
        resolved = target if target.is_absolute() else (dst.parent / target).resolve()
        assert resolved.is_relative_to(root) or str(resolved).startswith(('/opt/dtk/', '/opt/hyhal/')), name
        assert dst.parent.resolve().is_relative_to(root), name
        if dst.exists() or dst.is_symlink():
            assert dst.is_symlink() and str(dst.readlink()) == row['target'], 'Existing alias differs: ' + name
    for name, row in m['files'].items():
        dst = root / relative(name)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(t.extractfile(members[name]).read())
            dst.chmod(members[name].mode)
    for name, row in m.get('symlinks', {}).items():
        dst = root / relative(name)
        if not dst.is_symlink():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(row['target'], target_is_directory=row['directory'])
proof.write_text(json.dumps({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            'profile': a.profile, 'archive_sha256': expected['sha256'],
                            'files': expected['files'], 'symlinks': len(m.get('symlinks', {})),
                            'target_GPU_and_model_requalification_required': True}, indent=2))
print(json.dumps({'restored': True, 'proof': str(proof), 'qualified': False}))
