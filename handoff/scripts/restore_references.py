"""Safely restore the separate immutable reference archive into a fresh runtime."""
import argparse, hashlib, json, pathlib, tarfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('archive', type=pathlib.Path)
p.add_argument('--runtime', type=pathlib.Path, required=True)
p.add_argument('--manifest', type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1] / 'reference-data-manifest.json')
a = p.parse_args()
expected = json.loads(a.manifest.read_text())
h = hashlib.sha256()
with a.archive.open('rb') as f:
    for block in iter(lambda: f.read(1024 * 1024), b''):
        h.update(block)
assert h.hexdigest() == expected['archive_sha256'], 'Archive checksum mismatch'
root = a.runtime.resolve()
assert root.is_dir(), 'Materialize the runtime first'
with tarfile.open(a.archive) as t:
    for member in t.getmembers():
        name = pathlib.PurePosixPath(member.name)
        assert name.parts and name.parts[0] == 'reference-data'
        assert not name.is_absolute() and '..' not in name.parts
        assert not member.issym() and not member.islnk()
        if not member.isfile() or name.name == 'manifest.json' and len(name.parts) == 2:
            continue
        rel = pathlib.PurePosixPath(*name.parts[1:]).as_posix()
        assert rel in expected['files'], rel
        dst = root / rel
        assert dst.resolve().is_relative_to(root), rel
        raw = t.extractfile(member).read()
        row = expected['files'][rel]
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'], rel
        if dst.exists():
            assert dst.read_bytes() == raw, 'Existing file differs: ' + rel
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(raw)
print(json.dumps({'restored': True, 'files': len(expected['files']), 'runtime': str(root)}))
