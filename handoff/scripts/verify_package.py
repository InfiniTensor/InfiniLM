import hashlib, json, pathlib, sys

root = pathlib.Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'snapshot-manifest.json').read_text())
bad = []
for name, row in manifest['files'].items():
    path = root / name
    if not path.is_file():
        bad.append([name, 'missing'])
        continue
    raw = path.read_bytes()
    if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
        bad.append([name, 'checksum'])
print(json.dumps({'passed': not bad, 'files': len(manifest['files']), 'failures': bad}, ensure_ascii=False))
sys.exit(bool(bad))
