#!/bin/bash
set -euo pipefail
mkdir /refactor/candidates/throw-trace-r3730
g++ -std=c++17 -shared -fPIC -O2 /refactor/incoming/throw_trace_r3730.cpp -ldl -o /refactor/candidates/throw-trace-r3730/libthrow_trace_r3730.so
python3 - <<'PY'
import pathlib,json,hashlib,datetime
r=pathlib.Path('/refactor');p=r/'candidates/throw-trace-r3730/libthrow_trace_r3730.so';d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':True,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'Exceptiononlyfiltered callstack diagnostics; no math or inference result qualification.'};(r/'results/throw-trace-r3730-build.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
PY
