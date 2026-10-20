"""Run inside one allocated node, then aggregate the JSON files across nodes."""
import argparse, datetime, hashlib, importlib.metadata, json, os, pathlib, platform, subprocess

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--gpus', type=int, default=8)
p.add_argument('--arch', default='gfx936')
a = p.parse_args()
out = pathlib.Path(a.output)
if out.exists():
    raise SystemExit('Refuse to overwrite a previous preflight')
import torch
d = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
     'host': platform.node(), 'python': platform.python_version(),
     'torch': torch.__version__, 'HIP': torch.version.hip,
     'cxx11_abi': bool(torch.compiled_with_cxx11_abi()),
     'CPU_affinity': sorted(os.sched_getaffinity(0)), 'devices': [], 'completed': False}
assert torch.cuda.device_count() == a.gpus
for i in range(a.gpus):
    props = torch.cuda.get_device_properties(i)
    arch = getattr(props, 'gcnArchName', '')
    assert a.arch in props.name + arch, (i, props.name, arch)
    with torch.cuda.device(i):
        x = torch.arange(2048, device='cuda', dtype=torch.float32)
        assert x.sum().item() == 2047 * 2048 // 2
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        del x
    d['devices'].append({'ordinal': i, 'name': props.name, 'arch': arch,
                         'total_memory': total, 'free_memory': free,
                         'CU': props.multi_processor_count, 'small_compute_passed': True})
for name in ['numpy', 'transformers', 'safetensors', 'pybind11']:
    try:
        d[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        d[name] = None
for tool in [['g++', '--version'], ['cmake', '--version'], ['/opt/dtk/cuda/cuda/bin/nvcc', '--version']]:
    q = subprocess.run(tool, capture_output=True, text=True, check=True)
    d[tool[0]] = q.stdout.strip()
d['completed'] = True
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(d, ensure_ascii=False, indent=2))
print(json.dumps({'completed': True, 'devices': len(d['devices']), 'output': str(out)}))
