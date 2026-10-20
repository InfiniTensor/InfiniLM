"""Create an isolated source layout matching the archived /refactor build paths.

No binaries or weights are installed, and an existing destination is never reused.
"""
import argparse, hashlib, json, os, pathlib, shutil, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]

def run(*args):
    subprocess.run(args, check=True)

def copy(src, dst):
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc', '*.so', '*.o'))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=['legacy', 'current'], required=True)
    p.add_argument('--destination', type=pathlib.Path, required=True)
    p.add_argument('--cache', type=pathlib.Path, required=True, help='Git clones at exact pins, named by repository basename')
    p.add_argument('--fetch', action='store_true', help='Fetch missing exact commits from public upstream repositories')
    p.add_argument('--with-experiments', action='store_true', help='Stage isolated experimental sources as well; never enables them in a stable model')
    a = p.parse_args()
    dst = a.destination.resolve()
    if dst.exists():
        raise SystemExit('Destination must not exist: ' + str(dst))
    pins = json.loads((ROOT / 'pins.json').read_text())
    a.cache.mkdir(parents=True, exist_ok=True)
    needed = pins[a.profile]['repositories']
    for row in needed:
        name = row['repo'].split('/')[-1]
        target = a.cache / name
        if not target.exists():
            if not a.fetch:
                raise SystemExit('Missing cached repository: ' + str(target))
            run('git', 'clone', '--no-checkout', 'https://github.com/' + row['repo'] + '.git', str(target))
        has = subprocess.run(['git', '-C', str(target), 'cat-file', '-e', row['commit'] + '^{commit}'], capture_output=True).returncode == 0
        if not has and a.fetch:
            run('git', '-C', str(target), 'fetch', 'origin', row['commit'])
        if not has and not a.fetch:
            raise SystemExit('Missing cached commit: ' + row['commit'])
        if subprocess.check_output(['git', '-C', str(target), 'status', '--porcelain'], text=True).strip():
            raise SystemExit('Refuse to change a dirty dependency cache: ' + str(target))
        run('git', '-C', str(target), 'checkout', '--detach', row['commit'])
        assert subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip() == row['commit']
    dst.mkdir()
    for n in ['candidates', 'src', 'incoming', 'results', 'control', 'build', 'install', 'logs', 'build-deps']:
        (dst / n).mkdir()
    for name in ['json', 'spdlog', 'InfiniCore']:
        copy(a.cache / name, dst / '_upstream_deps' / name)
    def third_party(path):
        dep = path / 'third_party'
        dep.mkdir(exist_ok=True)
        for name in ['json', 'spdlog', 'InfiniCore']:
            q = dep / name
            if q.exists() or q.is_symlink():
                if q.is_symlink():
                    q.unlink()
                else:
                    shutil.rmtree(q)
            q.symlink_to(os.path.relpath(dst / '_upstream_deps' / name, q.parent), target_is_directory=True)
    if a.profile == 'current':
        manifest = json.loads((ROOT / 'current/manifest.json').read_text())
        for tag, row in manifest['overlays'].items():
            kind, model = tag.split('/')
            target = dst / 'candidates/rebased-3e0d89fe-r2842' / kind / model / row['project']
            copy(a.cache / row['project'], target)
            for name in row['delete']:
                f = target / name
                if f.exists():
                    f.unlink()
            for name in row['files']:
                f = ROOT / 'current/overlays' / tag / name
                q = target / name
                q.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, q)
            if kind != 'Ops':
                third_party(target)
        models = {'Kimi': 'models-3e0d89fe-r2863/Kimi', 'GLM': 'glm-batch-model-r3072', 'Deep': 'models-3e0d89fe-r2856/Deep'}
        for name, path in models.items():
            copy(ROOT / 'current/models' / name, dst / 'candidates' / path)
        for kind in ['RT', 'CCL']:
            copy(ROOT / 'dependencies/current' / kind, dst / 'src' / kind)
        if a.with_experiments:
            for source, target in [('glm-expert-order','glm-expert-order-r3348'),
                                   ('glm-prefill-cache','glm-prefill-model-r3148'),
                                   ('kimi-reduce-model','kimi-reduce-model-3e0d89fe-r3261')]:
                copy(ROOT / 'current/experiments' / source, dst / 'candidates' / target)
            for source, target in [('kimi-reduce-common','kimi-reduce-3e0d89fe-r3226/Common/InfiniLM'),
                                   ('kimi-wide-common','kimi-wide-3e0d89fe-r3393/Common/InfiniLM')]:
                q = dst / 'candidates' / target
                copy(dst / 'candidates/rebased-3e0d89fe-r2842/Common/Kimi/InfiniLM', q)
                shutil.copytree(ROOT / 'current/experiments' / source, q, dirs_exist_ok=True)
            q = dst / 'candidates/kimi-reduce-3e0d89fe-r3226/Model/InfiniLM'
            copy(dst / 'candidates/rebased-3e0d89fe-r2842/Model/Kimi/InfiniLM', q)
            shutil.copytree(ROOT / 'current/experiments/kimi-reduce-Model-overlay', q, dirs_exist_ok=True)
    else:
        paths = {
            'common/Kimi': 'candidates/kimi-trace-r3/InfiniLM',
            'common/GLMDeep': 'src/InfiniLM',
            'models/Kimi': 'candidates/kimi-token-only-r963',
            'models/Deep': 'candidates/native-engine-memory-r23/InfiniLM',
        }
        for source, path in paths.items():
            copy(ROOT / 'legacy' / source, dst / path)
        for path in paths.values():
            lm = dst / path
            if (lm / 'InfiniLM').is_dir():
                lm = lm / 'InfiniLM'
            if (lm / 'csrc').is_dir():
                third_party(lm)
        for name in ['Kimi', 'GLM', 'Deep']:
            copy(ROOT / 'dependencies/legacy/Ops' / name, dst / 'src' / ('Ops-' + name))
        for kind in ['RT', 'CCL']:
            copy(ROOT / 'dependencies/legacy' / kind, dst / 'src' / kind)
    for source in [ROOT / a.profile / 'incoming', ROOT / 'drivers']:
        if source.is_dir():
            shutil.copytree(source, dst / 'incoming', dirs_exist_ok=True)
    (dst / 'handoff-profile.json').write_text(json.dumps({'profile': a.profile, 'pins': pins[a.profile], 'binary_qualification': False}, indent=2))
    print(json.dumps({'materialized': str(dst), 'profile': a.profile, 'built': False}))

if __name__ == '__main__':
    main()
