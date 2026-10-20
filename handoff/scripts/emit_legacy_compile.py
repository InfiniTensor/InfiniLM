"""Extract original legacy model compiler commands, without historical build receipts.

The generated commands still require the original SDK, Common and helper layout.
Numerical benchmark gates are never changed by this utility.
"""
import argparse, pathlib, re

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('model', choices=['Kimi', 'GLM', 'Deep'])
p.add_argument('--output', type=pathlib.Path, required=True)
a = p.parse_args()
root = pathlib.Path(__file__).resolve().parents[1]
name = {'Kimi': 'refactor_kimi_token_only_build_r963_inside.sh',
        'GLM': 'refactor_glm_cache_view_build_r927_inside.sh',
        'Deep': 'refactor_deep_allselected_build_r815_inside.sh'}[a.model]
s = (root / 'legacy/build-reference' / name).read_text()
commands = []
lines = s.splitlines()
i = 0
while i < len(lines):
    line = lines[i]
    if line.startswith('g++ ') and '-shared ' in line:
        current = [line]
        while current[-1].rstrip().endswith('\\'):
            i += 1
            current.append(lines[i])
        commands.append('\n'.join(current))
    i += 1
if not commands:
    raise SystemExit('No shared-library compiler command found')
if a.output.exists():
    raise SystemExit('Refuse to overwrite an existing command file')
a.output.write_text('#!/bin/bash\nset -euo pipefail\n' + '\n'.join(commands) + '\n')
print('Wrote compiler commands. Set LM/SDK/BASE/HELPER/SOURCE/OLD/ABI as in original recipe; build all GPU helpers first.')
