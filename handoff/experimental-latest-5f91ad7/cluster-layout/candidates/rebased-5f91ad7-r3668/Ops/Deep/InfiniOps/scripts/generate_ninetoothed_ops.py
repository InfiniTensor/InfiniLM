import argparse
import importlib.util
import pathlib
import shutil
import sys

_PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
_OPS_DIR = _PROJECT_DIR / "src" / "ninetoothed" / "ops"


def _find_op_modules():
    return {
        path.parent.name: path
        for path in sorted(_OPS_DIR.glob("*/build.py"))
        if path.is_file()
    }


def _build_manifest(output_dir):
    return sorted(
        str(path)
        for path in pathlib.Path(output_dir).rglob("*.cpp")
        if not path.name.endswith(".tmp.cpp")
    )


def _write_cmake_manifest(output_dir, sources):
    manifest_path = pathlib.Path(output_dir) / "manifest.cmake"
    lines = ["set(INFINI_OPS_NINETOOTHED_SOURCES"]
    lines.extend(f'    "{source}"' for source in sources)
    lines.append(")")
    lines.append("")
    lines.append(f'set(INFINI_OPS_NINETOOTHED_INCLUDE_DIRS "{output_dir}")')
    lines.append("")
    manifest_path.write_text("\n".join(lines) + "\n")


def _load_op_module(op):
    path = _find_op_modules()[op]
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def generate(ops, *, output_dir):
    op_modules = _find_op_modules()
    unknown_ops = tuple(op for op in ops if op not in op_modules)

    if unknown_ops:
        raise ValueError(f"unsupported NineToothed ops: {', '.join(unknown_ops)}")

    output_dir = pathlib.Path(output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for op in ops:
        module = _load_op_module(op)
        module.build(output_dir)

    sources = _build_manifest(output_dir)
    _write_cmake_manifest(output_dir, sources)

    return sources


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate NineToothed operator sources for InfiniOps."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ops", nargs="+", default=tuple(_find_op_modules()))

    return parser.parse_args()


def main():
    args = _parse_args()
    generate(args.ops, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
