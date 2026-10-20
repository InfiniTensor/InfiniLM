import argparse
import dataclasses
import fnmatch
import importlib.metadata
import json
import os
import pathlib
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ops_config  # noqa: E402

_PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
_DEFAULT_SOURCE_ROOT = _PROJECT_DIR / "src" / "linked"
_DEFAULT_OUTPUT_DIR = _PROJECT_DIR / "generated" / "linked"
_LIBRARY_KEYS = {
    "python_distribution_package",
    "python_distribution_version",
    "library_glob",
    "include_glob",
}
_BINDING_KEYS = {
    "library",
    "link_libraries",
    "required_symbols",
    "operator_schema",
    "dispatch_key",
}
_SUPPORTED_TRANSPORTS = {"torch", "tvm_ffi"}


class ResolutionError(RuntimeError):
    pass


class _StrictLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)

    return mapping


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


@dataclasses.dataclass(frozen=True)
class LibraryConfig:
    device: str
    transport: str
    name: str
    path: pathlib.Path
    python_distribution_package: str
    library_glob: str
    include_glob: str | None = None
    python_distribution_version: str | None = None


@dataclasses.dataclass(frozen=True)
class BindingConfig:
    device: str
    transport: str
    implementation: str
    name: str
    path: pathlib.Path
    source: pathlib.Path
    library: str
    required_symbols: tuple[str, ...]
    operator_schema: str | None
    dispatch_key: str | None
    link_libraries: tuple[str, ...] = ()


def _load_yaml_mapping(path, expected_keys, required_keys=None):
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_StrictLoader)
    except (OSError, yaml.YAMLError) as error:
        raise ResolutionError(f"failed to read {path}: {error}") from error

    if not isinstance(data, dict):
        raise ResolutionError(f"{path} must contain a YAML mapping")

    keys = set(data)
    unknown_keys = sorted(keys - expected_keys)
    required_keys = expected_keys if required_keys is None else required_keys
    missing_keys = sorted(required_keys - keys)
    if unknown_keys:
        raise ResolutionError(
            f"{path} contains unknown keys: {', '.join(unknown_keys)}"
        )
    if missing_keys:
        raise ResolutionError(
            f"{path} is missing required keys: {', '.join(missing_keys)}"
        )

    return data


def _require_string(data, key, path):
    value = data[key]
    if not isinstance(value, str) or not value.strip():
        raise ResolutionError(f"{path}: {key} must be a non-empty string")
    return value.strip()


def _require_relative_glob(data, key, path):
    value = _require_string(data, key, path)
    normalized = pathlib.PurePosixPath(value.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ResolutionError(f"{path}: {key} must be relative to the distribution")
    return normalized.as_posix()


def _find_platform_dirs(source_root, device):
    platform_dirs = []
    for path in sorted(source_root.glob(f"*/{device}")):
        if not path.is_dir():
            continue

        transport = path.parent.name
        if transport not in _SUPPORTED_TRANSPORTS:
            supported = ", ".join(sorted(_SUPPORTED_TRANSPORTS))
            raise ResolutionError(
                f"{path}: unsupported transport {transport!r}; expected {supported}"
            )
        platform_dirs.append((transport, path))

    return platform_dirs


def _load_libraries(platform_dir, device, transport, selected_libraries=None):
    libraries = {}
    for path in sorted(platform_dir.glob("*.yaml")):
        if selected_libraries is not None and path.stem not in selected_libraries:
            continue
        data = _load_yaml_mapping(
            path,
            _LIBRARY_KEYS,
            {"python_distribution_package", "library_glob"},
        )
        libraries[path.stem] = LibraryConfig(
            device=device,
            transport=transport,
            name=path.stem,
            path=path,
            python_distribution_package=_require_string(
                data, "python_distribution_package", path
            ),
            library_glob=_require_relative_glob(data, "library_glob", path),
            python_distribution_version=(
                _require_string(data, "python_distribution_version", path)
                if "python_distribution_version" in data
                else None
            ),
            include_glob=(
                _require_relative_glob(data, "include_glob", path)
                if "include_glob" in data
                else None
            ),
        )

    return libraries


def _binding_is_selected(name, header, selected_ops, config):
    if selected_ops is not None and name not in selected_ops:
        return False

    if config is not None:
        selection = config.get(name)

        if selection is None:
            return False

        headers = selection["headers"]

        if headers is not None:
            selected_headers = {
                (
                    _PROJECT_DIR / ops_config.implementation_path(selected_header)
                ).resolve()
                for selected_header in headers
            }

            return header.resolve() in selected_headers

        slots = selection["implementations"]

        return slots is None or ops_config.implementation_slot(header) in slots

    return selected_ops is None or name in selected_ops


def _load_bindings(platform_dir, device, transport, selected_ops, config):
    bindings = []
    for path in sorted((platform_dir / "ops").glob("*/*.yaml")):
        name = path.parent.name
        header = path.with_suffix(".h")

        if config is not None and name not in config:
            continue
        if config is None and selected_ops is not None and name not in selected_ops:
            continue
        if not header.is_file():
            raise ResolutionError(f"{path}: missing sibling {header.name}")

        try:
            selected = _binding_is_selected(name, header, selected_ops, config)
        except ops_config.OpsConfigError as error:
            raise ResolutionError(str(error)) from error

        if not selected:
            continue

        source = path.with_suffix(".cc")
        cuda_source = path.with_suffix(".cu")
        if not source.is_file():
            if not cuda_source.is_file():
                raise ResolutionError(
                    f"{path}: missing sibling {source.name} or {cuda_source.name}"
                )
            source = cuda_source
        elif cuda_source.is_file():
            raise ResolutionError(
                f"{path}: both {source.name} and {cuda_source.name} are present"
            )

        data = _load_yaml_mapping(path, _BINDING_KEYS, {"library"})
        symbols = data.get("required_symbols")
        operator_schema = data.get("operator_schema")
        dispatch_key = data.get("dispatch_key")

        link_libraries = data.get("link_libraries", [])
        if not isinstance(link_libraries, list) or any(
            not isinstance(library, str) or not library.strip()
            for library in link_libraries
        ):
            raise ResolutionError(
                f"{path}: link_libraries must be a list of non-empty strings"
            )
        link_libraries = tuple(library.strip() for library in link_libraries)
        if len(link_libraries) != len(set(link_libraries)):
            raise ResolutionError(f"{path}: link_libraries contains duplicates")
        if (symbols is None) == (operator_schema is None):
            raise ResolutionError(
                f"{path} must define exactly one of required_symbols or operator_schema"
            )

        if symbols is not None:
            if dispatch_key is not None:
                raise ResolutionError(f"{path}: dispatch_key requires operator_schema")
            if (
                not isinstance(symbols, list)
                or not symbols
                or any(
                    not isinstance(symbol, str) or not symbol.strip()
                    for symbol in symbols
                )
            ):
                raise ResolutionError(
                    f"{path}: required_symbols must be a non-empty list of strings"
                )
            symbols = tuple(symbol.strip() for symbol in symbols)
            if len(symbols) != len(set(symbols)):
                raise ResolutionError(f"{path}: required_symbols contains duplicates")
            operator_schema = None
        else:
            symbols = ()
            operator_schema = _require_string(data, "operator_schema", path)
            if dispatch_key is None:
                raise ResolutionError(f"{path}: operator_schema requires dispatch_key")
            dispatch_key = _require_string(data, "dispatch_key", path)

        bindings.append(
            BindingConfig(
                device=device,
                transport=transport,
                implementation=path.stem,
                name=name,
                path=path,
                source=source.resolve(),
                library=_require_string(data, "library", path),
                required_symbols=symbols,
                operator_schema=operator_schema,
                dispatch_key=dispatch_key,
                link_libraries=link_libraries,
            )
        )

    return bindings


def _matches_library_glob(relative, pattern):
    relative_parts = pathlib.PurePosixPath(relative).parts
    pattern_parts = pathlib.PurePosixPath(pattern).parts
    return len(relative_parts) == len(pattern_parts) and all(
        fnmatch.fnmatchcase(part, pattern_part)
        for part, pattern_part in zip(relative_parts, pattern_parts)
    )


def _locate_editable_distribution_root(distribution):
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is None:
        return None

    try:
        metadata = json.loads(direct_url)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(metadata, dict):
        return None

    directory = metadata.get("dir_info")
    if not isinstance(directory, dict) or directory.get("editable") is not True:
        return None

    url = metadata.get("url")
    if not isinstance(url, str):
        return None

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "file":
        return None

    path = urllib.request.url2pathname(parsed.path)
    if parsed.netloc:
        path = f"//{parsed.netloc}{path}"

    root = pathlib.Path(path)
    if not root.is_absolute():
        return None

    root = root.resolve()
    return root if root.is_dir() else None


def _load_distribution(config):
    try:
        distribution = importlib.metadata.distribution(
            config.python_distribution_package
        )
    except importlib.metadata.PackageNotFoundError as error:
        raise ResolutionError(
            f"Python distribution package "
            f"{config.python_distribution_package!r} required by "
            f"{config.path} is not installed"
        ) from error
    if config.python_distribution_version is not None:
        try:
            constraint = SpecifierSet(config.python_distribution_version)
            version = Version(distribution.version)
        except (InvalidSpecifier, InvalidVersion) as error:
            raise ResolutionError(
                f"{config.path}: invalid Python distribution version constraint"
            ) from error
        if version not in constraint:
            raise ResolutionError(
                f"{config.path}: {config.python_distribution_package!r} version "
                f"{distribution.version!r} does not satisfy "
                f"{config.python_distribution_version!r}"
            )

    return distribution


def _locate_distribution_library(config):
    distribution = _load_distribution(config)

    matches = []
    distribution_root = pathlib.Path(distribution.locate_file("")).resolve()
    for entry in distribution.files or ():
        relative = pathlib.PurePosixPath(str(entry).replace("\\", "/"))
        if (
            not relative.is_absolute()
            and ".." not in relative.parts
            and _matches_library_glob(relative, config.library_glob)
        ):
            candidate = pathlib.Path(distribution.locate_file(entry)).resolve()
            if candidate.is_file() and candidate.is_relative_to(distribution_root):
                matches.append(candidate)

    search_roots = [distribution_root]
    editable_root = _locate_editable_distribution_root(distribution)
    if editable_root is not None:
        search_roots.append(editable_root)

    # Some vendor wheels intentionally ship sparse or empty RECORD metadata.
    # Editable distributions also keep project files outside the metadata root.
    for root in search_roots:
        for candidate in root.glob(config.library_glob):
            candidate = candidate.resolve()
            if candidate.is_file() and candidate.is_relative_to(root):
                matches.append(candidate)

    matches = sorted(set(matches))
    if len(matches) != 1:
        formatted = ", ".join(str(path) for path in matches) or "none"
        raise ResolutionError(
            f"{config.path}: library_glob {config.library_glob!r} matched "
            f"{len(matches)} files in "
            f"{config.python_distribution_package!r}: {formatted}"
        )

    return matches[0]


def _locate_distribution_include(config):
    if config.include_glob is None:
        return None

    distribution = _load_distribution(config)

    roots = [pathlib.Path(distribution.locate_file("")).resolve()]
    editable_root = _locate_editable_distribution_root(distribution)
    if editable_root is not None:
        roots.append(editable_root)

    matches = []
    for root in roots:
        for candidate in root.glob(config.include_glob):
            candidate = candidate.resolve()
            if candidate.is_dir() and candidate.is_relative_to(root):
                matches.append(candidate)

    matches = sorted(set(matches))
    if len(matches) != 1:
        formatted = ", ".join(str(path) for path in matches) or "none"
        raise ResolutionError(
            f"{config.path}: include_glob {config.include_glob!r} matched "
            f"{len(matches)} directories in "
            f"{config.python_distribution_package!r}: {formatted}"
        )

    return matches[0]


def _run_symbol_tool(command, library_path):
    try:
        result = subprocess.run(
            [*command, str(library_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        stderr = getattr(error, "stderr", "") or ""
        detail = f": {stderr.strip()}" if stderr.strip() else ""
        raise ResolutionError(
            f"failed to inspect {library_path} with {command[0]}{detail}"
        ) from error

    return result.stdout


def _parse_nm_symbols(output):
    symbols = set()
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) == 3:
            symbols.add(fields[2])
    return symbols


def _parse_readelf_symbols(output):
    symbols = set()
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=7)
        if (
            len(fields) == 8
            and fields[0].endswith(":")
            and fields[3] == "FUNC"
            and fields[4] == "GLOBAL"
            and fields[5] == "DEFAULT"
            and fields[6] != "UND"
        ):
            symbols.add(fields[7])
    return symbols


def _demangle_symbols(symbols, cxxfilt):
    if not symbols:
        return set()

    ordered = sorted(symbols)
    try:
        result = subprocess.run(
            [cxxfilt],
            check=True,
            capture_output=True,
            text=True,
            input="\n".join(ordered) + "\n",
        )
    except (OSError, subprocess.CalledProcessError) as error:
        stderr = getattr(error, "stderr", "") or ""
        detail = f": {stderr.strip()}" if stderr.strip() else ""
        raise ResolutionError(
            f"failed to demangle dynamic symbols with {cxxfilt}{detail}"
        ) from error

    demangled = result.stdout.splitlines()
    if len(demangled) != len(ordered):
        raise ResolutionError(f"{cxxfilt} returned an unexpected number of symbols")

    return set(demangled)


def _inspect_dynamic_symbols(library_path, nm, readelf, cxxfilt):
    nm_output = _run_symbol_tool([nm, "-D", "--defined-only", "-C"], library_path)
    readelf_output = _run_symbol_tool([readelf, "--dyn-syms", "--wide"], library_path)
    readelf_symbols = _parse_readelf_symbols(readelf_output)
    return _parse_nm_symbols(nm_output), _demangle_symbols(readelf_symbols, cxxfilt)


def _matches_required_symbol(exported, required):
    return exported == required


def _verify_required_symbols(config, library_path, nm_symbols, readelf_symbols):
    for required in config.required_symbols:
        missing_from = []
        if not any(
            _matches_required_symbol(exported, required) for exported in nm_symbols
        ):
            missing_from.append("nm")
        if not any(
            _matches_required_symbol(exported, required) for exported in readelf_symbols
        ):
            missing_from.append("readelf")
        if missing_from:
            raise ResolutionError(
                f"{config.path}: required symbol {required!r} is not exported by "
                f"{library_path} according to {' and '.join(missing_from)}"
            )


def _verify_dispatcher_contracts(contracts):
    payload = [
        {
            "binding_path": str(config.path),
            "library_path": str(library_path),
            "schema": config.operator_schema,
            "dispatch_key": config.dispatch_key,
        }
        for config, library_path in contracts
    ]
    script = (
        "import json\n"
        "import sys\n"
        "import torch\n"
        "contracts = json.loads(sys.argv[1])\n"
        "loaded = set()\n"
        "for contract in contracts:\n"
        "    library_path = contract['library_path']\n"
        "    if library_path not in loaded:\n"
        "        torch.ops.load_library(library_path)\n"
        "        loaded.add(library_path)\n"
        "for contract in contracts:\n"
        "    expected = torch._C.parse_schema(contract['schema'])\n"
        "    actual = torch._C._dispatch_find_schema_or_throw(\n"
        "        expected.name, expected.overload_name\n"
        "    ).schema()\n"
        "    if str(actual) != str(expected):\n"
        "        sys.exit(\n"
        "            f\"{contract['binding_path']}: expected {expected}, \"\n"
        "            f'found {actual}'\n"
        "        )\n"
        "    qualified_name = expected.name\n"
        "    if expected.overload_name:\n"
        "        qualified_name += f'.{expected.overload_name}'\n"
        "    dispatch_key = contract['dispatch_key']\n"
        "    if not torch._C._dispatch_has_kernel_for_dispatch_key(\n"
        "        qualified_name, dispatch_key\n"
        "    ):\n"
        "        sys.exit(\n"
        "            f\"{contract['binding_path']}: {qualified_name} has no \"\n"
        "            f'{dispatch_key} kernel'\n"
        "        )\n"
    )
    try:
        subprocess.run(
            [sys.executable, "-c", script, json.dumps(payload)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        stderr = getattr(error, "stderr", "") or ""
        detail = f": {stderr.strip()}" if stderr.strip() else ""
        raise ResolutionError(
            f"dispatcher contracts are not provided by the resolved libraries{detail}"
        ) from error


def _cmake_quote(value):
    return str(value).replace("\\", "/").replace(";", "\\;").replace('"', '\\"')


def _render_cmake_manifest(payload):
    variables = {
        "INFINI_OPS_LINKED_SOURCES": [
            operator["source"] for operator in payload["operators"]
        ],
        "INFINI_OPS_LINKED_TORCH_SOURCES": [
            operator["source"]
            for operator in payload["operators"]
            if operator["transport"] == "torch"
        ],
        "INFINI_OPS_LINKED_TVM_FFI_SOURCES": [
            operator["source"]
            for operator in payload["operators"]
            if operator["transport"] == "tvm_ffi"
        ],
        "INFINI_OPS_LINKED_LIBRARIES": [
            library["path"] for library in payload["libraries"]
        ],
        "INFINI_OPS_LINKED_FORCE_LOAD_LIBRARIES": [
            library["path"] for library in payload["libraries"] if library["force_load"]
        ],
        "INFINI_OPS_LINKED_RUNTIME_DIRS": [
            library["runtime_dir"] for library in payload["libraries"]
        ],
        "INFINI_OPS_LINKED_INCLUDE_DIRS": [
            library["include_dir"]
            for library in payload["libraries"]
            if "include_dir" in library
        ],
        "INFINI_OPS_LINKED_TRANSPORTS": [
            library["transport"] for library in payload["libraries"]
        ],
    }

    lines = ["# Generated by scripts/resolve_linked_ops.py. Do not edit."]
    for name, values in variables.items():
        lines.append(f"set({name}")
        lines.extend(f'    "{_cmake_quote(value)}"' for value in sorted(set(values)))
        lines.append(")")
        lines.append("")

    return "\n".join(lines)


def _write_if_changed(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def _normalize_values(values):
    if values is None:
        return None
    return tuple(
        dict.fromkeys(
            item.strip().lower()
            for value in values
            for item in re.split(r"[;,]", value)
            if item.strip()
        )
    )


def resolve_linked_ops(
    devices,
    ops=None,
    config_path=None,
    *,
    source_root=_DEFAULT_SOURCE_ROOT,
    output_dir=_DEFAULT_OUTPUT_DIR,
    nm="nm",
    readelf="readelf",
    cxxfilt="c++filt",
):
    source_root = pathlib.Path(source_root).resolve()
    output_dir = pathlib.Path(output_dir).resolve()
    devices = _normalize_values(devices) or ()
    selected_ops = _normalize_values(ops)
    selected_op_set = set(selected_ops) if selected_ops is not None else None

    try:
        selection_config = (
            ops_config.load_ops_config(config_path) if config_path is not None else None
        )
    except ops_config.OpsConfigError as error:
        raise ResolutionError(str(error)) from error

    bindings = []
    library_configs = {}
    for device in devices:
        for transport, platform_dir in _find_platform_dirs(source_root, device):
            platform_bindings = _load_bindings(
                platform_dir,
                device,
                transport,
                selected_op_set,
                selection_config,
            )
            bindings.extend(platform_bindings)
            selected_libraries = {
                library
                for binding in platform_bindings
                for library in (binding.library, *binding.link_libraries)
            }
            libraries = _load_libraries(
                platform_dir,
                device,
                transport,
                selected_libraries,
            )
            for name, library_config in libraries.items():
                library_configs[(transport, device, name)] = library_config

    bindings.sort(
        key=lambda binding: (
            binding.transport,
            binding.device,
            binding.name,
            binding.implementation,
        )
    )
    resolved_libraries = {}
    inspected_symbols = {}
    dispatcher_contracts = []
    for binding in bindings:
        for library_name in (binding.library, *binding.link_libraries):
            dependency_key = (binding.transport, binding.device, library_name)
            library_config = library_configs.get(dependency_key)
            if library_config is None:
                raise ResolutionError(
                    f"{binding.path}: unknown library {library_name!r} for "
                    f"device {binding.device}"
                )
            if dependency_key not in resolved_libraries:
                resolved_libraries[dependency_key] = _locate_distribution_library(
                    library_config
                )
        key = (binding.transport, binding.device, binding.library)

        library_path = resolved_libraries[key]
        if binding.required_symbols:
            if key not in inspected_symbols:
                inspected_symbols[key] = _inspect_dynamic_symbols(
                    library_path, nm, readelf, cxxfilt
                )
            nm_symbols, readelf_symbols = inspected_symbols[key]
            _verify_required_symbols(binding, library_path, nm_symbols, readelf_symbols)
        else:
            dispatcher_contracts.append((binding, library_path))

    if dispatcher_contracts:
        _verify_dispatcher_contracts(dispatcher_contracts)
    required_symbols = {
        symbol for binding in bindings for symbol in binding.required_symbols
    }
    for required_symbol in sorted(required_symbols):
        exporters = {
            resolved_libraries[key]
            for key, (nm_symbols, readelf_symbols) in inspected_symbols.items()
            if required_symbol in nm_symbols and required_symbol in readelf_symbols
        }
        if len(exporters) > 1:
            formatted = ", ".join(str(path) for path in sorted(exporters))
            raise ResolutionError(
                f"required symbol {required_symbol!r} is exported by multiple "
                f"linked libraries: {formatted}"
            )

    libraries_by_basename = {}
    for library_path in resolved_libraries.values():
        previous = libraries_by_basename.setdefault(library_path.name, library_path)
        if previous != library_path:
            raise ResolutionError(
                f"linked libraries share basename {library_path.name!r}: "
                f"{previous} and {library_path}"
            )

    force_load_keys = {
        (binding.transport, binding.device, binding.library)
        for binding in bindings
        if binding.operator_schema is not None
    }
    force_load_keys.update(
        (binding.transport, binding.device, library)
        for binding in bindings
        for library in binding.link_libraries
    )
    libraries = []
    for key in sorted(resolved_libraries):
        config = library_configs[key]
        library_path = resolved_libraries[key]
        include_dir = _locate_distribution_include(config)
        libraries.append(
            {
                "device": config.device,
                "force_load": key in force_load_keys,
                "name": config.name,
                "path": str(library_path),
                "python_distribution_package": (config.python_distribution_package),
                "runtime_dir": str(library_path.parent),
                "transport": config.transport,
            }
        )
        if include_dir is not None:
            libraries[-1]["include_dir"] = str(include_dir)

    operators = []
    for binding in bindings:
        operator = {
            "device": binding.device,
            "transport": binding.transport,
            "library": binding.library,
            "implementation": binding.implementation,
            "name": binding.name,
            "source": str(binding.source),
        }
        if binding.link_libraries:
            operator["link_libraries"] = list(binding.link_libraries)
        if binding.required_symbols:
            operator["required_symbols"] = list(binding.required_symbols)
        else:
            operator["operator_schema"] = binding.operator_schema
            operator["dispatch_key"] = binding.dispatch_key
        operators.append(operator)

    payload = {
        "devices": list(devices),
        "libraries": libraries,
        "operators": operators,
    }
    _write_if_changed(output_dir / "manifest.cmake", _render_cmake_manifest(payload))
    _write_if_changed(
        output_dir / "resolved.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )

    return payload


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Resolve installed linked operator libraries for InfiniOps."
    )
    parser.add_argument("--devices", nargs="+", required=True)
    parser.add_argument("--ops", nargs="*")
    parser.add_argument("--ops-config", type=pathlib.Path)
    parser.add_argument("--source-root", default=_DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--nm", default=os.environ.get("CMAKE_NM", "nm"))
    parser.add_argument("--readelf", default=os.environ.get("CMAKE_READELF", "readelf"))
    parser.add_argument("--cxxfilt", default="c++filt")
    return parser.parse_args()


def _selection_from_environment(ops, config_path):
    if ops is not None or config_path is not None:
        return ops, config_path

    value = os.environ.get("INFINI_OPS_OPS")

    if value is None:
        return ops, config_path
    if pathlib.Path(value).suffix.lower() == ".json":
        return None, pathlib.Path(value)

    return [value], None


def main():
    args = _parse_args()
    ops, config_path = _selection_from_environment(args.ops, args.ops_config)
    try:
        resolve_linked_ops(
            args.devices,
            ops,
            config_path,
            source_root=args.source_root,
            output_dir=args.output_dir,
            nm=args.nm,
            readelf=args.readelf,
            cxxfilt=args.cxxfilt,
        )
    except ResolutionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
