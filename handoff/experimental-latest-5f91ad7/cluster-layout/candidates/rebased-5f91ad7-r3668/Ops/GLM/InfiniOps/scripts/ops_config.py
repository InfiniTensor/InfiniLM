import json
import pathlib
import re


_OPERATOR_SPECIALIZATION_RE = re.compile(
    r"\bclass\s+Operator<\s*[^,>]+\s*,\s*[^,>]+\s*"
    r"(?:,\s*(\d+)\s*)?>"
)


class OpsConfigError(ValueError):
    pass


class _StrictJsonObject(dict):
    pass


def _strict_object(pairs):
    value = _StrictJsonObject()

    for key, item in pairs:
        if key in value:
            raise OpsConfigError(f"duplicate key {key!r}")
        value[key] = item

    return value


def load_ops_config(path):
    path = pathlib.Path(path)

    try:
        config = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object
        )
    except (OSError, json.JSONDecodeError, OpsConfigError) as error:
        raise OpsConfigError(f"failed to read {path}: {error}") from error

    if not isinstance(config, dict):
        raise OpsConfigError(f"{path} must contain a JSON object")

    normalized = {}

    for op_name, value in config.items():
        if not isinstance(op_name, str) or not op_name.strip():
            raise OpsConfigError(f"{path}: operator names must be non-empty strings")
        if op_name != op_name.strip():
            raise OpsConfigError(
                f"{path}: operator name {op_name!r} contains surrounding whitespace"
            )

        normalized[op_name] = _normalize_selection(path, op_name, value)

    return normalized


def _normalize_selection(path, op_name, value):
    if isinstance(value, str):
        if not value.strip():
            raise OpsConfigError(
                f"{path}: {op_name!r} implementation path must not be empty"
            )

        return {"headers": [value], "implementations": None}

    if isinstance(value, list):
        if not value:
            raise OpsConfigError(
                f"{path}: {op_name!r} implementation paths must be a non-empty array"
            )
        headers = [_normalize_header(path, op_name, item) for item in value]
        identities = [
            header if isinstance(header, str) else (header["path"], header["backend"])
            for header in headers
        ]
        if len(identities) != len(set(identities)):
            raise OpsConfigError(
                f"{path}: {op_name!r} implementation paths contain duplicates"
            )
        return {"headers": headers, "implementations": None}

    if not isinstance(value, dict):
        raise OpsConfigError(
            f"{path}: {op_name!r} must map to implementation path(s) or an object"
        )

    if "public_api_only" in value:
        if set(value) != {"public_api_only"} or value["public_api_only"] is not True:
            raise OpsConfigError(
                f"{path}: {op_name!r} public_api_only must be exactly true "
                "without implementation selections"
            )
        return {"headers": None, "implementations": (), "public_api_only": True}

    unknown_keys = sorted(set(value) - {"implementations"})
    if unknown_keys:
        raise OpsConfigError(
            f"{path}: {op_name!r} contains unknown keys: {', '.join(unknown_keys)}"
        )
    if "implementations" not in value:
        raise OpsConfigError(
            f"{path}: {op_name!r} is missing required key 'implementations'"
        )

    implementations = value["implementations"]

    if implementations == "all":
        implementations = None
    elif isinstance(implementations, list):
        if not implementations:
            raise OpsConfigError(
                f"{path}: {op_name!r} implementations must not be empty"
            )
        if any(type(slot) is not int or not 0 <= slot < 32 for slot in implementations):
            raise OpsConfigError(
                f"{path}: {op_name!r} implementations must contain integers "
                "between 0 and 31"
            )
        if len(implementations) != len(set(implementations)):
            raise OpsConfigError(
                f"{path}: {op_name!r} implementations contain duplicates"
            )
        implementations = tuple(implementations)
    else:
        raise OpsConfigError(
            f"{path}: {op_name!r} implementations must be 'all' or an array"
        )

    return {"headers": None, "implementations": implementations}


def _normalize_header(path, op_name, value):
    if isinstance(value, str):
        if value.strip():
            return value
        raise OpsConfigError(
            f"{path}: {op_name!r} implementation path must not be empty"
        )

    if not isinstance(value, dict):
        raise OpsConfigError(
            f"{path}: {op_name!r} implementation entries must be paths or "
            "structured descriptors"
        )

    unknown_keys = sorted(set(value) - {"path", "backend"})
    if unknown_keys:
        raise OpsConfigError(
            f"{path}: {op_name!r} implementation descriptor contains unknown "
            f"keys: {', '.join(unknown_keys)}"
        )

    for key in ("path", "backend"):
        if (
            key not in value
            or not isinstance(value[key], str)
            or not value[key].strip()
        ):
            raise OpsConfigError(
                f"{path}: {op_name!r} implementation descriptor requires a "
                f"non-empty string {key!r}"
            )

    return {"path": value["path"], "backend": value["backend"]}


def implementation_path(header):
    return header if isinstance(header, str) else header["path"]


def selected_op_names(config):
    return list(config)


def selected_slots(config, op_name):
    selection = config.get(op_name)

    if selection is None or selection["headers"] is not None:
        return None

    return selection["implementations"]


def implementation_slot(path):
    path = pathlib.Path(path)

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise OpsConfigError(f"failed to read {path}: {error}") from error

    slots = {
        int(match.group(1)) if match.group(1) is not None else 0
        for match in _OPERATOR_SPECIALIZATION_RE.finditer(text)
    }

    if len(slots) != 1:
        formatted = ", ".join(str(slot) for slot in sorted(slots)) or "none"
        raise OpsConfigError(
            f"{path} must declare exactly one implementation slot; found {formatted}"
        )

    return slots.pop()


def torch_op_names(config, default_ops=(), slot=8):
    selected = []

    for op_name, selection in config.items():
        if selection["headers"] is not None:
            continue

        implementations = selection["implementations"]

        if (implementations is None and op_name in default_ops) or (
            implementations is not None and slot in implementations
        ):
            selected.append(op_name)

    return selected
