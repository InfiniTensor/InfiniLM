import functools
import inspect
import itertools
import pathlib

import ninetoothed
from ninetoothed.aot import _HEADER_PATH

CURRENT_FILE_PATH = pathlib.Path(__file__)

BUILD_DIRECTORY_PATH = (
    CURRENT_FILE_PATH.parent.parent.parent.parent / "build" / "ninetoothed"
)


def build(premake, constexpr_param_grid, caller, op_name, output_dir):
    headers = []
    all_param_names = []
    launches = []

    for combination in _generate_param_value_combinations(constexpr_param_grid):
        arrangement, application, tensors = premake(**combination)

        for param_name, param_value in combination.items():
            if isinstance(param_value, str):
                combination[param_name] = (
                    f"INFINI_DTYPE_{combination[param_name].replace('fp', 'F').upper()}"
                )

        combination = {f"{name}_": value for name, value in combination.items()}

        kernel_name = f"{op_name}_{_generate_suffix(combination.values())}"

        ninetoothed.make(
            arrangement,
            application,
            tensors,
            caller=caller,
            kernel_name=kernel_name,
            output_dir=output_dir,
        )

        header = output_dir / f"{kernel_name}.h"
        param_names = ("stream",) + tuple(
            inspect.signature(application).parameters.keys()
        )
        launch = f"""    if ({_generate_condition(combination)})
        return launch_{kernel_name}({", ".join(param_names)});"""

        headers.append(header)
        all_param_names.append(param_names)
        launches.append(launch)

    includes = "\n".join(f'#include "{header}"' for header in headers)

    param_names = list(
        functools.reduce(
            lambda x, y: dict.fromkeys(x) | dict.fromkeys(y),
            sorted(all_param_names, key=len, reverse=True),
            {},
        )
    )
    param_types = [
        "NineToothedStream",
    ] + ["NineToothedTensor" for _ in range(len(param_names) - 1)]

    for param_name in combination:
        param_names.append(param_name)
        param_types.append("int")

    param_decls = ", ".join(
        f"{type} {param}" for param, type in zip(param_names, param_types)
    )

    source_file_name = f"{op_name}.c"
    header_file_name = f"{op_name}.h"

    func_sig = f"NineToothedResult launch_{op_name}({param_decls})"

    joined_launches = "\n".join(launches)

    op_decl = f'#ifdef __cplusplus\nextern "C" {func_sig};\n#else\n{func_sig};\n#endif'
    op_def = f"""{func_sig} {{
{joined_launches}
    return INFINI_STATUS_NOT_IMPLEMENTED;
}}"""

    source_content = f"""#include "{header_file_name}"

#include "infinicore.h"

{includes}\n\n{op_def}\n"""
    header_content = f"""#include "{_HEADER_PATH}"
\n{op_decl}\n"""

    (BUILD_DIRECTORY_PATH / source_file_name).write_text(source_content)
    (BUILD_DIRECTORY_PATH / header_file_name).write_text(header_content)


def _generate_condition(combination):
    return " && ".join(f"{param} == {value}" for param, value in combination.items())


def _generate_suffix(values):
    return "_".join(f"{value}" for value in values)


def _generate_param_value_combinations(param_grid):
    keys = list(param_grid.keys())
    value_combinations = itertools.product(*param_grid.values())

    return tuple(dict(zip(keys, combination)) for combination in value_combinations)
