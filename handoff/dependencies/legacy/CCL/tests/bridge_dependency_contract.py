import pathlib
import re
import sys


def main():
    source = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")

    header_glob = re.search(
        r"file\(GLOB_RECURSE BRIDGE_DEP_HEADERS CONFIGURE_DEPENDS(.*?)\n\)",
        source,
        re.DOTALL,
    )
    assert header_glob is not None
    assert set(re.findall(r'"([^"]+)"', header_glob.group(1))) == {
        "base/*.h",
        "backends/*.h",
        "devices/*.h",
        "devices/*.cuh",
    }

    assert re.search(
        r"file\(GENERATE\s+"
        r'OUTPUT "\$\{BRIDGE_DEPENDENCY_MANIFEST\}"\s+'
        r'CONTENT "\$\{BRIDGE_DEPENDENCY_CONTENT\}\\n"\s*'
        r"\)",
        source,
    )

    bridge_command = re.search(
        r"add_custom_command\(\s+"
        r'OUTPUT "\$\{GENERATED_BRIDGE\}" "\$\{GENERATED_MANIFEST\}"'
        r"(.*?)\n\)",
        source,
        re.DOTALL,
    )
    assert bridge_command is not None
    dependencies = bridge_command.group(1)
    assert '"${BRIDGE_DEPENDENCY_MANIFEST}"' in dependencies
    assert "${BRIDGE_DEP_HEADERS}" in dependencies


if __name__ == "__main__":
    main()
