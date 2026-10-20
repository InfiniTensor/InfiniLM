import os
import re
import socket
import subprocess
import sys
from pathlib import Path

import yaml

PATH_LIKE = {"LD_LIBRARY_PATH", "PATH", "CPATH", "LIBRARY_PATH"}


def _is_infiniccl_source_dir(path):
    cmake_path = Path(path).expanduser() / "CMakeLists.txt"
    try:
        cmake_text = cmake_path.read_text(errors="ignore")
    except OSError:
        return False

    return re.search(r"project\s*\(\s*InfiniCCL(?:\s|\))", cmake_text) is not None


class ICCLLauncher:
    def __init__(self, manual_config_path):
        self.config = None
        # Priority: 1. Argument, 2. Current Dir, 3. examples/ folder.
        search_paths = [
            manual_config_path,
            "./cluster.yaml",
            os.path.join(os.path.dirname(__file__), "../examples/cluster.yaml"),
        ]

        for path in search_paths:
            if path and os.path.exists(path):
                with open(path, "r") as f:
                    self.config = yaml.safe_load(f)
                break

        if not self.config:
            print("Error: `cluster.yaml` not found.")
            sys.exit(1)

        # Detect the InfiniCCL source root. Prefer the configured tree
        # checkout before environment variables, which may point at stale installs.
        script_dir = os.path.dirname(os.path.realpath(__file__))
        candidates = []
        common_dir = self.config.get("common_dir")
        if common_dir:
            common_dir = os.path.expanduser(common_dir)
            candidates.extend([common_dir])
        candidates.extend(
            root
            for root in [
                os.environ.get("INFINICCL_ROOT"),
                os.environ.get("InfiniCCL_ROOT"),
            ]
            if root
        )
        candidates.extend(
            [
                os.path.join(script_dir, ".."),
                os.path.join(script_dir, "../.."),
                os.path.join(script_dir, "../../.."),
            ]
        )

        for candidate in candidates:
            root = os.path.abspath(os.path.expanduser(candidate))
            if _is_infiniccl_source_dir(root):
                self.infiniccl_root = root
                break
        else:
            self.infiniccl_root = os.path.abspath(os.path.join(script_dir, "../../.."))

    def _is_local(self, ip):
        """Check if the IP/hostname refers to the local machine."""
        try:
            local_ips = {"127.0.0.1", "localhost"}

            # Add hostname-resolved IPs.
            local_ips.update(socket.gethostbyname_ex(socket.gethostname())[2])

            # Add all interface IPs via `hostname -I`.
            try:
                import subprocess

                ips = (
                    subprocess.check_output(["hostname", "-I"], text=True)
                    .strip()
                    .split()
                )
                local_ips.update(ips)
            except Exception:
                pass

            # Resolve input to IP if possible.
            try:
                ip_resolved = socket.gethostbyname(ip)
            except Exception:
                ip_resolved = ip

            return ip in local_ips or ip_resolved in local_ips

        except Exception:
            return False

    def orchestrate_build(self):
        global_common_dir = self.config["common_dir"]
        infiniccl_root = self.infiniccl_root

        for node in self.config["nodes"]:
            node_common_dir = node.get("dir", global_common_dir)
            node_is_local = self._is_local(node["ip"])
            base_install = self.config.get("install_dir", node_common_dir)

            arch = node["type"]
            install_path = os.path.join(base_install, "install", arch)
            user_cmake_flags = node.get(
                "cmake_flags", self.config.get("cmake_flags", "")
            )

            internal_cmd = (
                f"mkdir -p {node_common_dir}/build/{arch} && cd {node_common_dir}/build/{arch} && "
                f"cmake -DCMAKE_INSTALL_PREFIX={install_path} {user_cmake_flags} {node_common_dir} && "
                f"make -j$(nproc) install"
            )
            lib_cmd = (
                f"mkdir -p {infiniccl_root}/build/{arch} && cd {infiniccl_root}/build/{arch} && "
                f"cmake -DCMAKE_INSTALL_PREFIX={install_path} {user_cmake_flags} {infiniccl_root} && "
                f"make -j$(nproc) install"
            )
            app_cmd = (
                f"export INFINICCL_INSTALL={install_path} && "
                f"mkdir -p {node_common_dir}/build/{arch} && cd {node_common_dir}/build/{arch} && "
                f"cmake {user_cmake_flags} {node_common_dir} && "
                f"make -j$(nproc)"
            )
            external_cmd = f"{lib_cmd} && {app_cmd}"

            if node_is_local:
                full_cmd = (
                    internal_cmd
                    if _is_infiniccl_source_dir(node_common_dir)
                    else external_cmd
                )
            else:
                full_cmd = (
                    f'if grep -Eq "project[[:space:]]*\\([[:space:]]*InfiniCCL([[:space:]]|\\))" {node_common_dir}/CMakeLists.txt 2>/dev/null; '
                    f"then {internal_cmd}; else {external_cmd}; fi"
                )

            # Execute via SSH or locally.
            user = node.get("user", self.config.get("common_user", "root"))
            print(f"[*] Orchestrating {arch} on {node['ip']}...")

            if node_is_local:
                subprocess.run(["bash", "-l", "-c", full_cmd], check=True)
            else:
                subprocess.run(
                    ["ssh", f"{user}@{node['ip']}", f"bash -l -c '{full_cmd}'"],
                    check=True,
                )

        return self.ensure_launcher_exists()

    def ensure_launcher_exists(self):
        global_common_dir = Path(self.config["common_dir"]).expanduser().resolve()
        wrapper_path = str(global_common_dir / "build" / "run_wrapper.sh")
        os.makedirs(os.path.dirname(wrapper_path), exist_ok=True)

        global_env = self.config.get("backend_env", {})

        blocks = []
        first = True

        for node in self.config["nodes"]:
            ip_or_host = node["ip"]
            n_type = node["type"]

            node_common_dir_str = node.get("dir", self.config["common_dir"])
            node_is_local = self._is_local(ip_or_host)
            node_common_dir = (
                str(Path(node_common_dir_str).expanduser().resolve())
                if node_is_local
                else node_common_dir_str
            )

            base_install_str = self.config.get("install_dir", node_common_dir_str)
            base_install = (
                str(Path(base_install_str).expanduser().resolve())
                if node_is_local
                else base_install_str
            )

            install_lib = os.path.join(base_install, "install", n_type, "lib")

            resolved_ips = set()

            if ip_or_host in ("localhost", "127.0.0.1"):
                try:
                    hostname = socket.gethostname()
                    for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
                        ip = info[4][0]
                        if not ip.startswith("127."):
                            resolved_ips.add(ip)

                except Exception:
                    pass

                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.connect(("8.8.8.8", 80))
                    resolved_ips.add(s.getsockname()[0])
                    s.close()
                except Exception:
                    pass

                if not resolved_ips:
                    print(
                        "[ERROR] Failed to determine local machine IPs", file=sys.stderr
                    )
                    sys.exit(1)

            else:
                try:
                    resolved_ips.add(socket.gethostbyname(ip_or_host))
                except Exception:
                    print(
                        f"[ERROR] Failed to resolve node identifier '{ip_or_host}'",
                        file=sys.stderr,
                    )
                    sys.exit(1)

            node_env = node.get("backend_env", {})
            merged_env = global_env.copy()

            for env_key, env_value in node_env.items():
                if env_key in PATH_LIKE and env_key in merged_env:
                    merged_env[env_key] = f"{env_value}:{merged_env[env_key]}"
                else:
                    merged_env[env_key] = env_value

            ip_conditions = [
                f'[[ "$HOST_IPS" == *"{ip}"* ]]' for ip in sorted(resolved_ips)
            ]

            condition = " || ".join(ip_conditions)

            export_lines = []
            export_lines.append(f'INSTALL_LIB="$(expand_path "{install_lib}")"')
            export_lines.append(
                'export LD_LIBRARY_PATH="$INSTALL_LIB:${LD_LIBRARY_PATH:-}"'
            )

            for k, v in merged_env.items():
                if k in PATH_LIKE:
                    v = f"{v}:${{{k}:-}}"

                export_lines.append(f'export {k}="{v}"')

            export_lines.append(f'ARCH="{n_type}"')
            export_lines.append(f'NODE_COMMON_DIR="$(expand_path "{node_common_dir}")"')
            export_lines.append(
                'if grep -Eq "project[[:space:]]*\\([[:space:]]*InfiniCCL([[:space:]]|\\))" "$NODE_COMMON_DIR/CMakeLists.txt" 2>/dev/null; then'
            )
            export_lines.append('    BIN_SUB="examples/$1"')
            export_lines.append("else")
            export_lines.append('    BIN_SUB="$1"')
            export_lines.append("fi")

            body = "\n".join(f"    {line}" for line in export_lines)

            keyword = "if" if first else "elif"
            first = False

            blocks.append(f"{keyword} {condition}; then\n{body}\n")

        script_lines = [
            "#!/bin/bash",
            "",
            "expand_path() {",
            '    case "$1" in',
            '        "~") printf "%s\\n" "$HOME" ;;',
            '        "~/"*) printf "%s\\n" "$HOME/${1#~/}" ;;',
            '        *) printf "%s\\n" "$1" ;;',
            "    esac",
            "}",
            "",
            'HOST_IPS="$(hostname -I)"',
            "",
            "".join(blocks).rstrip(),
            "else",
            '    echo "[ERROR] Unknown host:"',
            '    echo "HOSTNAME=$(hostname)"',
            '    echo "HOST_IPS=$HOST_IPS"',
            "    exit 1",
            "fi",
            "",
            'EXE="$NODE_COMMON_DIR/build/$ARCH/$BIN_SUB"',
            'if [[ ! -x "$EXE" ]]; then',
            '    echo "[ERROR] Executable not found or not executable: $EXE"',
            "    exit 1",
            "fi",
            "",
            "shift",
            "",
            'exec "$EXE" "$@"',
            "",
        ]

        content = "\n".join(script_lines)

        with open(wrapper_path, "w") as f:
            f.write(content)

        os.chmod(wrapper_path, 0o755)

        return wrapper_path

    def launch(self, launcher_type, executable, args, launcher_obj):
        if launcher_type == "ompi":
            from backends.ompi import OmpiBackend

            backend = OmpiBackend()
            cmd, env = backend.get_launch_command(
                self.config, executable, args, launcher_obj
            )

        elif launcher_type == "mpich":
            from backends.mpich import MpichBackend

            backend = MpichBackend()
            cmd, env = backend.get_launch_command(
                self.config, executable, args, launcher_obj
            )

        elif launcher_type == "none":
            launcher_script = self.config.get("launcher_script")
            if not launcher_script:
                launcher_script = launcher_obj.ensure_launcher_exists()

            cmd, env = [launcher_script, executable] + list(args), os.environ.copy()

        else:
            print(f"Error: Unsupported launcher environment '{launcher_type}'")
            sys.exit(1)

        subprocess.run(cmd, env=env, check=True)
