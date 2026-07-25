import importlib.util
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "scripts/build_infini_stack.py"
SPEC = importlib.util.spec_from_file_location("build_infini_stack", MODULE_PATH)
build_infini_stack = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_infini_stack)


class BuildInfiniStackTest(unittest.TestCase):
    def _patch_main_dependencies(self, stack, args):
        stack.enter_context(
            mock.patch.object(build_infini_stack.sys, "platform", "linux")
        )
        stack.enter_context(
            mock.patch.object(build_infini_stack, "parse_args", return_value=args)
        )
        stack.enter_context(
            mock.patch.object(
                build_infini_stack,
                "validate_submodule",
                return_value="submodule-sha",
            )
        )
        stack.enter_context(
            mock.patch.object(
                build_infini_stack,
                "git_capture",
                side_effect=["infinilm-sha", "core-sha"],
            )
        )
        for helper, command in (
            ("build_infinirt_commands", "build-infinirt"),
            ("build_infiniops_commands", "build-ops"),
            ("build_infiniccl_commands", "build-ccl"),
        ):
            stack.enter_context(
                mock.patch.object(
                    build_infini_stack,
                    helper,
                    return_value=[[command]],
                )
            )
        run = stack.enter_context(mock.patch.object(build_infini_stack, "run"))
        write_manifest = stack.enter_context(
            mock.patch.object(build_infini_stack, "write_manifest")
        )
        return run, write_manifest

    def test_operator_set_is_stable(self):
        operators = build_infini_stack.read_operator_set()

        self.assertEqual(len(operators), 23)
        self.assertEqual(operators, sorted(set(operators)))
        self.assertIn("paged_attention_infinilm", operators)
        self.assertIn("rotary_embedding_infinilm", operators)

    def test_invalid_operator_set_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ops.txt"
            path.write_text("rms_norm\nadd\nadd\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicates"):
                build_infini_stack.read_operator_set(path)

    def test_parse_gitlink(self):
        output = "160000 commit abcdef1234567890\tsubmodules/InfiniRT\n"

        self.assertEqual(
            build_infini_stack.parse_gitlink(output, Path("submodules/InfiniRT")),
            "abcdef1234567890",
        )

    def test_uninitialized_submodule_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            core_root = Path(directory)

            with self.assertRaisesRegex(RuntimeError, "Submodule is not initialized"):
                build_infini_stack.validate_submodule(
                    core_root, Path("submodules/InfiniRT")
                )

    def test_gitlink_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            core_root = Path(directory)
            relative_path = Path("submodules/InfiniRT")
            source_root = core_root / relative_path
            source_root.mkdir(parents=True)
            (source_root / "CMakeLists.txt").touch()
            with mock.patch.object(
                build_infini_stack,
                "git_capture",
                side_effect=[
                    "160000 commit expected-sha\tsubmodules/InfiniRT",
                    "actual-sha",
                ],
            ):
                with self.assertRaisesRegex(RuntimeError, "revision mismatch"):
                    build_infini_stack.validate_submodule(core_root, relative_path)

    def test_dirty_submodule_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            core_root = Path(directory)
            relative_path = Path("submodules/InfiniRT")
            source_root = core_root / relative_path
            source_root.mkdir(parents=True)
            (source_root / "CMakeLists.txt").touch()
            with mock.patch.object(
                build_infini_stack,
                "git_capture",
                side_effect=[
                    "160000 commit expected-sha\tsubmodules/InfiniRT",
                    "expected-sha",
                    " M CMakeLists.txt",
                ],
            ) as git_capture:
                with self.assertRaisesRegex(RuntimeError, "worktree is dirty") as error:
                    build_infini_stack.validate_submodule(core_root, relative_path)

            self.assertIn(str(relative_path), str(error.exception))
            self.assertEqual(
                git_capture.call_args_list[-1].args,
                (
                    ["status", "--porcelain"],
                    source_root,
                ),
            )

    def test_untracked_submodule_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            core_root = Path(directory)
            relative_path = Path("submodules/InfiniRT")
            source_root = core_root / relative_path
            source_root.mkdir(parents=True)
            (source_root / "CMakeLists.txt").touch()

            def git_capture(arguments, _):
                if arguments[0] == "ls-tree":
                    return "160000 commit expected-sha\tsubmodules/InfiniRT"
                if arguments[0] == "rev-parse":
                    return "expected-sha"
                if arguments == ["status", "--porcelain"]:
                    return "?? src/extra.cc"
                return ""

            with mock.patch.object(
                build_infini_stack,
                "git_capture",
                side_effect=git_capture,
            ):
                with self.assertRaisesRegex(RuntimeError, "worktree is dirty"):
                    build_infini_stack.validate_submodule(core_root, relative_path)

    def test_capture_includes_command_error(self):
        with self.assertRaisesRegex(RuntimeError, "expected failure"):
            build_infini_stack.capture(
                [
                    build_infini_stack.sys.executable,
                    "-c",
                    "import sys; print('expected failure', file=sys.stderr); sys.exit(1)",
                ],
                PROJECT_ROOT,
            )

    def test_git_capture_scopes_safe_directory(self):
        with mock.patch.object(
            build_infini_stack, "capture", return_value="revision"
        ) as capture:
            revision = build_infini_stack.git_capture(
                ["rev-parse", "HEAD"], PROJECT_ROOT
            )

        self.assertEqual(revision, "revision")
        command = capture.call_args.args[0]
        self.assertEqual(command[:2], ["git", "-c"])
        self.assertEqual(command[2], f"safe.directory={PROJECT_ROOT}")

    def test_infinicore_root_is_required(self):
        with (
            mock.patch.object(build_infini_stack.sys, "stderr"),
            self.assertRaises(SystemExit),
        ):
            build_infini_stack.parse_args([])

    def test_cuda_arch_list_is_converted_for_cmake(self):
        commands = build_infini_stack.build_infinirt_commands(
            Path("rt"),
            Path("build/rt"),
            Path("build/prefix"),
            "Release",
            8,
            "sm_80,sm_86,sm_90a",
            False,
        )

        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=80;86;90a", commands[0])

    def test_invalid_cuda_arch_is_rejected_by_argparse(self):
        for cuda_arch in ("", "sm80", "sm_80,", "sm_80,,sm_90", "sm_80,sm_80"):
            with self.subTest(cuda_arch=cuda_arch):
                with (
                    mock.patch.object(build_infini_stack.sys, "stderr"),
                    self.assertRaises(SystemExit),
                ):
                    build_infini_stack.parse_args(
                        [
                            "--infinicore-root",
                            "core",
                            "--cuda-arch",
                            cuda_arch,
                        ]
                    )

    def test_infinirt_commands_share_one_prefix(self):
        commands = build_infini_stack.build_infinirt_commands(
            Path("rt"),
            Path("build/rt"),
            Path("build/prefix"),
            "Release",
            8,
            "sm_80",
            True,
        )

        self.assertIn(f"-DCMAKE_INSTALL_PREFIX={Path('build/prefix')}", commands[0])
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=80", commands[0])
        self.assertEqual(commands[2][0], "ctest")
        self.assertEqual(commands[-1], ["cmake", "--install", str(Path("build/rt"))])

    def test_infiniops_commands_use_pinned_runtime_and_operator_set(self):
        commands = build_infini_stack.build_infiniops_commands(
            Path("ops"),
            Path("build/ops"),
            Path("build/prefix"),
            "Release",
            8,
            "sm_80",
            ["add", "rms_norm"],
        )

        configure = commands[0]
        self.assertIn(f"-DINFINI_RT_ROOT={Path('build/prefix')}", configure)
        self.assertIn("-DINFINI_OPS_OPS=add,rms_norm", configure)
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=80", configure)
        self.assertEqual(commands[1][3:5], ["--target", "infiniops"])
        self.assertEqual(commands[-1], ["cmake", "--install", str(Path("build/ops"))])

    def test_infiniccl_commands_run_two_gpu_smoke_test(self):
        commands = build_infini_stack.build_infiniccl_commands(
            Path("ccl"),
            Path("build/ccl"),
            Path("build/prefix"),
            "Release",
            8,
            "sm_80",
            True,
        )

        configure = commands[0]
        self.assertIn("-DWITH_NVIDIA=ON", configure)
        self.assertIn("-DWITH_NCCL=ON", configure)
        self.assertIn("-DBUILD_EXAMPLES=ON", configure)
        self.assertEqual(
            commands[-1][0], str(Path("build/ccl/examples/ccl/all_reduce"))
        )
        self.assertEqual(commands[-1][1:3], ["-g", "2"])

    def test_main_uses_core_sources_infini_lm_cwd_and_one_prefix(self):
        args = build_infini_stack.parse_args(
            [
                "--infinicore-root",
                "core",
                "--build-root",
                "relative-build",
                "--dry-run",
                "--jobs",
                "1",
            ]
        )
        core_root = (PROJECT_ROOT / "core").resolve()
        build_root = (PROJECT_ROOT / "relative-build").resolve()
        prefix = build_root / "prefix"
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.dict(build_infini_stack.os.environ, {}, clear=True)
            )
            stack.enter_context(
                mock.patch.object(build_infini_stack, "parse_args", return_value=args)
            )
            validate_submodule = stack.enter_context(
                mock.patch.object(
                    build_infini_stack,
                    "validate_submodule",
                    return_value="submodule-sha",
                )
            )
            stack.enter_context(
                mock.patch.object(
                    build_infini_stack,
                    "git_capture",
                    side_effect=["infinilm-sha", "core-sha"],
                )
            )
            build_infinirt = stack.enter_context(
                mock.patch.object(
                    build_infini_stack,
                    "build_infinirt_commands",
                    return_value=[["build-infinirt"]],
                )
            )
            build_infiniops = stack.enter_context(
                mock.patch.object(
                    build_infini_stack,
                    "build_infiniops_commands",
                    return_value=[["build-ops"]],
                )
            )
            build_infiniccl = stack.enter_context(
                mock.patch.object(
                    build_infini_stack,
                    "build_infiniccl_commands",
                    return_value=[["build-ccl"]],
                )
            )
            stack.enter_context(mock.patch.object(build_infini_stack, "write_manifest"))
            run = stack.enter_context(mock.patch.object(build_infini_stack, "run"))
            build_infini_stack.main([])

        self.assertEqual(
            validate_submodule.call_args_list[0].args,
            (
                core_root,
                Path("submodules/InfiniRT"),
            ),
        )
        self.assertEqual(
            build_infinirt.call_args.args[:3],
            (
                core_root / "submodules/InfiniRT",
                build_root / "infinirt",
                prefix,
            ),
        )
        self.assertEqual(
            build_infiniops.call_args.args[:3],
            (
                core_root / "submodules/InfiniOps",
                build_root / "infiniops",
                prefix,
            ),
        )
        self.assertEqual(
            build_infiniccl.call_args.args[:3],
            (
                core_root / "submodules/InfiniCCL",
                build_root / "infiniccl",
                prefix,
            ),
        )
        self.assertTrue(
            all(call.args[1] == PROJECT_ROOT for call in run.call_args_list)
        )
        infinirt_env = run.call_args_list[0].args[2]
        infiniops_env = run.call_args_list[1].args[2]
        infiniccl_env = run.call_args_list[2].args[2]
        self.assertNotIn("LD_LIBRARY_PATH", infinirt_env)
        self.assertTrue(infiniops_env["LD_LIBRARY_PATH"].startswith(str(prefix)))
        self.assertEqual(infiniops_env, infiniccl_env)

    def test_manifest_is_written_after_all_commands_succeed(self):
        args = build_infini_stack.parse_args(
            ["--infinicore-root", "core", "--jobs", "1"]
        )
        events = []
        with ExitStack() as stack:
            run, write_manifest = self._patch_main_dependencies(stack, args)
            run.side_effect = lambda command, *_: events.append(command[0])
            write_manifest.side_effect = lambda *_: events.append("write-manifest")
            build_infini_stack.main([])

        self.assertEqual(
            events,
            ["build-infinirt", "build-ops", "build-ccl", "write-manifest"],
        )

    def test_failed_command_does_not_write_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            build_root = Path(directory)
            manifest_path = build_root / "manifest.json"
            staging_path = build_root / "manifest.json.tmp"
            manifest_path.write_text("old manifest\n", encoding="utf-8")
            staging_path.write_text("partial manifest\n", encoding="utf-8")
            args = build_infini_stack.parse_args(
                [
                    "--infinicore-root",
                    "core",
                    "--build-root",
                    str(build_root),
                    "--jobs",
                    "1",
                ]
            )

            def fail_run(*_):
                self.assertFalse(manifest_path.exists())
                self.assertFalse(staging_path.exists())
                raise RuntimeError("build failed")

            with ExitStack() as stack:
                run, write_manifest = self._patch_main_dependencies(stack, args)
                run.side_effect = fail_run
                with self.assertRaisesRegex(RuntimeError, "build failed"):
                    build_infini_stack.main([])

            write_manifest.assert_not_called()
            self.assertFalse(manifest_path.exists())
            self.assertFalse(staging_path.exists())

    def test_manifest_records_revisions_operators_and_options(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            staging_path = path.with_name(f"{path.name}.tmp")
            path.write_text('{"old": true}\n', encoding="utf-8")
            staging_path.write_text("partial manifest\n", encoding="utf-8")
            build_infini_stack.write_manifest(
                path,
                "infinilm-sha",
                "core-sha",
                {
                    "InfiniRT": "rt-sha",
                    "InfiniOps": "ops-sha",
                    "InfiniCCL": "ccl-sha",
                },
                ["add", "rms_norm"],
                Path("prefix"),
                "Release",
                "sm_80,sm_90a",
                8,
                True,
            )

            manifest = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["backend"], "nvidia")
            self.assertEqual(manifest["infinilm"], "infinilm-sha")
            self.assertEqual(manifest["infinicore"], "core-sha")
            self.assertEqual(manifest["submodules"]["InfiniRT"], "rt-sha")
            self.assertEqual(manifest["submodules"]["InfiniCCL"], "ccl-sha")
            self.assertEqual(manifest["operators"], ["add", "rms_norm"])
            self.assertEqual(manifest["install_prefix"], str(Path("prefix")))
            self.assertEqual(manifest["build_type"], "Release")
            self.assertEqual(manifest["cuda_arch"], "sm_80,sm_90a")
            self.assertEqual(manifest["jobs"], 8)
            self.assertTrue(manifest["test"])
            self.assertNotIn("old", manifest)
            self.assertFalse(staging_path.exists())


if __name__ == "__main__":
    unittest.main()
