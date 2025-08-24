import argparse
import subprocess
import os
from pathlib import Path
from colorama import Fore, Style

# 支持的文件类型
SUPPORTED_FILES = {
    ".h": "c",
    ".hh": "c",
    ".hpp": "c",
    ".c": "c",
    ".cc": "c",
    ".cpp": "c",
    ".cxx": "c",
    ".cu": "c",
    ".cuh": "c",
    ".mlu": "c",
    ".cl": "c",
    ".py": "py",
}


def format_file(file: Path, check: bool, formatter) -> bool:
    formatter = formatter.get(SUPPORTED_FILES.get(file.suffix, None), None)
    if not formatter:
        return True  # 文件类型不支持，跳过

    try:
        cmd = []
        if formatter.startswith("clang-format"):
            cmd = [formatter, "-style=file", "-i", file]
            if check:
                cmd.insert(2, "-dry-run")
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                if process.stderr:
                    print(f"{Fore.YELLOW}{file} is not formatted.{Style.RESET_ALL}")
                    print(
                        f"Use {Fore.CYAN}{formatter} -style=file -i {file}{Style.RESET_ALL} to format it."
                    )
                    return False
            else:
                subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                print(f"{Fore.CYAN}Formatted: {file}{Style.RESET_ALL}")
        elif formatter == "black":
            cmd = [formatter, file]
            if check:
                cmd.insert(1, "--check")
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                if process.returncode != 0:
                    print(f"{Fore.YELLOW}{file} is not formatted.{Style.RESET_ALL}")
                    print(
                        f"Use {Fore.CYAN}{formatter} {file}{Style.RESET_ALL} to format it."
                    )
                    return False
            else:
                subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                print(f"{Fore.CYAN}Formatted: {file}{Style.RESET_ALL}")
    except FileNotFoundError:
        print(
            f"{Fore.RED}Formatter {formatter} not found, {file} skipped.{Style.RESET_ALL}"
        )
    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}Formatter {formatter} failed: {e}{Style.RESET_ALL}")

    return True


def git_added_files():
    """获取所有已暂存更改的文件"""
    try:
        # 使用 git diff --cached --name-only 获取所有已添加到暂存区的文件
        result = subprocess.run(
            ["git", "diff", "--cached", "--diff-filter=AMR", "--name-only"],
            capture_output=True,
            text=True,
            check=True,
        )
        for file in result.stdout.splitlines():
            yield Path(file.strip())
    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}Git diff failed: {e}{Style.RESET_ALL}")


def git_modified_since_ref(ref):
    """获取从指定的 Git 引用到当前状态的修改文件列表"""
    try:
        result = subprocess.run(
            ["git", "diff", f"{ref}..", "--diff-filter=AMR", "--name-only"],
            capture_output=True,
            text=True,
            check=True,
        )
        for file in result.stdout.splitlines():
            yield Path(file.strip())
    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}Git diff failed: {e}{Style.RESET_ALL}")


def list_files(paths):
    """递归获取指定路径下的所有文件"""
    files = []
    for path in paths:
        if path.is_file():
            yield path
        elif path.is_dir():
            for dirpath, _, filenames in os.walk(path):
                for name in filenames:
                    yield Path(dirpath) / name
        else:
            print(
                f"{Fore.RED}Error: {path} is not a file or directory.{Style.RESET_ALL}"
            )


def filter_in_path(file: Path, path) -> bool:
    """判断文件是否在指定路径下"""
    for p in path:
        if file.is_relative_to(p):
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref", type=str, help="Git reference (commit hash) to compare against."
    )
    parser.add_argument(
        "--path", nargs="*", type=Path, help="Files to format or check."
    )
    parser.add_argument(
        "--check", action="store_true", help="Check files without modifying them."
    )
    parser.add_argument(
        "--c", default="clang-format-16", help="C formatter (default: clang-format-16)"
    )
    parser.add_argument(
        "--py", default="black", help="Python formatter (default: black)"
    )
    args = parser.parse_args()

    if args.ref is None and args.path is None:
        # Last commit.
        print(f"{Fore.GREEN}Formating git added files.{Style.RESET_ALL}")
        files = git_added_files()

    else:
        if args.ref is None:
            print(f"{Fore.GREEN}Formating files in {args.path}.{Style.RESET_ALL}")
            files = list_files(args.path)
        elif args.path is None:
            print(
                f"{Fore.GREEN}Formating git modified files from {args.ref}.{Style.RESET_ALL}"
            )
            files = git_modified_since_ref(args.ref)
        else:
            print(
                f"{Fore.GREEN}Formating git modified files from {args.ref} in {args.path}.{Style.RESET_ALL}"
            )
            files = (
                file
                for file in git_modified_since_ref(args.ref)
                if filter_in_path(file, args.path)
            )

    formatted = True
    for file in files:
        if not format_file(
            file,
            args.check,
            {
                "c": args.c,
                "py": args.py,
            },
        ):
            formatted = False

    if not formatted:
        exit(1)


if __name__ == "__main__":
    main()
