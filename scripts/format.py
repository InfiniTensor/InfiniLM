import argparse
import os
import subprocess
from pathlib import Path

from colorama import Fore, Style

# Supported file types and their corresponding formatter categories
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
        return True  # Unsupported file type, skip

    formatted = True
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
                    formatted = False
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
                    check=False,
                )
                if process.returncode != 0:
                    print(f"{Fore.YELLOW}{file} is not formatted.{Style.RESET_ALL}")
                    print(
                        f"Use {Fore.CYAN}{formatter} {file}{Style.RESET_ALL} to format it."
                    )
                    formatted = False
            else:
                subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                print(f"{Fore.CYAN}Formatted: {file}{Style.RESET_ALL}")
        elif formatter == "ruff":
            ruff_cmd = ["ruff", "check", file]
            ruff_format_cmd = ["ruff", "format", file]
            try:
                if check:
                    process = subprocess.run(
                        ruff_cmd,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if process.returncode != 0:
                        print(f"{Fore.YELLOW}{file} has ruff issues.{Style.RESET_ALL}")
                        print(
                            f"Use {Fore.CYAN}ruff check --fix {file}{Style.RESET_ALL} to fix it."
                        )
                        formatted = False

                    ruff_format_cmd.insert(2, "--check")
                    process = subprocess.run(
                        ruff_format_cmd,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if process.returncode != 0:
                        print(
                            f"{Fore.YELLOW}{file} is not ruff-formatted.{Style.RESET_ALL}"
                        )
                        print(
                            f"Use {Fore.CYAN}ruff format {file}{Style.RESET_ALL} to format it."
                        )
                        formatted = False
                else:
                    ruff_cmd.insert(2, "--fix")
                    process = subprocess.run(
                        ruff_cmd,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if process.returncode != 0:
                        print(f"{Fore.RED}Ruff failed for {file}.{Style.RESET_ALL}")
                        if process.stdout:
                            print(process.stdout, end="")
                        if process.stderr:
                            print(process.stderr, end="")
                        formatted = False
                    else:
                        print(f"{Fore.CYAN}Ruff fixed: {file}{Style.RESET_ALL}")

                    process = subprocess.run(
                        ruff_format_cmd,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if process.returncode != 0:
                        print(
                            f"{Fore.RED}Ruff format failed for {file}.{Style.RESET_ALL}"
                        )
                        if process.stdout:
                            print(process.stdout, end="")
                        if process.stderr:
                            print(process.stderr, end="")
                        formatted = False
                    else:
                        print(f"{Fore.CYAN}Ruff formatted: {file}{Style.RESET_ALL}")
            except FileNotFoundError:
                print(
                    f"{Fore.RED}Formatter ruff not found, {file} skipped.{Style.RESET_ALL}"
                )
                formatted = False
    except FileNotFoundError:
        print(
            f"{Fore.RED}Formatter {formatter} not found, {file} skipped.{Style.RESET_ALL}"
        )
        formatted = False
    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}Formatter {formatter} failed: {e}{Style.RESET_ALL}")
        formatted = False

    if not ensure_single_trailing_newline(file, check):
        formatted = False

    return formatted


def ensure_single_trailing_newline(file: Path, check: bool) -> bool:
    """Ensure the file ends with exactly one LF byte."""
    content = file.read_bytes()
    fixed = content.rstrip(b"\r\n") + b"\n"
    if content == fixed:
        return True

    if check:
        print(
            f"{Fore.YELLOW}{file} does not end with exactly one newline.{Style.RESET_ALL}"
        )
        return False

    file.write_bytes(fixed)
    print(f"{Fore.CYAN}Fixed trailing newline: {file}{Style.RESET_ALL}")
    return True


def git_added_files():
    """Get all staged files"""
    try:
        # Use git diff --cached --name-only to get all files added to staging area
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
    """Get list of files modified from the specified Git reference to the current state"""
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
    """Recursively get all files under the specified paths"""
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
    """Check if file is within the specified paths"""
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
    parser.add_argument("--py", default="ruff", help="Python formatter (default: ruff)")
    args = parser.parse_args()

    if args.ref is None and args.path is None:
        # Last commit.
        print(f"{Fore.GREEN}Formatting git staged files.{Style.RESET_ALL}")
        files = git_added_files()

    else:
        if args.ref is None:
            print(f"{Fore.GREEN}Formatting files in {args.path}.{Style.RESET_ALL}")
            files = list_files(args.path)
        elif args.path is None:
            print(
                f"{Fore.GREEN}Formatting git modified files from {args.ref}.{Style.RESET_ALL}"
            )
            files = git_modified_since_ref(args.ref)
        else:
            print(
                f"{Fore.GREEN}Formatting git modified files from {args.ref} in {args.path}.{Style.RESET_ALL}"
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
