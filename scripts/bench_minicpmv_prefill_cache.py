import argparse
import datetime
import subprocess
import sys
from pathlib import Path


def run_and_log(command, log_file, cwd, check=False):
    printable_command = " ".join(command)
    write_line(log_file, f"$ {printable_command}")
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError:
        write_line(log_file, f"Command not found: {command[0]}")
        if check:
            raise
        return 127

    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        log_file.write(line)
    return_code = process.wait()
    if check and return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return return_code


def write_line(log_file, message=""):
    print(message)
    log_file.write(f"{message}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the MiniCPM-V multimodal benchmark and save a timestamped log. "
            "Start an InfiniLM OpenAI-compatible server before running this script."
        )
    )
    parser.add_argument(
        "--api-url",
        default="127.0.0.1:8000",
        help="Server address without scheme.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model name sent to the OpenAI API.",
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing .jpg or .jpeg images.",
    )
    parser.add_argument(
        "--mm-port",
        default=None,
        help="Optional local image HTTP server port for scripts/test_perf.py.",
    )
    parser.add_argument(
        "--log-dir",
        default="logs/minicpmv-prefill-cache",
        help="Output directory for benchmark logs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-request details from scripts/test_perf.py.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    image_dir = Path(args.image_dir).expanduser().resolve()
    if not image_dir.is_dir():
        raise ValueError(f"Not a valid image directory: {image_dir}")

    log_dir = repo_root / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"minicpmv-prefill-cache-{timestamp}.log"

    command = [
        sys.executable,
        "scripts/test_perf.py",
        "--api-url",
        args.api_url,
        "--model",
        args.model,
        "--image-dir",
        str(image_dir),
    ]
    if args.mm_port is not None:
        command.extend(["--mm-port", args.mm_port])
    if args.verbose:
        command.append("--verbose")

    with log_path.open("w", encoding="utf-8") as log_file:
        write_line(log_file, "=== MiniCPM-V prefill packing cache benchmark ===")
        write_line(log_file, f"Timestamp: {datetime.datetime.now().isoformat()}")
        write_line(log_file, f"Repository: {repo_root}")
        run_and_log(["git", "rev-parse", "HEAD"], log_file, repo_root)
        run_and_log(["git", "status", "--short"], log_file, repo_root)
        run_and_log([sys.executable, "--version"], log_file, repo_root)
        run_and_log(["nvidia-smi"], log_file, repo_root)
        write_line(log_file, f"API URL: {args.api_url}")
        write_line(log_file, f"Model: {args.model}")
        write_line(log_file, f"Image dir: {image_dir}")
        write_line(log_file, f"MM port: {args.mm_port}")
        write_line(log_file)
        return_code = run_and_log(command, log_file, repo_root)

    print(f"Log saved to {log_path}")
    return return_code


if __name__ == "__main__":
    sys.exit(main())
