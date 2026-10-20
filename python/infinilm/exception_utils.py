import logging
import os
import sys
from types import ModuleType
from typing import Iterator

logger = logging.getLogger(__name__)


def _iter_exception_chain(
    e: BaseException, *, max_depth: int = 6
) -> Iterator[BaseException]:
    """Iterate through exception chain with depth limit."""
    cur: BaseException | None = e
    depth = 0
    seen: set[int] = set()
    while cur is not None and depth < max_depth:
        cur_id = id(cur)
        if cur_id in seen:
            break
        seen.add(cur_id)
        yield cur
        depth += 1
        cur = cur.__cause__ or cur.__context__


def is_oom_exception(e: BaseException) -> bool:
    """
    Conservative OOM detector for MetaX allocator failures and CUDA/PyTorch OOMs.
    Checks exception type (when available) and message substrings across chained exceptions.
    """
    # Inspect an already-loaded Torch module without importing it on the
    # production error path. An absent or malformed type is not an OOM signal.
    torch_module = sys.modules.get("torch")
    oom_type = (
        vars(torch_module).get("OutOfMemoryError")
        if isinstance(torch_module, ModuleType)
        else None
    )
    if isinstance(oom_type, type) and issubclass(oom_type, BaseException):
        if any(isinstance(ex, oom_type) for ex in _iter_exception_chain(e)):
            return True

    # Common patterns observed for allocator failures.
    # Keep this allowlist small to avoid hard-exiting on unrelated errors.
    patterns = (
        # MetaX / infinirt allocator
        "hcmalloc",
        "out of memory",
        # CUDA / driver / runtime alloc failures
        "cuda out of memory",
        "cumemalloc",
        "cublas_status_alloc_failed",
        "cudnn_status_alloc_failed",
    )

    for ex in _iter_exception_chain(e):
        msg = str(ex)
        if not msg:
            continue
        msg_l = msg.lower()
        if any(p in msg_l for p in patterns):
            return True
    return False


def handle_oom_and_exit(e: BaseException, exit_code: int = 137) -> None:
    """Handle OOM exception by logging and exiting."""
    if is_oom_exception(e):
        logger.error(
            "OOM-like exception: exiting worker with code %d: %r",
            exit_code,
            e,
            exc_info=False,
        )
        os._exit(exit_code)
