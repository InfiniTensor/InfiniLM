#!/usr/bin/env python3
"""Run vLLM 0.9.2's official serving benchmark client.

This launcher imports only the serving benchmark modules, so the client does
not need to initialize a vLLM inference engine. It also supplies the generic
``--extra-body`` option that was added after vLLM 0.9.2.
"""

import argparse
import json

from vllm.benchmarks import endpoint_request_func
from vllm.benchmarks.serve import add_cli_args, main


def json_object(value: str) -> dict:
    """Parse a JSON object for ``--extra-body``."""
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--extra-body must be a JSON object")
    return parsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM serving benchmark client")
    add_cli_args(parser)
    parser.add_argument("--extra-body", type=json_object, default=None)
    args = parser.parse_args()

    if args.extra_body:
        original = endpoint_request_func.ASYNC_REQUEST_FUNCS["openai-chat"]

        async def openai_chat_with_extra_body(request_func_input, pbar=None):
            merged = dict(request_func_input.extra_body or {})
            merged.update(args.extra_body)
            request_func_input.extra_body = merged
            return await original(request_func_input, pbar)

        endpoint_request_func.ASYNC_REQUEST_FUNCS["openai-chat"] = (
            openai_chat_with_extra_body
        )

    main(args)
