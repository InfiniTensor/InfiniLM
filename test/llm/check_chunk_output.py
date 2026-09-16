"""Native output suppression plus the existing per-rank KV lifecycle checks."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from check_chunk_tp import run


def check(args):
    from infinilm.lib import _infinilm

    # Fail before loading a model if the native extension has not been rebuilt.
    assert _infinilm.InferEngine.Input(prefill_only=True).prefill_only
    native = _infinilm.InferEngine.forward
    calls = []
    rejected = []

    def forward(engine, inputs):
        if not calls:
            # Rejection must happen before any worker job is submitted; the
            # subsequent normal lifecycle must still be able to use this engine.
            for kwargs, message in (
                (
                    {"prefill_only": True, "sample_all_positions": True},
                    "sample_all_positions=false",
                ),
                ({"prefill_only": True}, "input_offsets"),
            ):
                try:
                    native(engine, _infinilm.InferEngine.Input(**kwargs))
                except ValueError as error:
                    assert message in str(error)
                    rejected.append(message)
                else:
                    raise AssertionError("invalid outputless forward was accepted")
        output = native(engine, inputs)
        if inputs.prefill_only:
            assert not output.output_ids
            assert not output.logits
            assert not output.hidden_states
        else:
            assert output.output_ids
            assert output.logits
        calls.append(dict(prefill_only=inputs.prefill_only))
        return output

    with patch.object(_infinilm.InferEngine, "forward", forward):
        result = run(args)
    skipped = sum(c["prefill_only"] for c in calls)
    assert skipped > 0
    result["native_rejections"] = rejected
    assert len(rejected) == 2
    result["native_outputs"] = dict(
        calls=len(calls), prefill_only=skipped, checked=True
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--cache-off", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(graph=False, pp=1, stage=0, port=29761, policy="lru")
    args = parser.parse_args()
    try:
        payload = check(args)
    except Exception as error:
        args.output.write_text(
            json.dumps(dict(status="failure", error=repr(error)), indent=2) + "\n"
        )
        raise
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["native_outputs"]))
