import argparse
import json
import pathlib


def _load_results(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a JSON array or an object with results")
    return data


def _params_key(params):
    return json.dumps(params or {}, sort_keys=True, separators=(",", ":"))


def _result_key(result):
    return (
        result.get("backend", ""),
        result.get("benchmark", ""),
        _params_key(result.get("params")),
        result.get("unit", ""),
    )


def _index_results(results):
    indexed = {}
    for result in results:
        indexed[_result_key(result)] = result
    return indexed


def _format_params(params):
    if not params:
        return ""
    return " ".join(f"{key}={params[key]}" for key in sorted(params))


def _delta_percent(baseline, candidate):
    if baseline == 0:
        return None
    return (candidate - baseline) / baseline * 100.0


def _format_delta(delta):
    if delta is None:
        return "n/a"
    return f"{delta:+.2f}%"


def _print_metric(name, baseline, candidate):
    delta = _delta_percent(baseline, candidate)
    print(f"  baseline {name}:  {baseline:.6g}")
    print(f"  candidate {name}: {candidate:.6g}")
    print(f"  {name} delta:     {_format_delta(delta)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=pathlib.Path)
    parser.add_argument("--candidate", required=True, type=pathlib.Path)
    args = parser.parse_args()

    baseline = _index_results(_load_results(args.baseline))
    candidate = _index_results(_load_results(args.candidate))
    common_keys = sorted(set(baseline) & set(candidate))
    missing_candidate = sorted(set(baseline) - set(candidate))
    new_candidate = sorted(set(candidate) - set(baseline))

    for key in common_keys:
        baseline_result = baseline[key]
        candidate_result = candidate[key]
        backend, benchmark, _, unit = key
        params = _format_params(baseline_result.get("params"))
        suffix = f" {params}" if params else ""
        print(f"{benchmark}{suffix} backend={backend} unit={unit}")
        _print_metric("mean", baseline_result["mean"], candidate_result["mean"])
        print()
        _print_metric("median", baseline_result["median"], candidate_result["median"])
        print()

    if missing_candidate:
        print("missing candidate results:")
        for backend, benchmark, params, unit in missing_candidate:
            print(f"  {benchmark} backend={backend} params={params} unit={unit}")

    if new_candidate:
        print("new candidate results:")
        for backend, benchmark, params, unit in new_candidate:
            print(f"  {benchmark} backend={backend} params={params} unit={unit}")


if __name__ == "__main__":
    main()
