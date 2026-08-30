"""Measure /predict latency under concurrent load.

Answers the question PRODUCTION_READINESS.md item 11 leaves open: what are the
p50/p95/p99 latencies as concurrency rises, and where does throughput flatten?

Uses only the standard library so it adds no dependency. Each worker keeps one
persistent connection and fires requests back-to-back; a level's throughput is
total completed requests over wall time.

    uvicorn single_prediction.api:app --port 8000        # terminal 1
    python scripts/load_test.py                          # terminal 2
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

PAYLOAD = json.dumps({
    "user_id": "user_000001",
    "video_id": "video_0000001",
    "watch_time": 45.0,
    "hour_of_day": 14,
}).encode()


def worker(url: str, stop_at: float, latencies: list[float], errors: list[int]) -> None:
    while time.perf_counter() < stop_at:
        request = urllib.request.Request(
            url, data=PAYLOAD, headers={"Content-Type": "application/json"}
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                response.read()
                if response.status != 200:
                    errors.append(response.status)
                    continue
        except (urllib.error.URLError, OSError):
            errors.append(0)
            continue
        latencies.append((time.perf_counter() - started) * 1000)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def run_level(url: str, concurrency: int, seconds: float) -> dict:
    latencies: list[float] = []
    errors: list[int] = []
    stop_at = time.perf_counter() + seconds
    started = time.perf_counter()
    threads = [
        threading.Thread(target=worker, args=(url, stop_at, latencies, errors))
        for _ in range(concurrency)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - started
    if not latencies:
        return {"concurrency": concurrency, "requests": 0, "errors": len(errors)}
    return {
        "concurrency": concurrency,
        "requests": len(latencies),
        "errors": len(errors),
        "rps": round(len(latencies) / elapsed, 1),
        "p50_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(percentile(latencies, 0.95), 2),
        "p99_ms": round(percentile(latencies, 0.99), 2),
        "max_ms": round(max(latencies), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Load-test the /predict route")
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--levels", default="1,4,16,64",
                        help="comma-separated concurrency levels")
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="duration per level")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    health = args.url.rsplit("/", 1)[0] + "/health"
    try:
        with urllib.request.urlopen(health, timeout=5) as response:
            if json.load(response).get("status") != "healthy":
                print("API is up but not healthy; train a model first", file=sys.stderr)
                return 1
    except (urllib.error.URLError, OSError) as exc:
        print(f"cannot reach {health}: {exc}", file=sys.stderr)
        return 1

    levels = [int(level) for level in args.levels.split(",")]
    run_level(args.url, 2, 2.0)  # warm-up, discarded
    results = [run_level(args.url, level, args.seconds) for level in levels]

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"{'conc':>5} {'reqs':>7} {'err':>4} {'rps':>8} "
              f"{'p50':>8} {'p95':>8} {'p99':>8} {'max':>9}")
        for row in results:
            print(f"{row['concurrency']:>5} {row['requests']:>7} {row['errors']:>4} "
                  f"{row.get('rps', 0):>8} {row.get('p50_ms', '-'):>8} "
                  f"{row.get('p95_ms', '-'):>8} {row.get('p99_ms', '-'):>8} "
                  f"{row.get('max_ms', '-'):>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
