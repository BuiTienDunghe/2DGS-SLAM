"""Bounded PyTorch CUDA-sharing matrix.

This deliberately exercises the same ``spawn`` transport used by the stock
pipeline, while keeping CPU controls and direct/Queue transports separate.
The child sends an acknowledgement only after it has copied every value to
CPU and compared it byte-for-byte with the deterministic expected tensor.
Each case is independent and the producer tensor is held until the child has
exited, which avoids hiding lifetime errors behind producer cleanup.

Exit codes:
  0: every requested case passed
  1: at least one case failed
  2: CUDA/Torch environment could not be initialized
  3: a child exceeded the bounded timeout
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any


SMALL_VALUES = 32
LARGE_BYTES = 16 * 1024 * 1024
LARGE_VALUES = LARGE_BYTES // 4


@dataclass(frozen=True)
class CaseSpec:
    name: str
    transport: str
    device_kind: str
    values: int


CASE_SPECS = (
    CaseSpec("cpu-direct-small", "direct", "cpu", SMALL_VALUES),
    CaseSpec("cpu-queue-small", "queue", "cpu", SMALL_VALUES),
    CaseSpec("cpu-direct-large", "direct", "cpu", LARGE_VALUES),
    CaseSpec("cpu-queue-large", "queue", "cpu", LARGE_VALUES),
    CaseSpec("cuda-direct-small", "direct", "cuda", SMALL_VALUES),
    CaseSpec("cuda-queue-small", "queue", "cuda", SMALL_VALUES),
    CaseSpec("cuda-direct-large", "direct", "cuda", LARGE_VALUES),
    CaseSpec("cuda-queue-large", "queue", "cuda", LARGE_VALUES),
)


def _expected(values: int):
    import torch

    return torch.arange(values, dtype=torch.int32)


def _child_direct(tensor: Any, values: int, ack: Any) -> None:
    _child_check(tensor, values, ack, "direct")


def _child_queue(work_queue: Any, values: int, ack: Any, get_timeout: float) -> None:
    try:
        tensor = work_queue.get(timeout=get_timeout)
    except queue.Empty:
        ack.send(
            {
                "ok": False,
                "child_values_exact": False,
                "child_values_exact_checked": False,
                "error": "queue.get timed out",
                "failure_api": "multiprocessing.Queue.get",
            }
        )
        return
    _child_check(tensor, values, ack, "queue")


def _child_check(tensor: Any, values: int, ack: Any, transport: str) -> None:
    try:
        import torch

        if tensor.is_cuda:
            # This synchronizes the imported allocation and makes the exact
            # host-side comparison independent of asynchronous kernels.
            torch.cuda.synchronize(tensor.device)
        received = tensor.detach().cpu()
        expected = _expected(values)
        exact = bool(torch.equal(received, expected))
        ack.send(
            {
                "ok": exact,
                "child_values_exact": exact,
                "child_values_exact_checked": True,
                "transport": transport,
                "received_shape": tuple(received.shape),
                "error": "child received values are not byte-exact" if not exact else "",
                "failure_api": "child tensor validation" if not exact else "",
            }
        )
    except BaseException as exc:  # also reports CUDA errors raised in target
        report = {
            "ok": False,
            "child_values_exact": False,
            "child_values_exact_checked": False,
            "transport": transport,
            "error": f"{type(exc).__name__}: {exc}",
            "failure_api": "child tensor validation",
        }
        try:
            ack.send(report)
        except BaseException:
            pass
        traceback.print_exc()
        raise


def _stop_process(process: Any) -> None:
    if process.pid is None:
        return
    if process.is_alive():
        process.terminate()
        process.join(3)
    if process.is_alive():
        process.kill()
        process.join(3)
    else:
        process.join(0)


def _close_queue(work_queue: Any) -> None:
    try:
        # The feeder thread may be inside CUDA reduction error handling.  Do
        # not let a diagnostic cleanup wait forever on that thread.
        work_queue.cancel_join_thread()
        work_queue.close()
    except (AttributeError, OSError, ValueError):
        pass


def _read_ack(connection: Any) -> dict[str, Any] | None:
    message: dict[str, Any] | None = None
    try:
        while connection.poll(0):
            value = connection.recv()
            if isinstance(value, dict):
                message = value
    except (EOFError, OSError):
        pass
    return message


def _run_case(spec: CaseSpec, ctx: Any, timeout_seconds: float) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {
        "schema": "cuda-ipc-matrix/v1",
        "case": spec.name,
        "transport": spec.transport,
        "device_kind": spec.device_kind,
        "values": spec.values,
        "payload_bytes": spec.values * 4,
        "status": "FAIL",
        "child_exit": None,
        "child_values_exact": False,
        "child_values_exact_checked": False,
        "sender_values_exact_before": False,
        "sender_values_exact_before_checked": False,
        "sender_values_unchanged": False,
        "sender_values_unchanged_checked": False,
        "timed_out": False,
        "failure_api": "",
        "error": "",
    }
    device = "cuda" if spec.device_kind == "cuda" else "cpu"
    tensor = None
    work_queue = None
    process = None
    parent_ack, child_ack = ctx.Pipe(duplex=False)
    try:
        expected = _expected(spec.values)
        tensor = torch.arange(spec.values, dtype=torch.int32, device=device)
        before = tensor.detach().cpu()
        result["sender_values_exact_before"] = bool(torch.equal(before, expected))
        result["sender_values_exact_before_checked"] = True

        if spec.transport == "direct":
            process = ctx.Process(target=_child_direct, args=(tensor, spec.values, child_ack))
        else:
            work_queue = ctx.Queue(maxsize=1)
            process = ctx.Process(
                target=_child_queue,
                args=(work_queue, spec.values, child_ack, timeout_seconds),
            )

        start_time = time.monotonic()
        try:
            process.start()
        except BaseException as exc:
            result["error"] = f"parent process.start: {type(exc).__name__}: {exc}"
            result["failure_api"] = "torch.multiprocessing serialization"
            return result

        if spec.transport == "queue":
            try:
                work_queue.put(tensor, timeout=min(5.0, timeout_seconds))
            except BaseException as exc:
                result["error"] = f"parent Queue.put: {type(exc).__name__}: {exc}"
                result["failure_api"] = "multiprocessing.Queue.put"

        remaining = max(0.1, timeout_seconds - (time.monotonic() - start_time))
        process.join(remaining)
        if process.is_alive():
            result["timed_out"] = True
            result["status"] = "TIMEOUT"
            result["failure_api"] = "child lifetime"
            result["error"] = f"child exceeded {timeout_seconds:.1f}s timeout"
            _stop_process(process)
            return result

        result["child_exit"] = process.exitcode
        ack = _read_ack(parent_ack)
        if ack:
            result["child_values_exact"] = bool(ack.get("child_values_exact", False))
            result["child_values_exact_checked"] = bool(
                ack.get("child_values_exact_checked", False)
            )
            if not ack.get("ok", False):
                result["failure_api"] = str(ack.get("failure_api", "child tensor validation"))
                result["error"] = str(ack.get("error", "child reported failure"))

        after = tensor.detach().cpu()
        result["sender_values_unchanged"] = bool(
            torch.equal(before, after) and torch.equal(after, expected)
        )
        result["sender_values_unchanged_checked"] = True
        if process.exitcode != 0 and not result["failure_api"]:
            if spec.device_kind == "cuda":
                result["failure_api"] = "torch.multiprocessing.reductions.rebuild_cuda_tensor"
            else:
                result["failure_api"] = "child process"
            result["error"] = "child exited before an acknowledgement (often CUDA handle unpickle)"
        if (
            process.exitcode == 0
            and result["child_values_exact"]
            and result["sender_values_exact_before"]
            and result["sender_values_unchanged"]
        ):
            result["status"] = "PASS"
            result["error"] = ""
        elif not result["error"]:
            result["error"] = "one or more exact-value invariants failed"
        return result
    except BaseException as exc:
        result["error"] = f"parent case setup: {type(exc).__name__}: {exc}"
        result["failure_api"] = result["failure_api"] or "case setup"
        return result
    finally:
        if process is not None:
            _stop_process(process)
        _close_queue(work_queue)
        try:
            child_ack.close()
            parent_ack.close()
        except (OSError, ValueError):
            pass
        del tensor
        if spec.device_kind == "cuda":
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except (RuntimeError, AttributeError):
                pass


def _environment(check_cuda: bool = True) -> dict[str, Any]:
    import torch

    report: dict[str, Any] = {
        "schema": "cuda-ipc-matrix/v1",
        "record": "environment",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_checked": check_cuda,
        "cuda_available": bool(torch.cuda.is_available()) if check_cuda else None,
    }
    if report["cuda_available"]:
        report["gpu"] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        report["gpu_free_bytes"] = int(free)
        report["gpu_total_bytes"] = int(total)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout-seconds", type=float, default=30.0,
        help="maximum child lifetime per case (default: 30)",
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="repeat each case; use 20 only after a candidate fix (default: 1)",
    )
    parser.add_argument(
        "--only", action="append", choices=[spec.name for spec in CASE_SPECS],
        help="run only this case; may be supplied more than once",
    )
    parser.add_argument(
        "--cpu-only", action="store_true",
        help="run only CPU spawn/Queue controls; do not initialize CUDA",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.timeout_seconds <= 0 or args.timeout_seconds > 300:
        print("timeout must be in (0, 300] seconds", file=sys.stderr)
        return 2
    if args.repeat <= 0 or args.repeat > 20:
        print("repeat must be in [1, 20]", file=sys.stderr)
        return 2

    selected = CASE_SPECS
    if args.only:
        selected = tuple(spec for spec in CASE_SPECS if spec.name in args.only)
    if args.cpu_only:
        if args.only and any(spec.device_kind != "cpu" for spec in selected):
            print("--cpu-only cannot be combined with CUDA cases", file=sys.stderr)
            return 2
        selected = tuple(spec for spec in selected if spec.device_kind == "cpu")
    if not selected:
        print("no matrix cases selected", file=sys.stderr)
        return 2
    needs_cuda = any(spec.device_kind == "cuda" for spec in selected)

    try:
        import torch

        environment = _environment(needs_cuda)
        print(json.dumps(environment, sort_keys=True), flush=True)
        if needs_cuda and not torch.cuda.is_available():
            print(
                json.dumps(
                    {
                        "schema": "cuda-ipc-matrix/v1",
                        "record": "environment_error",
                        "error": "torch.cuda.is_available() is false",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return 2
    except BaseException as exc:
        print(
            json.dumps(
                {
                    "schema": "cuda-ipc-matrix/v1",
                    "record": "environment_error",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        traceback.print_exc()
        return 2

    ctx = mp.get_context("spawn")
    statuses: list[str] = []
    for repetition in range(1, args.repeat + 1):
        for spec in selected:
            result = _run_case(spec, ctx, args.timeout_seconds)
            result["repetition"] = repetition
            print(json.dumps(result, sort_keys=True), flush=True)
            statuses.append(str(result["status"]))
    if any(status == "TIMEOUT" for status in statuses):
        return 3
    return 0 if statuses and all(status == "PASS" for status in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
