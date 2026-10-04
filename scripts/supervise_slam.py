#!/usr/bin/env python3
"""Run one SLAM command with bounded process-group supervision.

The upstream program creates its result directory from the YAML configuration
at runtime.  This module deliberately does not import ``slam.py``: importing
it would initialize CUDA before the child process is launched.  Instead, the
small amount of configuration needed for post-run acceptance is resolved from
YAML here.

The command's output is appended to ``--log``.  A JSON-lines provenance file
records a start event and a final event, so an interrupted or failed run keeps
its evidence and is never mistaken for a later successful run.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml


EXIT_FAILURE = 1
DEFAULT_TERM_TIMEOUT = 5.0
DEFAULT_POLL_INTERVAL = 0.10
DEFAULT_WORKER_EXIT_GRACE = 1.0
DEFAULT_PREFLIGHT_TIMEOUT = 45.0
DEFAULT_RUN_TIMEOUT = 24.0 * 60.0 * 60.0
TRACEBACK_DRAIN_TIMEOUT = 0.50
PLY_HEADER_LIMIT = 1024 * 1024

EXPECTED_METRICS = (
    "ate_rmse_keyframes_m",
    "ate_rmse_all_tracked_m",
    "mean_psnr",
    "mean_ssim",
    "mean_lpips",
)


ERROR_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "cuda_ipc",
        re.compile(
            r"(?:\b(?:RuntimeError|CudaError|CUDAError)\s*:\s*|"
            r"\bCUDA\s*(?:runtime\s*)?error\s*:\s*)"
            r"(?:(?:CUDA\s*(?:runtime\s*)?error\s*:\s*)?)"
            r"invalid\s+resource\s+handle\b",
            re.IGNORECASE,
        ),
    ),
    (
        "oom",
        re.compile(
            r"(?:\b(?:OutOfMemoryError|RuntimeError|MemoryError)\s*:\s*|"
            r"\bCUDA\s*(?:runtime\s*)?error\s*:\s*)"
            r"(?:(?:CUDA\s*(?:runtime\s*)?error\s*:\s*)?)"
            r"(?:CUDA\s+)?out\s+of\s+memory\b",
            re.IGNORECASE,
        ),
    ),
    (
        "traceback",
        re.compile(r"traceback\s*\(most recent call last\):", re.IGNORECASE),
    ),
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Match this checkout's config loader's recursive override semantics."""

    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _inherit_path(raw: str, repo_root: Path) -> Path:
    candidate = Path(os.path.expanduser(raw))
    if candidate.is_absolute():
        return candidate.resolve()
    # Upstream resolves its usual "configs/..." paths from the checkout cwd.
    return (repo_root / candidate).resolve()


def resolve_config(
    config_path: str | os.PathLike[str], repo_root: str | os.PathLike[str]
) -> tuple[dict[str, Any], str]:
    """Resolve YAML inheritance without importing any SLAM module.

    Returns the resolved mapping and the path of the requested scene config.
    A cycle or malformed YAML fails before starting the child.
    """

    root = Path(repo_root).resolve()
    requested = Path(config_path).expanduser()
    if not requested.is_absolute():
        requested = root / requested
    requested = requested.resolve()

    def load(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
        if path in stack:
            cycle = " -> ".join(str(item) for item in (*stack, path))
            raise ValueError(f"config inheritance cycle: {cycle}")
        with path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"config must contain a mapping: {path}")
        parent_raw = raw.get("inherit_from")
        parent: dict[str, Any] = {}
        if parent_raw is not None:
            if not isinstance(parent_raw, str) or not parent_raw.strip():
                raise ValueError(f"inherit_from must be a non-empty path: {path}")
            parent = load(_inherit_path(parent_raw, root), (*stack, path))
        return _deep_merge(parent, raw)

    return load(requested, ()), str(requested)


def scene_result_parent(config: Mapping[str, Any], repo_root: str | os.PathLike[str]) -> Path:
    """Return ``Results.save_dir/Dataset.type/Dataset.sequence_name``."""

    results = config.get("Results")
    dataset = config.get("Dataset")
    if not isinstance(results, Mapping) or not isinstance(dataset, Mapping):
        raise ValueError("resolved config needs Results and Dataset mappings")
    save_dir = results.get("save_dir")
    dataset_type = dataset.get("type")
    sequence_name = dataset.get("sequence_name")
    if not isinstance(save_dir, (str, os.PathLike)) or not str(save_dir).strip():
        raise ValueError("resolved config needs Results.save_dir")
    if not isinstance(dataset_type, str) or not dataset_type.strip():
        raise ValueError("resolved config needs Dataset.type")
    if not isinstance(sequence_name, str) or not sequence_name.strip():
        raise ValueError("resolved config needs Dataset.sequence_name")

    save_path = Path(os.path.expanduser(str(save_dir)))
    if not save_path.is_absolute():
        save_path = Path(repo_root).resolve() / save_path
    return (save_path / dataset_type / sequence_name).resolve()


def apply_frame_range(
    config: Mapping[str, Any], frame_range: Sequence[int] | None
) -> dict[str, Any]:
    """Apply the same frame override that slam.py applies for smoke runs."""

    if frame_range is None:
        return dict(config)
    if len(frame_range) != 3:
        raise ValueError("frame range needs BEGIN END STEP")
    begin, end, step = (int(value) for value in frame_range)
    if begin < 0 or end <= begin or step < 1:
        raise ValueError("frame range needs 0 <= BEGIN < END and STEP >= 1")
    return _deep_merge(
        config,
        {"Dataset": {"frame_begin": begin, "frame_end": end, "frame_step": step}},
    )


def snapshot_result_dirs(parent: Path) -> set[Path]:
    """Snapshot existing immediate run directories without creating output."""

    if not parent.is_dir():
        return set()
    return {
        entry.resolve()
        for entry in parent.iterdir()
        if entry.is_dir() and not entry.is_symlink()
    }


def find_new_result_dirs(parent: Path, before: set[Path]) -> list[Path]:
    if not parent.is_dir():
        return []
    return sorted(
        (
            entry.resolve()
            for entry in parent.iterdir()
            if entry.is_dir() and not entry.is_symlink() and entry.resolve() not in before
        ),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )


def _nonempty_files(directory: Path, patterns: Iterable[str]) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        for path in directory.glob(pattern):
            if path.is_file() and not path.is_symlink() and path.stat().st_size > 0:
                found.append(path)
    return sorted(set(found))


def _valid_ply(path: Path) -> tuple[bool, str | None]:
    """Check the portable part of a PLY header before accepting an export."""

    header: list[str] = []
    try:
        with path.open("rb") as stream:
            total = 0
            for raw_line in stream:
                total += len(raw_line)
                if total > PLY_HEADER_LIMIT:
                    return False, "PLY header exceeds safety limit"
                try:
                    line = raw_line.decode("ascii").strip()
                except UnicodeDecodeError:
                    return False, "PLY header is not ASCII"
                header.append(line)
                if line == "end_header":
                    break
    except (OSError, UnicodeError) as exc:
        return False, f"cannot read PLY: {exc}"

    if not header or header[0] != "ply":
        return False, "PLY header does not start with 'ply'"
    if not any(line.startswith("format ") for line in header):
        return False, "PLY header has no format declaration"
    if "end_header" not in header:
        return False, "PLY header has no end_header"
    vertices: list[int] = []
    for line in header:
        fields = line.split()
        if len(fields) == 3 and fields[0] == "element" and fields[1] == "vertex":
            try:
                vertices.append(int(fields[2]))
            except ValueError:
                return False, "PLY vertex count is not an integer"
    if len(vertices) != 1 or vertices[0] <= 0:
        return False, "PLY must declare one positive vertex count"
    return True, None


def _valid_tum(path: Path) -> tuple[bool, str | None]:
    """Require finite eight-column rows: frame/time plus xyz and quaternion."""

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return False, f"cannot read TUM trajectory: {exc}"
    rows = [line.split() for line in text.splitlines() if line.strip()]
    if not rows:
        return False, "TUM trajectory has no data rows"
    for row_number, fields in enumerate(rows, start=1):
        if len(fields) != 8:
            return False, f"TUM row {row_number} has {len(fields)} columns, expected 8"
        try:
            values = [float(value) for value in fields]
        except ValueError:
            return False, f"TUM row {row_number} contains non-numeric data"
        if not all(math.isfinite(value) for value in values):
            return False, f"TUM row {row_number} contains non-finite data"
    return True, None


def _finite_metrics(path: Path) -> tuple[bool, str | None]:
    try:
        with path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                return False, "metrics.csv has no header"
            rows = list(reader)
    except (OSError, csv.Error, UnicodeError) as exc:
        return False, f"cannot read metrics.csv: {exc}"
    if not rows:
        return False, "metrics.csv has no data row"

    required = set(EXPECTED_METRICS)
    missing_fields = sorted(required.difference(reader.fieldnames))
    if missing_fields:
        return False, f"metrics.csv is missing fields: {', '.join(missing_fields)}"

    for row_number, row in enumerate(rows, start=2):
        # The upstream writer may gain metadata columns.  Acceptance is tied
        # to the five metric columns above; unrelated columns do not change
        # whether a run is complete.
        for field in EXPECTED_METRICS:
            raw = row.get(field)
            if raw is None or not str(raw).strip():
                return False, f"metrics.csv row {row_number} field {field!r} is blank"
            try:
                value = float(str(raw).strip())
            except ValueError:
                return False, f"metrics.csv row {row_number} field {field!r} is not numeric"
            if not math.isfinite(value):
                return False, f"metrics.csv row {row_number} field {field!r} is not finite"
    return True, None


def validate_artifacts(result_dirs: Sequence[Path], before: set[Path]) -> dict[str, Any]:
    """Accept exactly one new result directory with final non-empty artefacts."""

    candidates = [path for path in result_dirs if path.resolve() not in before]
    if len(candidates) != 1:
        return {
            "ok": False,
            "reason": "expected exactly one newly created scene result directory",
            "candidate_dirs": [str(path) for path in candidates],
        }

    directory = candidates[0]
    groups = {
        # The default Replica config emits these canonical final artefacts.
        # A refined-only directory is incomplete because the base export is
        # part of the stock run contract too.
        "gaussian_map": ("gsmap_*.ply",),
        "mesh": ("mesh_*.ply",),
        # TUM is the canonical trajectory artefact; a trajectory PLY alone is
        # insufficient to accept a completed run.
        "trajectory": ("traj_tum_*.txt",),
        "metrics": ("metrics.csv",),
    }
    artifacts = {name: _nonempty_files(directory, patterns) for name, patterns in groups.items()}
    missing = [name for name, paths in artifacts.items() if not paths]
    metric_error: str | None = None
    format_errors: dict[str, str] = {}
    if not missing:
        for name in ("gaussian_map", "mesh"):
            valid, error = _valid_ply(artifacts[name][0])
            if not valid and error is not None:
                format_errors[name] = error
        valid, error = _valid_tum(artifacts["trajectory"][0])
        if not valid and error is not None:
            format_errors["trajectory"] = error
        metrics_ok, metric_error = _finite_metrics(artifacts["metrics"][0])
        if not metrics_ok:
            missing.append("finite_metrics")
        if format_errors:
            missing.append("invalid_formats")
    return {
        "ok": not missing,
        "reason": None if not missing else "missing or invalid final artefacts",
        "result_dir": str(directory),
        "artifacts": {name: [str(path) for path in paths] for name, paths in artifacts.items()},
        "missing": missing,
        "metrics_error": metric_error,
        "format_errors": format_errors,
    }


def _read_new_lines(stream: Any, offset: int) -> tuple[int, list[str]]:
    """Read complete lines appended after offset; retain an incomplete tail."""

    try:
        stream.seek(offset)
        data = stream.read()
    except OSError:
        return offset, []
    if not data:
        return offset, []
    # Use a binary reader so offsets are byte offsets even when a child emits
    # non-ASCII diagnostics.  The supervisor writes only ASCII lifecycle
    # markers, but child logs are outside its control.
    complete_end = data.rfind(b"\n")
    if complete_end < 0:
        return offset, []
    complete = data[: complete_end + 1]
    new_offset = offset + len(complete)
    # An incomplete line is re-read on the next poll.  This is important for
    # traceback/error lines emitted in more than one write.
    return new_offset, [
        item.decode("utf-8", errors="replace").rstrip("\r\n")
        for item in complete.splitlines()
    ]


def first_error(
    lines: Iterable[str], *, include_traceback: bool = True
) -> dict[str, Any] | None:
    for line_number, line in enumerate(lines, start=1):
        for kind, pattern in ERROR_PATTERNS:
            if not include_traceback and kind == "traceback":
                continue
            if pattern.search(line):
                return {"kind": kind, "line": line_number, "text": line[:2000]}
    return None


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _process_group_members(pgid: int) -> dict[int, str]:
    """Return Linux process-group members and their state (best effort).

    The supervisor only inspects the process group created for its Popen
    leader.  This lets it notice a spawned worker becoming a zombie while the
    leader is still waiting, without using a machine-wide process search or a
    broad kill operation.
    """

    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return {}
    members: dict[int, str] = {}
    for stat_path in proc_root.glob("[0-9]*/stat"):
        try:
            raw = stat_path.read_text(encoding="utf-8")
            closing_comm = raw.rfind(")")
            if closing_comm < 0:
                continue
            # The command name is parenthesized and may itself contain ')'.
            # After the final ')' the first fields are state, ppid, pgrp.
            fields = raw[closing_comm + 2 :].split()
            if len(fields) < 3:
                continue
            if int(fields[2]) == pgid:
                members[int(stat_path.parent.name)] = fields[0]
        except (OSError, ValueError):
            continue
    return members


def terminate_process_group(
    process: subprocess.Popen[Any], timeout: float = DEFAULT_TERM_TIMEOUT
) -> dict[str, Any]:
    """Terminate only ``process``'s own session/process group, bounded by timeout."""

    pgid = process.pid
    sent_term = False
    sent_kill = False
    try:
        os.killpg(pgid, signal.SIGTERM)
        sent_term = True
    except ProcessLookupError:
        pass

    deadline = time.monotonic() + max(0.0, timeout)
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    if process.poll() is None or _group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
            sent_kill = True
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=max(0.2, timeout))
    except subprocess.TimeoutExpired:
        # A second bounded wait is useful on a busy machine, but never allow
        # cleanup to turn a failed run into a permanently waiting supervisor.
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass
    return {"sent_term": sent_term, "sent_kill": sent_kill, "timeout_sec": timeout}


class SceneRunLock:
    """Serialize supervisors targeting one upstream second-resolution scene path."""

    def __init__(self, result_parent: Path) -> None:
        self.result_parent = result_parent.resolve()
        self.path = self.result_parent / ".supervise.lock"
        self._stream: Any | None = None

    def acquire(self) -> None:
        self.result_parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        except BaseException:
            stream.close()
            raise
        self._stream = stream

    def release(self) -> None:
        if self._stream is None:
            return
        try:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        finally:
            self._stream.close()
            self._stream = None


def run_ipc_preflight(
    repo_root: Path,
    evidence_dir: Path,
    *,
    timeout: float = DEFAULT_PREFLIGHT_TIMEOUT,
    stop_requested: Callable[[], int | None] | None = None,
) -> dict[str, Any]:
    """Run the checked-in CUDA IPC probe before launching the SLAM child."""

    probe = repo_root / "runs" / "setup" / "diagnostics" / "cuda_ipc_probe.py"
    log_path = evidence_dir / "ipc-preflight.log"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    if not probe.is_file():
        result = {
            "ok": False,
            "status": "MISSING",
            "probe": str(probe),
            "return_code": None,
            "timed_out": False,
        }
        log_path.write_text("CUDA IPC preflight probe is missing: %s\n" % probe, encoding="utf-8")
        return result

    with log_path.open("w", encoding="utf-8", buffering=1) as stream:
        stream.write("[supervisor] CUDA IPC preflight: %s\n" % shlex.join([sys.executable, str(probe)]))
        stream.flush()
        try:
            process = subprocess.Popen(
                [sys.executable, str(probe)],
                cwd=str(repo_root.resolve()),
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            stream.write(f"[supervisor] preflight launch failed: {exc}\n")
            return {
                "ok": False,
                "status": "LAUNCH_ERROR",
                "probe": str(probe),
                "return_code": None,
                "timed_out": False,
                "error": str(exc),
            }

        timed_out = False
        interrupted: int | None = None
        termination = None
        deadline = time.monotonic() + max(0.1, timeout)
        while process.poll() is None:
            if stop_requested is not None:
                interrupted = stop_requested()
                if interrupted is not None:
                    termination = terminate_process_group(process, min(max(0.1, timeout), 5.0))
                    break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                termination = terminate_process_group(process, min(max(0.1, timeout), 5.0))
                break
            try:
                process.wait(timeout=min(0.10, remaining))
            except subprocess.TimeoutExpired:
                continue
        return_code = process.poll()
        if interrupted is None and stop_requested is not None:
            interrupted = stop_requested()
        ok = not timed_out and interrupted is None and return_code == 0
        stream.write(
            f"[supervisor] CUDA IPC preflight {'passed' if ok else 'failed'} "
            f"(return_code={return_code})\n"
        )
        return {
            "ok": ok,
            "status": "INTERRUPTED" if interrupted is not None else "PASS" if ok else "FAIL",
            "probe": str(probe),
            "return_code": return_code,
            "timed_out": timed_out,
            "termination": termination,
            "interrupted": interrupted,
        }


def _append_jsonl(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(event), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def _write_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def _append_log(stream: Any, message: str) -> None:
    stream.write(f"[supervisor] {message}\n")
    stream.flush()


def run_supervised(
    command: Sequence[str],
    *,
    log_path: Path,
    evidence_dir: Path,
    config_path: Path,
    repo_root: Path,
    frame_range: Sequence[int] | None = None,
    artifact_parent: Path | None = None,
    term_timeout: float = DEFAULT_TERM_TIMEOUT,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    worker_exit_grace: float = DEFAULT_WORKER_EXIT_GRACE,
    run_timeout: float = DEFAULT_RUN_TIMEOUT,
    skip_ipc_preflight: bool = False,
) -> int:
    """Run one child, fail fast on fatal output, and accept only its new result."""

    if not command:
        raise ValueError("the supervised command is empty")
    if run_timeout <= 0:
        raise ValueError("run timeout must be positive")
    repo_root = repo_root.resolve()
    config, requested_config = resolve_config(config_path, repo_root)
    config = apply_frame_range(config, frame_range)
    parent = artifact_parent if artifact_parent is not None else scene_result_parent(config, repo_root)
    parent = (repo_root / parent if not parent.is_absolute() else parent).resolve()
    scene_lock = SceneRunLock(parent)
    scene_lock.acquire()
    before = snapshot_result_dirs(parent)

    evidence_dir.mkdir(parents=True, exist_ok=True)
    _write_once(
        evidence_dir / "resolved_config.yaml",
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    )
    _write_once(evidence_dir / "command.txt", shlex.join(map(str, command)) + "\n")
    provenance_path = evidence_dir / "provenance.jsonl"
    _append_jsonl(
        provenance_path,
        {
            "event": "start",
            "started_at": utc_now(),
            "config": requested_config,
            "repo_root": str(repo_root),
            "command": [str(arg) for arg in command],
            "effective_config": config,
            "frame_range": list(frame_range) if frame_range is not None else None,
            "effective_overrides": (
                {"Dataset": {"frame_begin": frame_range[0], "frame_end": frame_range[1], "frame_step": frame_range[2]}}
                if frame_range is not None
                else {}
            ),
            "artifact_parent": str(parent),
            "result_dirs_before": sorted(str(path) for path in before),
            "run_timeout_sec": run_timeout,
            "lock_path": str(scene_lock.path),
        },
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_stream = log_path.open("a", encoding="utf-8", buffering=1)
    log_stream.seek(0, os.SEEK_END)
    log_offset = log_stream.tell()
    _append_log(log_stream, f"start command: {shlex.join(map(str, command))}")
    log_offset = log_stream.tell()

    previous_handlers: dict[int, Any] = {}
    external_signal: int | None = None

    def on_signal(signum: int, _frame: Any) -> None:
        nonlocal external_signal
        external_signal = signum

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, on_signal)

    process: subprocess.Popen[Any] | None = None
    termination: dict[str, Any] | None = None
    detected: dict[str, Any] | None = None
    child_return: int | None = None
    all_lines: list[str] = []
    traceback_deadline: float | None = None
    seen_workers: set[int] = set()
    worker_loss: dict[str, Any] | None = None
    worker_loss_deadline: float | None = None
    status = "FAILED"
    reason = "launch_error"
    acceptance: dict[str, Any] = {}
    preflight: dict[str, Any] | None = None
    run_deadline: float | None = None

    def refresh_detection() -> None:
        nonlocal detected
        if detected is None:
            detected = first_error(all_lines)
        if detected is not None and detected["kind"] == "traceback" and "detail" not in detected:
            detail = first_error(all_lines, include_traceback=False)
            if detail is not None and detail["line"] > detected["line"]:
                detected["detail"] = detail

    try:
        if skip_ipc_preflight:
            preflight = {
                "ok": True,
                "status": "SKIPPED",
                "probe": str(repo_root / "runs" / "setup" / "diagnostics" / "cuda_ipc_probe.py"),
            }
            _append_log(log_stream, "CUDA IPC preflight skipped (diagnostic mode)")
        else:
            preflight = run_ipc_preflight(
                repo_root,
                evidence_dir,
                stop_requested=lambda: external_signal,
            )
            if preflight.get("interrupted"):
                status, reason = "INTERRUPTED", "external_signal"
            elif not preflight["ok"]:
                reason = "ipc_preflight"
                _append_log(log_stream, "CUDA IPC preflight failed; refusing to launch SLAM child")
            elif external_signal is not None:
                status, reason = "INTERRUPTED", "external_signal"

        if preflight["ok"] and external_signal is None:
            try:
                process = subprocess.Popen(
                    [str(arg) for arg in command],
                    cwd=str(repo_root),
                    stdin=subprocess.DEVNULL,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except OSError as exc:
                _append_log(log_stream, f"launch failed: {exc}")
                reason = "launch_error"

        if process is not None:
            run_deadline = time.monotonic() + run_timeout
            with log_path.open("rb") as reader:
                while True:
                    log_offset, new_lines = _read_new_lines(reader, log_offset)
                    all_lines.extend(new_lines)
                    refresh_detection()
                    child_return = process.poll()

                    if child_return is None and run_deadline is not None and time.monotonic() >= run_deadline:
                        _append_log(log_stream, "run timeout reached; terminating process group")
                        termination = terminate_process_group(process, term_timeout)
                        reason = "run_timeout"
                        break

                    if child_return is None:
                        members = _process_group_members(process.pid)
                        seen_workers.update(pid for pid in members if pid != process.pid)
                        lost = sorted(pid for pid in seen_workers if pid not in members or members.get(pid) == "Z")
                        if lost:
                            if worker_loss is None:
                                worker_loss = {"pids": lost}
                                worker_loss_deadline = time.monotonic() + max(0.0, worker_exit_grace)
                            elif worker_loss_deadline is not None and time.monotonic() >= worker_loss_deadline:
                                _append_log(log_stream, f"owned worker disappeared: {worker_loss['pids']}; terminating process group")
                                termination = terminate_process_group(process, term_timeout)
                                reason = "worker_disappeared"
                                break

                    if external_signal is not None:
                        _append_log(log_stream, f"external signal received: {external_signal}")
                        termination = terminate_process_group(process, term_timeout)
                        reason = "external_signal"
                        break
                    if detected is not None and process.poll() is None:
                        if detected["kind"] == "traceback":
                            if traceback_deadline is None:
                                traceback_deadline = time.monotonic() + TRACEBACK_DRAIN_TIMEOUT
                            if time.monotonic() < traceback_deadline:
                                time.sleep(min(max(0.01, poll_interval), traceback_deadline - time.monotonic()))
                                continue
                        _append_log(log_stream, f"recognized {detected['kind']} error; terminating process group")
                        termination = terminate_process_group(process, term_timeout)
                        reason = "backend_error"
                        break
                    if child_return is not None:
                        # One final read captures complete lines flushed at exit.
                        log_offset, new_lines = _read_new_lines(reader, log_offset)
                        all_lines.extend(new_lines)
                        refresh_detection()
                        break
                    time.sleep(max(0.01, poll_interval))

            if child_return is None:
                child_return = process.poll()
            if child_return is None:
                child_return = process.wait(timeout=max(0.2, term_timeout))

            # SIGTERM/SIGKILL from cleanup is not the original failure.
            if external_signal is not None:
                status, reason = "INTERRUPTED", "external_signal"
            elif detected is not None:
                status, reason = "FAILED", "backend_error"
            elif worker_loss is not None:
                status, reason = "FAILED", "worker_disappeared"
            elif reason == "run_timeout":
                status, reason = "FAILED", "run_timeout"
            elif child_return != 0:
                status, reason = "FAILED", "child_exit"
            else:
                acceptance = validate_artifacts(find_new_result_dirs(parent, before), before)
                status, reason = (
                    ("COMPLETE", "accepted") if acceptance.get("ok") else ("FAILED", "missing_artifacts")
                )
    except subprocess.TimeoutExpired:
        if process is not None:
            termination = terminate_process_group(process, term_timeout)
        status, reason = "FAILED", "supervisor_cleanup_timeout"
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if process is not None and (
            external_signal is not None or detected is not None or worker_loss is not None or child_return not in (None, 0)
        ):
            if process.poll() is None or _group_exists(process.pid):
                termination = termination or terminate_process_group(process, term_timeout)
        if process is not None and child_return is None:
            child_return = process.poll()
        _append_log(log_stream, f"finished status={status} reason={reason} child_return={child_return}")
        log_stream.close()

    final_event = {
        "event": "finish",
        "finished_at": utc_now(),
        "status": status,
        "reason": reason,
        "run_timeout_sec": run_timeout,
        "return_code": 0 if status == "COMPLETE" else 130 if external_signal == signal.SIGINT else 143 if external_signal == signal.SIGTERM else EXIT_FAILURE,
        "child_return_code": child_return,
        "external_signal": external_signal,
        "first_detected_error": detected,
        "worker_loss": worker_loss,
        "termination": termination,
        "acceptance": acceptance,
        "ipc_preflight": preflight,
    }
    _append_jsonl(provenance_path, final_event)
    (evidence_dir / "status.json").write_text(json.dumps(final_event, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    scene_lock.release()
    if status == "COMPLETE":
        return 0
    if external_signal == signal.SIGINT:
        return 130
    if external_signal == signal.SIGTERM:
        return 143
    return EXIT_FAILURE


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="scene YAML used for static artefact discovery"
    )
    parser.add_argument("--log", required=True, type=Path, help="append-only child stdout/stderr log")
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-parent", type=Path, default=None)
    parser.add_argument("--range", nargs=3, type=int, default=None, metavar=("BEGIN", "END", "STEP"))
    parser.add_argument("--term-timeout", type=float, default=DEFAULT_TERM_TIMEOUT)
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL)
    parser.add_argument(
        "--run-timeout",
        type=float,
        default=DEFAULT_RUN_TIMEOUT,
        help="maximum child wall-clock time in seconds (default: 86400)",
    )
    parser.add_argument(
        "--worker-exit-grace", type=float, default=DEFAULT_WORKER_EXIT_GRACE
    )
    parser.add_argument(
        "--skip-ipc-preflight",
        action="store_true",
        help="skip CUDA IPC preflight for CPU-only supervisor diagnostics",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command after --")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        _parser().error("a supervised command is required after --")
    try:
        return run_supervised(
            command,
            log_path=args.log,
            evidence_dir=args.evidence_dir,
            config_path=Path(args.config),
            repo_root=args.repo_root,
            frame_range=args.range,
            artifact_parent=args.artifact_parent,
            term_timeout=args.term_timeout,
            poll_interval=args.poll_interval,
            worker_exit_grace=args.worker_exit_grace,
            run_timeout=args.run_timeout,
            skip_ipc_preflight=args.skip_ipc_preflight,
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"supervise_slam: {exc}", file=sys.stderr)
        return EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
