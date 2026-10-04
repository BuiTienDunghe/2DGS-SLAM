"""CPU-only acceptance and process-group tests for supervise_slam.py."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

import pytest

from scripts.supervise_slam import SceneRunLock, run_supervised, terminate_process_group


ROOT = Path(__file__).resolve().parents[1]
VALID_PLY = (
    "ply\n"
    "format ascii 1.0\n"
    "element vertex 1\n"
    "property float x\n"
    "property float y\n"
    "property float z\n"
    "end_header\n"
    "0 0 0\n"
)
VALID_TUM = "0 0 0 0 0 0 0 1\n"


def _config(tmp_path: Path, result_parent: Path) -> Path:
    config = tmp_path / "scene.yaml"
    config.write_text(
        "\n".join(
            [
                "Results:",
                f'  save_dir: "{result_parent.parent.parent}"',
                "Dataset:",
                "  type: replica",
                "  sequence_name: office0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config


def _child(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "fake_child.py"
    script.write_text(body, encoding="utf-8")
    return script


def _run(tmp_path: Path, result_parent: Path, child: Path) -> tuple[int, Path]:
    evidence = tmp_path / "evidence"
    log = tmp_path / "scene.log"
    rc = run_supervised(
        [sys.executable, str(child), str(result_parent)],
        log_path=log,
        evidence_dir=evidence,
        config_path=_config(tmp_path, result_parent),
        repo_root=ROOT,
        skip_ipc_preflight=True,
        poll_interval=0.02,
        term_timeout=0.25,
    )
    return rc, evidence


def _write_complete_result(parent: Path) -> None:
    result = parent / "new_run"
    result.mkdir(parents=True)
    (result / "gsmap_test.ply").write_text(VALID_PLY, encoding="utf-8")
    (result / "mesh_test.ply").write_text(VALID_PLY, encoding="utf-8")
    (result / "traj_tum_test.txt").write_text(VALID_TUM, encoding="utf-8")
    (result / "metrics.csv").write_text(
        "run_name,ate_rmse_keyframes_m,ate_rmse_all_tracked_m,mean_psnr,mean_ssim,mean_lpips\n"
        "new_run,0.1,0.2,20.0,0.9,0.1\n",
        encoding="utf-8",
    )


def test_backend_error_stops_parent_that_is_waiting(tmp_path: Path) -> None:
    child = _child(
        tmp_path,
        "import time\n"
        "print('Traceback (most recent call last):', flush=True)\n"
        "print('RuntimeError: CUDA error: invalid resource handle', flush=True)\n"
        "while True: time.sleep(1)\n",
    )
    started = time.monotonic()
    rc, evidence = _run(tmp_path, tmp_path / "results" / "replica" / "office0", child)
    assert rc != 0
    assert time.monotonic() - started < 3
    events = [json.loads(line) for line in (evidence / "provenance.jsonl").read_text().splitlines()]
    finish = events[-1]
    assert finish["status"] == "FAILED"
    assert finish["reason"] == "backend_error"
    assert finish["first_detected_error"]["kind"] == "traceback"
    assert finish["first_detected_error"]["detail"]["kind"] == "cuda_ipc"
    assert finish["termination"]["sent_term"] is True


def test_silent_owned_worker_death_stops_waiting_parent(tmp_path: Path) -> None:
    child = _child(
        tmp_path,
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(0.12)'])\n"
        "print('backend worker launched', flush=True)\n"
        "while True: time.sleep(1)\n",
    )
    evidence = tmp_path / "evidence"
    rc = run_supervised(
        [sys.executable, str(child)],
        log_path=tmp_path / "scene.log",
        evidence_dir=evidence,
        config_path=_config(tmp_path, tmp_path / "results" / "replica" / "office0"),
        repo_root=ROOT,
        skip_ipc_preflight=True,
        poll_interval=0.02,
        worker_exit_grace=0.12,
        term_timeout=0.2,
    )
    assert rc != 0
    finish = json.loads((evidence / "provenance.jsonl").read_text().splitlines()[-1])
    assert finish["status"] == "FAILED"
    assert finish["reason"] == "worker_disappeared"
    assert finish["worker_loss"]["pids"]
    assert finish["child_return_code"] in (-signal.SIGTERM, -signal.SIGKILL)


def test_run_timeout_stops_silent_hung_child(tmp_path: Path) -> None:
    child = _child(
        tmp_path,
        "import time\n"
        "print('child is silently hung', flush=True)\n"
        "while True: time.sleep(1)\n",
    )
    evidence = tmp_path / "evidence"
    started = time.monotonic()
    rc = run_supervised(
        [sys.executable, str(child)],
        log_path=tmp_path / "scene.log",
        evidence_dir=evidence,
        config_path=_config(tmp_path, tmp_path / "results" / "replica" / "office0"),
        repo_root=ROOT,
        skip_ipc_preflight=True,
        run_timeout=0.20,
        term_timeout=0.20,
        poll_interval=0.02,
    )
    assert rc != 0
    assert time.monotonic() - started < 2
    finish = json.loads((evidence / "provenance.jsonl").read_text().splitlines()[-1])
    assert finish["status"] == "FAILED"
    assert finish["reason"] == "run_timeout"
    assert finish["termination"]["sent_term"] is True


def test_range_is_in_effective_config_and_provenance(tmp_path: Path) -> None:
    parent = tmp_path / "results" / "replica" / "office0"
    child = _child(tmp_path, "import sys; sys.exit(0)\n")
    evidence = tmp_path / "evidence"
    rc = run_supervised(
        [sys.executable, str(child)],
        log_path=tmp_path / "scene.log",
        evidence_dir=evidence,
        config_path=_config(tmp_path, parent),
        repo_root=ROOT,
        frame_range=(3, 17, 2),
        skip_ipc_preflight=True,
    )
    assert rc != 0
    start = json.loads((evidence / "provenance.jsonl").read_text().splitlines()[0])
    assert start["frame_range"] == [3, 17, 2]
    assert start["effective_config"]["Dataset"] == {
        "type": "replica",
        "sequence_name": "office0",
        "frame_begin": 3,
        "frame_end": 17,
        "frame_step": 2,
    }


def test_failed_ipc_preflight_refuses_to_launch_child(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    probe = repo / "runs" / "setup" / "diagnostics" / "cuda_ipc_probe.py"
    probe.parent.mkdir(parents=True)
    probe.write_text("raise SystemExit(1)\n", encoding="utf-8")
    marker = repo / "launched.marker"
    child = _child(
        repo,
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('launched')\n",
    )
    parent = repo / "results" / "replica" / "office0"
    evidence = repo / "evidence"
    rc = run_supervised(
        [sys.executable, str(child)],
        log_path=repo / "scene.log",
        evidence_dir=evidence,
        config_path=_config(repo, parent),
        repo_root=repo,
    )
    assert rc != 0
    assert not marker.exists()
    finish = json.loads((evidence / "provenance.jsonl").read_text().splitlines()[-1])
    assert finish["status"] == "FAILED"
    assert finish["reason"] == "ipc_preflight"
    assert finish["child_return_code"] is None
    assert finish["ipc_preflight"]["status"] == "FAIL"
    assert finish["ipc_preflight"]["return_code"] == 1
    assert (evidence / "ipc-preflight.log").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("signum", "expected_return"),
    ((signal.SIGINT, 130), (signal.SIGTERM, 143)),
)
def test_signal_during_ipc_preflight_cleans_probe_without_launching_child(
    tmp_path: Path, signum: int, expected_return: int
) -> None:
    repo = tmp_path / "repo"
    probe = repo / "runs" / "setup" / "diagnostics" / "cuda_ipc_probe.py"
    probe.parent.mkdir(parents=True)
    probe_started = probe.parent / "started"
    probe.write_text(
        "from pathlib import Path\n"
        "import time\n"
        f"Path({str(probe_started)!r}).write_text('started')\n"
        "time.sleep(10)\n",
        encoding="utf-8",
    )
    marker = repo / "launched.marker"
    child = _child(repo, "from pathlib import Path\n" f"Path({str(marker)!r}).write_text('launched')\n")
    parent = repo / "results" / "replica" / "office0"
    evidence = repo / "evidence"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "supervise_slam.py"),
        "--config",
        str(_config(repo, parent)),
        "--log",
        str(repo / "scene.log"),
        "--evidence-dir",
        str(evidence),
        "--repo-root",
        str(repo),
        "--term-timeout",
        "0.2",
        "--",
        sys.executable,
        str(child),
    ]
    supervisor = subprocess.Popen(command, cwd=ROOT)
    try:
        deadline = time.monotonic() + 3
        while not probe_started.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert probe_started.exists()
        supervisor.send_signal(signum)
        assert supervisor.wait(timeout=3) == expected_return
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait()
    assert not marker.exists()
    finish = json.loads((evidence / "provenance.jsonl").read_text().splitlines()[-1])
    assert finish["status"] == "INTERRUPTED"
    assert finish["reason"] == "external_signal"
    assert finish["external_signal"] == signum
    assert finish["ipc_preflight"]["status"] == "INTERRUPTED"
    assert finish["ipc_preflight"]["termination"]["sent_term"] is True


def test_wrapper_uses_one_run_root_and_records_preflight_reason(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "configs" / "replica").mkdir(parents=True)
    (repo / "configs" / "replica" / "office0.yaml").write_text("{}\n", encoding="utf-8")
    scripts = repo / "scripts"
    scripts.mkdir()
    shutil.copy2(ROOT / "scripts" / "run_replica_logged.sh", scripts / "run_replica_logged.sh")
    fake_runner = scripts / "run_local.sh"
    fake_runner.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "if [[ \"$1\" == \"scripts/supervise_slam.py\" ]]; then\n"
        "  printf '%s\\n' \"$*\" > \"$PWD/runner-args\"\n"
        "  evidence= log=\n"
        "  while (($#)); do\n"
        "    case \"$1\" in\n"
        "      --evidence-dir) evidence=$2; shift 2 ;;\n"
        "      --log) log=$2; shift 2 ;;\n"
        "      *) shift ;;\n"
        "    esac\n"
        "  done\n"
        "  mkdir -p \"$evidence\" \"$(dirname \"$log\")\"\n"
        "  printf '{\\n  \"reason\": \"ipc_preflight\",\\n  \"status\": \"FAILED\"\\n}\\n' > \"$evidence/status.json\"\n"
        "  : > \"$log\"\n"
        "  exit 1\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_runner.chmod(fake_runner.stat().st_mode | os.X_OK)

    result = subprocess.run(
        ["bash", "scripts/run_replica_logged.sh", "--smoke", "--run-timeout", "7"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    run_dirs = sorted((repo / "runs").glob("replica-*") )
    assert len(run_dirs) == 1
    status_lines = (run_dirs[0] / "status.tsv").read_text(encoding="utf-8").splitlines()
    assert status_lines[0] == "scene\tstatus\texit_code\treason\tlog\tevidence"
    fields = status_lines[1].split("\t")
    assert fields[:4] == ["office0", "FAILED", "1", "ipc_preflight"]
    assert Path(fields[4]).parent == run_dirs[0]
    assert Path(fields[5]).parent == run_dirs[0] / "supervisor"
    assert "--run-timeout 7" in (repo / "runner-args").read_text(encoding="utf-8")


def test_scene_run_lock_serializes_contention_and_releases(tmp_path: Path) -> None:
    parent = tmp_path / "results" / "replica" / "office0"
    ready = tmp_path / "holder.ready"
    acquired = tmp_path / "contender.acquired"
    holder_code = (
        "from pathlib import Path\n"
        "import sys, time\n"
        "from scripts.supervise_slam import SceneRunLock\n"
        "lock = SceneRunLock(Path(sys.argv[1])); lock.acquire()\n"
        "Path(sys.argv[2]).write_text('ready')\n"
        "time.sleep(0.45); lock.release()\n"
    )
    contender_code = (
        "from pathlib import Path\n"
        "import sys\n"
        "from scripts.supervise_slam import SceneRunLock\n"
        "lock = SceneRunLock(Path(sys.argv[1])); lock.acquire()\n"
        "Path(sys.argv[2]).write_text('acquired'); lock.release()\n"
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", holder_code, str(parent), str(ready)],
        cwd=ROOT,
    )
    contender = None
    try:
        deadline = time.monotonic() + 2
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        started = time.monotonic()
        contender = subprocess.Popen(
            [sys.executable, "-c", contender_code, str(parent), str(acquired)],
            cwd=ROOT,
        )
        time.sleep(0.10)
        assert not acquired.exists()
        assert contender.wait(timeout=2) == 0
        assert time.monotonic() - started >= 0.25
        assert acquired.read_text(encoding="utf-8") == "acquired"
    finally:
        if contender is not None and contender.poll() is None:
            contender.kill()
            contender.wait()
        if holder.poll() is None:
            holder.kill()
            holder.wait()


def test_success_requires_new_complete_result_and_finite_metrics(tmp_path: Path) -> None:
    parent = tmp_path / "results" / "replica" / "office0"
    # A complete stale result must not satisfy this run.
    stale = parent / "old_run"
    stale.mkdir(parents=True)
    (stale / "gsmap_test.ply").write_text(VALID_PLY, encoding="utf-8")
    (stale / "mesh_test.ply").write_text(VALID_PLY, encoding="utf-8")
    (stale / "traj_tum_test.txt").write_text(VALID_TUM, encoding="utf-8")
    (stale / "metrics.csv").write_text(
        "run_name,ate_rmse_keyframes_m,ate_rmse_all_tracked_m,mean_psnr,mean_ssim,mean_lpips\n"
        "old_run,0.1,0.2,20.0,0.9,0.1\n",
        encoding="utf-8",
    )
    ply_literal = repr(VALID_PLY)
    tum_literal = repr(VALID_TUM)
    child = _child(
        tmp_path,
        "from pathlib import Path\n"
        "import sys\n"
        "parent = Path(sys.argv[1])\n"
        "run = parent / 'new_run'\n"
        "run.mkdir(parents=True)\n"
        f"(run / 'gsmap_test.ply').write_text({ply_literal})\n"
        f"(run / 'mesh_test.ply').write_text({ply_literal})\n"
        f"(run / 'traj_tum_test.txt').write_text({tum_literal})\n"
        "(run / 'metrics.csv').write_text('run_name,ate_rmse_keyframes_m,ate_rmse_all_tracked_m,mean_psnr,mean_ssim,mean_lpips\\nnew_run,0.1,0.2,20.0,0.9,0.1\\n')\n",
    )
    evidence = tmp_path / "evidence"
    log = tmp_path / "scene.log"
    rc = run_supervised(
        [sys.executable, str(child), str(parent)],
        log_path=log,
        evidence_dir=evidence,
        config_path=_config(tmp_path, parent),
        repo_root=ROOT,
        skip_ipc_preflight=True,
        poll_interval=0.02,
        term_timeout=0.25,
    )
    assert rc == 0
    finish = json.loads((evidence / "provenance.jsonl").read_text().splitlines()[-1])
    assert finish["status"] == "COMPLETE"
    assert finish["acceptance"]["result_dir"].endswith("new_run")


def test_zero_exit_rejects_ply_without_positive_vertex_count(tmp_path: Path) -> None:
    parent = tmp_path / "results" / "replica" / "office0"
    child = _child(
        tmp_path,
        "from pathlib import Path\n"
        "import sys\n"
        "run = Path(sys.argv[1]) / 'new_run'\n"
        "run.mkdir(parents=True)\n"
        "bad = 'ply\\nformat ascii 1.0\\nelement vertex 0\\nend_header\\n'\n"
        "(run / 'gsmap_test.ply').write_text(bad)\n"
        f"(run / 'mesh_test.ply').write_text({repr(VALID_PLY)})\n"
        f"(run / 'traj_tum_test.txt').write_text({repr(VALID_TUM)})\n"
        "(run / 'metrics.csv').write_text('run_name,ate_rmse_keyframes_m,ate_rmse_all_tracked_m,mean_psnr,mean_ssim,mean_lpips\\nnew,0.1,0.2,20,0.9,0.1\\n')\n",
    )
    rc, evidence = _run(tmp_path, parent, child)
    assert rc != 0
    finish = json.loads((evidence / "provenance.jsonl").read_text().splitlines()[-1])
    assert finish["reason"] == "missing_artifacts"
    assert "invalid_formats" in finish["acceptance"]["missing"]
    assert "gaussian_map" in finish["acceptance"]["format_errors"]


def test_zero_exit_rejects_non_numeric_tum_trajectory(tmp_path: Path) -> None:
    parent = tmp_path / "results" / "replica" / "office0"
    child = _child(
        tmp_path,
        "from pathlib import Path\n"
        "import sys\n"
        "run = Path(sys.argv[1]) / 'new_run'\n"
        "run.mkdir(parents=True)\n"
        f"(run / 'gsmap_test.ply').write_text({repr(VALID_PLY)})\n"
        f"(run / 'mesh_test.ply').write_text({repr(VALID_PLY)})\n"
        "(run / 'traj_tum_test.txt').write_text('0 0 0 bad 0 0 0 1\\n')\n"
        "(run / 'metrics.csv').write_text('run_name,ate_rmse_keyframes_m,ate_rmse_all_tracked_m,mean_psnr,mean_ssim,mean_lpips\\nnew,0.1,0.2,20,0.9,0.1\\n')\n",
    )
    rc, evidence = _run(tmp_path, parent, child)
    assert rc != 0
    finish = json.loads((evidence / "provenance.jsonl").read_text().splitlines()[-1])
    assert finish["reason"] == "missing_artifacts"
    assert finish["acceptance"]["format_errors"]["trajectory"]


def test_zero_exit_with_missing_or_nonfinite_artifacts_fails(tmp_path: Path) -> None:
    parent = tmp_path / "results" / "replica" / "office0"
    ply_literal = repr(VALID_PLY)
    tum_literal = repr(VALID_TUM)
    child = _child(
        tmp_path,
        "from pathlib import Path\n"
        "import sys\n"
        "run = Path(sys.argv[1]) / 'new_run'\n"
        "run.mkdir(parents=True)\n"
        f"(run / 'gsmap_test.ply').write_text({ply_literal})\n"
        f"(run / 'mesh_test.ply').write_text({ply_literal})\n"
        f"(run / 'traj_tum_test.txt').write_text({tum_literal})\n"
        "(run / 'metrics.csv').write_text('run_name,ate_rmse_keyframes_m,ate_rmse_all_tracked_m,mean_psnr,mean_ssim,mean_lpips\\nnew,nan,0.2,20,0.9,0.1\\n')\n",
    )
    rc, evidence = _run(tmp_path, parent, child)
    assert rc != 0
    finish = json.loads((evidence / "provenance.jsonl").read_text().splitlines()[-1])
    assert finish["status"] == "FAILED"
    assert finish["reason"] == "missing_artifacts"
    assert "finite_metrics" in finish["acceptance"]["missing"]


def test_nonzero_exit_without_diagnostic_is_not_called_backend_error(tmp_path: Path) -> None:
    child = _child(
        tmp_path,
        "import sys\n"
        "print('child exited with status 139', flush=True)\n"
        "sys.exit(139)\n",
    )
    rc, evidence = _run(tmp_path, tmp_path / "results" / "replica" / "office0", child)
    assert rc != 0
    finish = json.loads((evidence / "provenance.jsonl").read_text().splitlines()[-1])
    assert finish["status"] == "FAILED"
    assert finish["reason"] == "child_exit"
    assert finish["child_return_code"] == 139
    assert finish["first_detected_error"] is None


def test_termination_is_limited_to_child_process_group() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    try:
        details = terminate_process_group(process, timeout=0.05)
        assert process.poll() is not None
        assert details["sent_term"] is True
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_external_interrupt_is_recorded_separately(tmp_path: Path) -> None:
    parent = tmp_path / "results" / "replica" / "office0"
    child = _child(
        tmp_path,
        "import time\n"
        "print('child is waiting', flush=True)\n"
        "while True: time.sleep(1)\n",
    )
    evidence = tmp_path / "evidence"
    log = tmp_path / "scene.log"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "supervise_slam.py"),
        "--config",
        str(_config(tmp_path, parent)),
        "--log",
        str(log),
        "--evidence-dir",
        str(evidence),
        "--repo-root",
        str(ROOT),
        "--skip-ipc-preflight",
        "--poll-interval",
        "0.02",
        "--term-timeout",
        "0.25",
        "--",
        sys.executable,
        str(child),
        str(parent),
    ]
    supervisor = subprocess.Popen(command)
    try:
        time.sleep(0.15)
        supervisor.send_signal(signal.SIGINT)
        assert supervisor.wait(timeout=3) == 130
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait()
    finish = json.loads((evidence / "provenance.jsonl").read_text().splitlines()[-1])
    assert finish["status"] == "INTERRUPTED"
    assert finish["reason"] == "external_signal"
