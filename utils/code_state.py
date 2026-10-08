"""plan v6 C9: record which code and which resolved config produced a run (record-only).

Writes into the run folder:
  code_state.json       HEAD, branch, `git status` (tracked deletions under submodules/ only counted),
                        sha256 of untracked code/config files
  code_state.patch      `git diff HEAD` without submodules/ (the uncommitted part of the code)
  config_resolved.yaml  the config after inheritance and command-line overrides
"""
import hashlib
import os
import subprocess
import time

import yaml

from utils import loop_dump

_CODE_EXT = (".py", ".yaml", ".yml", ".sh", ".txt")


def _git(args, cwd):
    return subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True, timeout=60).stdout


def record(run_dir, cfg):
    if not run_dir:
        return
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        status = _git(["status", "--porcelain", "--untracked-files=all"], root).splitlines()
        sub_deleted = [s for s in status if s.startswith(" D submodules/")]
        other = [s for s in status if not s.startswith(" D submodules/")]
        untracked = {}
        for s in other:
            if not s.startswith("?? "):
                continue
            p = s[3:].strip().strip('"')
            fp = os.path.join(root, p)
            if p.endswith(_CODE_EXT) and os.path.isfile(fp) and os.path.getsize(fp) < 2_000_000:
                with open(fp, "rb") as f:
                    untracked[p] = hashlib.sha256(f.read()).hexdigest()
        patch = _git(["diff", "HEAD", "--", ".", ":(exclude)submodules"], root)
        with open(os.path.join(run_dir, "code_state.patch"), "w", encoding="utf-8") as f:
            f.write(patch)
        loop_dump.write_json(os.path.join(run_dir, "code_state.json"), {
            "head": _git(["rev-parse", "HEAD"], root).strip(),
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], root).strip(),
            "status_without_submodule_deletions": other,
            "n_deleted_under_submodules": len(sub_deleted),
            "untracked_code_sha256": untracked,
            "patch_sha256": hashlib.sha256(patch.encode("utf-8")).hexdigest(),
            "patch_lines": patch.count("\n"),
            "wall_time": time.time(),
        })
        with open(os.path.join(run_dir, "config_resolved.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=True)
    except Exception as e:  # never let bookkeeping stop a run
        print(f"[code_state] not recorded: {e}")
