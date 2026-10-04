#!/usr/bin/env bash
set -euo pipefail

# Run the stock Replica command through supervise_slam.py. The supervisor is
# a separate process so it can watch the merged child log while the upstream
# parent is waiting for its multiprocessing workers.
cd "$(dirname "$0")/.."

usage() {
  sed -n '1,24p' "$0"
}

smoke=0
range_args=()
run_timeout_args=()
while (($#)); do
  case "$1" in
    --smoke)
      smoke=1
      shift
      ;;
    --range)
      if (($# < 4)); then
        echo "--range requires BEGIN END STEP" >&2
        exit 2
      fi
      for range_value in "$2" "$3" "$4"; do
        if [[ ! "$range_value" =~ ^[0-9]+$ ]]; then
          echo "--range values must be non-negative integers" >&2
          exit 2
        fi
      done
      if (( $3 <= $2 || $4 < 1 )); then
        echo "--range requires BEGIN < END and STEP >= 1" >&2
        exit 2
      fi
      range_args=("$2" "$3" "$4")
      # Supplying a range selects the bounded office0 smoke mode. The normal
      # invocation below always keeps the complete stride-2 scene range.
      smoke=1
      shift 4
      ;;
    --run-timeout)
      if (($# < 2)) || [[ ! "$2" =~ ^(([1-9][0-9]*)|0[.][0-9]*[1-9][0-9]*)$ ]]; then
        echo "--run-timeout requires positive seconds" >&2
        exit 2
      fi
      run_timeout_args=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if (( smoke == 1 && ${#range_args[@]} == 0 )); then
  range_args=(0 200 2)
fi

run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
# Keep one absolute run root and pass paths derived from it to every child.
# This prevents a wrapper/supervisor cwd difference from splitting evidence.
run_dir="$PWD/runs/replica-${run_stamp}"
mkdir -p "$PWD/runs"
if ! mkdir "$run_dir" 2>/dev/null; then
  run_dir="runs/replica-${run_stamp}-$$"
  run_dir="$PWD/$run_dir"
  mkdir "$run_dir"
fi
mkdir -p "$run_dir/supervisor"
printf 'scene\tstatus\texit_code\treason\tlog\tevidence\n' > "$run_dir/status.tsv"

run_scene() {
  local scene="$1"
  local log_path="$run_dir/${scene}.log"
  local evidence_dir="$run_dir/supervisor/${scene}"
  local config_path="configs/replica/${scene}.yaml"
  local -a child_command=(bash scripts/run_local.sh slam.py --config "$config_path")
  if (( ${#range_args[@]} == 3 )); then
    child_command+=(--range "${range_args[0]}" "${range_args[1]}" "${range_args[2]}")
  fi
  local -a supervisor_command=(
    bash scripts/run_local.sh scripts/supervise_slam.py
    --config "$config_path"
    --log "$log_path"
    --evidence-dir "$evidence_dir"
    --repo-root "$PWD"
  )
  if (( ${#range_args[@]} == 3 )); then
    supervisor_command+=(--range "${range_args[0]}" "${range_args[1]}" "${range_args[2]}")
  fi
  if (( ${#run_timeout_args[@]} == 1 )); then
    supervisor_command+=(--run-timeout "${run_timeout_args[0]}")
  fi
  supervisor_command+=(-- "${child_command[@]}")
  local check_rc=0
  if "${supervisor_command[@]}"; then
    check_rc=0
  else
    check_rc=$?
  fi

  local scene_status=FAILED
  if (( check_rc == 0 )); then
    scene_status=COMPLETE
  elif [[ -f "$evidence_dir/status.json" ]] && grep -q '"status": "INTERRUPTED"' "$evidence_dir/status.json"; then
    scene_status=INTERRUPTED
  fi
  local scene_reason=supervisor_no_status
  if [[ -f "$evidence_dir/status.json" ]]; then
    scene_reason=$(awk -F'"' '$2 == "reason" { print $4; exit }' "$evidence_dir/status.json")
    [[ -n "$scene_reason" ]] || scene_reason=supervisor_no_reason
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$scene" "$scene_status" "$check_rc" "$scene_reason" "$log_path" "$evidence_dir" >> "$run_dir/status.tsv"
  if (( check_rc != 0 )); then
    printf 'Failed: %s (%s, exit %s). Log: %s\n' "$scene" "$scene_status" "$check_rc" "$log_path" >&2
    return "$check_rc"
  fi
}

if (( smoke == 1 )); then
  run_scene office0
  printf 'Completed bounded smoke (office0 --range %s %s %s). Run: %s\n' \
    "${range_args[0]}" "${range_args[1]}" "${range_args[2]}" "$run_dir"
  exit 0
fi

for scene in office0 office1 office2 office3 office4 room0 room1 room2; do
  run_scene "$scene"
done
printf 'Completed all scenes. Logs and supervisor evidence: %s\n' "$run_dir"
