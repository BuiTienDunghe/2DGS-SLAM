#!/usr/bin/env bash
set -u
set -o pipefail

compile_only=false
for arg in "$@"; do
  case "$arg" in
    --compile-only) compile_only=true ;;
    *) echo "[native-wrapper] unknown option: $arg" >&2; exit 2 ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cuda_home="${CUDA_HOME:-/home/vha72/miniconda3/envs/2dgs-slam-e0-1}"
nvcc="${NVCC:-$cuda_home/bin/nvcc}"
cxx="${CXX:-$cuda_home/bin/x86_64-conda-linux-gnu-g++}"
evidence_root="${IPC_NATIVE_EVIDENCE_ROOT:-$repo_root/runs/setup/ipc-repair/native}"
attempt_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
attempt_dir="$evidence_root/attempt-${attempt_stamp}-$$"
attempt_number=0
if ! mkdir -p -- "$evidence_root"; then
  echo "[native-wrapper] cannot create evidence root: $evidence_root" >&2
  exit 2
fi
if [[ ! -d "$evidence_root" || ! -w "$evidence_root" || ! -x "$evidence_root" ]]; then
  echo "[native-wrapper] evidence root is not a writable directory: $evidence_root" >&2
  exit 2
fi
attempt_created=false
while (( attempt_number < 20 )); do
  if mkdir -- "$attempt_dir" 2>/dev/null; then
    attempt_created=true
    break
  fi
  if [[ ! -e "$attempt_dir" ]]; then
    echo "[native-wrapper] cannot create attempt directory under: $evidence_root" >&2
    exit 2
  fi
  attempt_number=$((attempt_number + 1))
  attempt_dir="$evidence_root/attempt-${attempt_stamp}-$$-${attempt_number}"
done
if [[ "$attempt_created" != true ]]; then
  echo "[native-wrapper] exhausted attempt-directory collision retries under: $evidence_root" >&2
  exit 2
fi
out_dir="$attempt_dir"
binary="$out_dir/check_cuda_ipc_native"
compile_log="$out_dir/compile.log"
summary="$out_dir/summary.jsonl"
metadata="$out_dir/metadata.txt"
source_file="$repo_root/scripts/check_cuda_ipc_native.cu"

export PATH="$cuda_home/bin:${PATH:-}"
export LD_LIBRARY_PATH="$cuda_home/lib:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="$cuda_home"

if [[ ! -x "$nvcc" ]]; then
  echo "[native-wrapper] missing nvcc: $nvcc" >&2
  exit 2
fi
if [[ ! -x "$cxx" ]]; then
  echo "[native-wrapper] missing CUDA host compiler: $cxx" >&2
  exit 2
fi
host_version="$("$cxx" -dumpfullversion -dumpversion 2>/dev/null || true)"
if [[ "$host_version" != 12.* && "$host_version" != 12 ]]; then
  echo "[native-wrapper] expected conda GCC 12 host compiler, got: $host_version ($cxx)" >&2
  exit 2
fi

compile_command=(
  "$nvcc" -std=c++17 -O2 -lineinfo -cudart shared -cudadevrt none
  -ccbin "$cxx" -arch=sm_86 -Xcompiler=-fno-omit-frame-pointer
  "$source_file" -o "$binary"
)
{
  printf 'source_path=%s\n' "$source_file"
  printf 'source_sha256='
  sha256sum -- "$source_file" | cut -d' ' -f1
  printf 'nvcc_path=%s\n' "$nvcc"
  printf 'nvcc_version:\n'
  "$nvcc" --version
  printf 'compiler_path=%s\n' "$cxx"
  printf 'compiler_version:\n'
  "$cxx" --version
  printf 'compile_command:'
  printf ' %q' "${compile_command[@]}"
  printf '\n'
} >"$metadata"

set +e
"${compile_command[@]}" >"$compile_log" 2>&1
compile_rc=$?
set -e
if [[ $compile_rc -ne 0 ]]; then
  echo "[native-wrapper] compile failed (rc=$compile_rc); see $compile_log" >&2
  exit 2
fi
printf 'binary_sha256=' >>"$metadata"
sha256sum -- "$binary" | cut -d' ' -f1 >>"$metadata"
if "$compile_only"; then
  echo "[native-wrapper] compile-only artifact: $binary" >&2
  exit 0
fi

overall_rc=0

run_case() {
  local label="$1"
  shift
  local log="$out_dir/${label}.log"
  echo "[native-wrapper] running $label" >&2
  set +e
  timeout --signal=TERM --kill-after=5s 90s \
    "$binary" --timeout-seconds 30 "$@" >"$log" 2>&1
  local rc=$?
  set -e
  local json_line
  json_line="$(rg 'schema.*status' "$log" | tail -n 1)"
  if [[ -n "$json_line" ]]; then
    printf '%s\n' "$json_line" >>"$summary"
  else
    printf '{"schema":"cuda-ipc-native/v1","case":"%s","status":"HARNESS_ERROR","exit_code":%d}\n' \
      "$label" "$rc" >>"$summary"
  fi
  if [[ $rc -ne 0 ]]; then overall_rc=1; fi
}

# Keep these cases independent: a broken memory import must not prevent the
# event path from identifying its own failing API.  The two memory orderings
# distinguish data written before handle export from data written after import.
run_case memory-write-before-handle-small --kind memory --ordering after --bytes 32
run_case memory-handle-before-write-small --kind memory --ordering before --bytes 32
run_case event-record-before-handle --kind event --ordering after
run_case event-handle-before-record --kind event --ordering before

echo "[native-wrapper] summary: $summary" >&2
echo "[native-wrapper] attempt: $attempt_dir" >&2
exit "$overall_rc"
