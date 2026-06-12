#!/usr/bin/env bash
# One command to benchmark every KSig kernel on the live device (Aurora XPU /
# CUDA / CPU): capability + wall-time, then a data-scaling sanity check.
#
#   bash scripts/run_kernel_benchmarks.sh              # full run, auto device
#   bash scripts/run_kernel_benchmarks.sh --quick      # fast pass (~2 min)
#   bash scripts/run_kernel_benchmarks.sh --device cpu # force a device
#   bash scripts/run_kernel_benchmarks.sh --big        # add N=400 to the scaling sweep
#
# Any flags are forwarded to BOTH stages; each stage ignores the flags it does
# not know (argparse parse_known_args), so mixed flags like --big (scaling-only)
# or --reps (profile-only) are safe.
#
# It sources the project torch/oneAPI env first, so this is genuinely one
# command -- you do NOT need to activate anything yourself. Override the env
# script with KSIG_ACTIVATE=/path/to/activate.sh. Runs on a login/UAN node too
# (it just reports CPU); the XPU numbers need a compute node.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$REPO_ROOT/scripts"
RESULTS="$SCRIPTS/results"
mkdir -p "$RESULTS"
STAMP="$(date +%Y%m%d_%H%M%S)"

section() { printf '\n=== %s ===\n' "$1"; }

# --- activate the project torch/oneAPI env (one-command convenience) ----------
ACTIVATE="${KSIG_ACTIVATE:-/home/siebenschuh/Projects/Aurora_HPC/environment/activate_ddp_venv.sh}"
section "Environment"
if [[ -f "$ACTIVATE" ]]; then
  echo "sourcing: $ACTIVATE"
  # shellcheck disable=SC1090
  source "$ACTIVATE"
else
  echo "env script not found ($ACTIVATE); assuming torch/ksig already importable"
  echo "  (set KSIG_ACTIVATE=/path/to/activate.sh to point at it)"
fi
cd "$REPO_ROOT"
echo "repo: $REPO_ROOT"
echo "python: $(command -v python)"

# --- stage 1: capability + performance ----------------------------------------
section "Stage 1/2 — capability + performance (profile_kernels.py)"
PROFILE_OUT="$RESULTS/profile_${STAMP}.txt"
python "$SCRIPTS/profile_kernels.py" "$@" 2>&1 | tee "$PROFILE_OUT"
s1=${PIPESTATUS[0]}

# --- stage 2: data scaling ----------------------------------------------------
section "Stage 2/2 — data scaling (scaling_check.py)"
SCALING_OUT="$RESULTS/scaling_${STAMP}.txt"
python "$SCRIPTS/scaling_check.py" "$@" 2>&1 | tee "$SCALING_OUT"
s2=${PIPESTATUS[0]}

# --- summary ------------------------------------------------------------------
section "Done"
echo "profile : exit $s1  -> $PROFILE_OUT"
echo "scaling : exit $s2  -> $SCALING_OUT"
[[ $s1 -eq 0 && $s2 -eq 0 ]] && echo "both stages OK" || echo "a stage exited non-zero (see logs above)"
exit $(( s1 != 0 || s2 != 0 ))
