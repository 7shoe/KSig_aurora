#!/usr/bin/env bash
# Probe the machine for the bits the KSig torch/SYCL port cares about:
# node type, Intel XPU visibility (torch), oneAPI compiler, SYCL devices, fp64.
#
# Safe to run on a login/UAN node (it will simply report "no XPU") or a compute
# node. Use it before building ksig._sycl: SYCL needs a compute node with a GPU.
set -uo pipefail

section() { printf '\n=== %s ===\n' "$1"; }

section "Node"
hostname
case "$(hostname)" in
  *uan*|*login*) echo "  -> looks like a LOGIN/UAN node (expect NO XPU device)";;
  *)            echo "  -> not obviously a login node (may be a compute node)";;
esac
uname -sr

section "CPU"
lscpu 2>/dev/null | grep -E "Model name|Socket|Core\(s\) per socket|Thread" || true

section "Loaded modules"
module list 2>&1 | sed 's/^/  /' || echo "  (no module system)"

section "oneAPI compiler"
if command -v icpx >/dev/null 2>&1; then
  echo "  icpx: $(command -v icpx)"
  icpx --version 2>/dev/null | head -1
else
  echo "  icpx NOT on PATH (module load frameworks / oneapi)"
fi

section "SYCL devices (sycl-ls)"
if command -v sycl-ls >/dev/null 2>&1; then
  # --ignore-device-selectors so ONEAPI_DEVICE_SELECTOR filtering doesn't hide GPUs.
  sycl-ls --ignore-device-selectors 2>&1 | sed 's/^/  /' || sycl-ls 2>&1 | sed 's/^/  /'
else
  echo "  sycl-ls NOT on PATH"
fi

section "xpu-smi"
if command -v xpu-smi >/dev/null 2>&1; then
  xpu-smi discovery 2>&1 | head -30 | sed 's/^/  /' || echo "  (xpu-smi discovery failed; no GPU?)"
else
  echo "  xpu-smi NOT on PATH"
fi

section "torch XPU view"
python - <<'PY' 2>&1 | grep -v "XPU device count is zero\|return torch._C" | sed 's/^/  /'
import torch
print("torch", torch.__version__)
has = hasattr(torch, "xpu") and torch.xpu.is_available()
print("xpu.is_available:", has)
if has:
    n = torch.xpu.device_count()
    print("xpu.device_count:", n)
    for i in range(n):
        try:
            print(f"  [{i}]", torch.xpu.get_device_name(i))
        except Exception as e:
            print(f"  [{i}] name? {e}")
    # fp64 check: PVC has it; MPS would not.
    try:
        x = torch.ones(2, dtype=torch.float64, device="xpu")
        print("fp64 on xpu: OK ->", (x + x).sum().item())
    except Exception as e:
        print("fp64 on xpu: NOT supported ->", e)
else:
    print("  (no XPU: SYCL fast-path cannot build/run here; use a compute node)")
print("cuda.is_available:", torch.cuda.is_available())
PY

section "ksig._sycl build probe"
python - <<'PY' 2>&1 | grep -v "XPU device count is zero\|return torch._C" | sed 's/^/  /'
try:
    from ksig._sycl import loader
    print("loader.available():", loader.available())
except Exception as e:
    print("loader import failed:", type(e).__name__, e)
PY

echo
echo "Done. To build/validate SYCL: on a compute node run 'pytest -m \"xpu and sycl\"'."
