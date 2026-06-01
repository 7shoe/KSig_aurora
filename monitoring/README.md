# `monitoring/` — performance harness (measured, not asserted)

Performance is **measured here, never asserted in the pytest gate** (timings are
noisy and hardware-bound). The harness produces JSONL/CSV artifacts maintainers
read to decide whether the SYCL fast-path earns its keep (`TEST_PLAN.md` §10–12).

## Layout
| file | role |
|---|---|
| `grids.py` | benchmark grid tiers (`small` / `medium` / `stress`), memory-bounded |
| `probes.py` | timing, peak-memory, device-utilization, CPU-fallback detection |
| `record.py` | JSONL + CSV writers + schema/provenance |
| `run_benchmarks.py` | driver: iterate a tier, time each kernel, write results |
| `results/` | `*.jsonl` / `*.csv` artifacts (git-ignored) |

## Run
```bash
# Legacy CuPy box (NVIDIA), small tier, 2 GB working-set budget:
CUDA_VISIBLE_DEVICES=0 python -m monitoring.run_benchmarks \
    --tier small --out monitoring/results --max-gb 2

# Aurora later (torch-xpu), once the port exists — same command, the driver
# auto-detects the backend (cupy -> torch) and device (cuda -> xpu).
```

## Guarantees
- **No silent caps.** Any grid point skipped for exceeding `--max-gb` is written
  as a row with `skipped=true` and a `note`, and printed as `SKIP`. A truncated
  run never reads as full coverage.
- **Backend-aware.** `probes` and the driver detect cupy today, torch (CUDA/XPU)
  after the port; peak memory uses the matching counter
  (`cupy` pool high-water / `torch.{cuda,xpu}.max_memory_allocated`).
- **Provenance.** Every row carries the same versions block as the golden
  sidecars, so numbers are comparable across machines/dates.

## SYCL acceptance (§12)
A `.sycl` kernel is kept on the canonical path only if it is **both** correct
(`tests/test_sycl.py`, vs the torch wavefront) **and** beneficial (this harness
shows a reproducible speedup OR memory reduction on ≥ the `medium` tier). If
correct but not beneficial → stay on torch. If beneficial but not correct →
reject.
