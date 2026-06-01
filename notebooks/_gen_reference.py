"""Freeze the NVIDIA-CUDA reference for the ``./notebooks`` demos.

Run ONCE on a working CUDA + CuPy box (same provenance idea as
``tests/freeze/make_golden.py``): for each demo feature it records

* a small, mostly hardware-independent **correctness summary** (shape, a 3x3
  value block for the deterministic kernels, the normalised diagonal, and — for
  the randomised feature maps — the Monte-Carlo error to the exact kernel), and
* a **scaling curve** (median wall time over a small size grid),

into ``notebooks/cuda_reference.json``.  The notebooks then draw that curve in
green and compare the live (blue) torch-native numbers against it on Aurora.

    python notebooks/_gen_reference.py            # writes notebooks/cuda_reference.json

Idempotent except for the ``created_utc`` field and timing noise.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np

import _nbtools as nb

HERE = Path(__file__).resolve().parent
OUT = HERE / "cuda_reference.json"

# Default demo input shared by every feature's correctness summary.
DEMO = dict(n=16, L=20, d=3, seed=0)


def _meta():
    env = nb.detect_env()
    versions = {}
    for m in ("numpy", "cupy", "numba", "torch", "ksig"):
        try:
            versions[m] = __import__(m).__version__
        except Exception:
            versions[m] = None
    gpu = None
    try:
        import cupy as cp
        p = cp.cuda.runtime.getDeviceProperties(0)["name"]
        gpu = p.decode() if isinstance(p, bytes) else p
    except Exception:
        pass
    return {"gpu": gpu, "backend": env["backend"], "device": env["device"],
            "versions": versions,
            "created_utc": _dt.datetime.utcnow().isoformat(timespec="seconds")}


def _block(K, k=3):
    return np.round(np.asarray(K)[:k, :k], 4).tolist()


def _rel_err(Kapprox, Kexact):
    Kapprox, Kexact = np.asarray(Kapprox), np.asarray(Kexact)
    return float(np.linalg.norm(Kapprox - Kexact) / np.linalg.norm(Kexact))


def _scaling(make_call, xs, device):
    """``make_call(x) -> (thunk, n_pairs)``; time each thunk on the device."""
    secs, thru = [], []
    for x in xs:
        thunk, npairs = make_call(x)
        t = nb.timeit(thunk, reps=5, warmup=1, device=device)
        secs.append(round(t, 6))
        thru.append(round(npairs / t, 1) if (t > 0 and npairs) else None)
    return {"x": list(xs), "seconds": secs, "pairs_per_sec": thru}


def _kernel_call(kobj, n, L, d, seed=1):
    """A ready-to-time Gram evaluation of an already-built kernel."""
    XX = nb.simulate(n, L, d, seed)
    return (lambda: kobj(XX)), n * n


def _feature_call(make_k, n, L, d, seed=1):
    """Build + fit a feature kernel, return a thunk timing its Gram."""
    XX = nb.simulate(n, L, d, seed)
    kk = make_k()
    kk.fit(XX)
    return (lambda: kk(XX)), n * n


def gen():
    import ksig
    sk = ksig.static.kernels
    K = ksig.kernels
    feats = ksig.static.features
    proj = ksig.projections

    dev = nb.detect_env()["device"]
    meta = _meta()
    out = {"meta": meta}
    X = nb.simulate(**DEMO)

    # ---- exact full-rank gram for the randomised-feature error checks ----
    def exact_sig(XX):
        e = K.SignatureKernel(n_levels=4, order=1, normalize=True,
                              static_kernel=sk.RBFKernel())
        return nb.as_host(e(XX))

    # ---- 1. full-rank Signature Kernel -----------------------------------
    sig = K.SignatureKernel(n_levels=4, order=2, normalize=True,
                            static_kernel=sk.RBFKernel())
    G = nb.as_host(sig(X))
    out["signature_kernel"] = {
        "meta": meta,
        "demo": {"input": "simulate(16, 20, 3, seed=0)",
                 "kwargs": "n_levels=4, order=2, normalize=True, static=RBF",
                 "gram_shape": list(G.shape), "gram_block": _block(G),
                 "diag_mean": round(float(np.diag(G).mean()), 6)},
        "scaling": {"axis": "sequence length L  (n=32, d=3, n_levels=4, order=2)",
                    **_scaling(lambda L: _kernel_call(sig, 32, L, 3),
                               [8, 16, 32, 64, 128], dev)},
    }

    # ---- 2. Signature-PDE Kernel -----------------------------------------
    pde = K.SignaturePDEKernel(difference=True, normalize=True,
                               static_kernel=sk.RBFKernel())
    G = nb.as_host(pde(X))
    out["signature_pde_kernel"] = {
        "meta": meta,
        "demo": {"input": "simulate(16, 20, 3, seed=0)",
                 "kwargs": "difference=True, normalize=True, static=RBF",
                 "gram_shape": list(G.shape), "gram_block": _block(G),
                 "diag_mean": round(float(np.diag(G).mean()), 6)},
        "scaling": {"axis": "sequence length L  (n=32, d=3)",
                    **_scaling(lambda L: _kernel_call(pde, 32, L, 3),
                               [8, 16, 32, 64, 128], dev)},
    }

    # ---- 3. Global Alignment Kernel --------------------------------------
    # Shorter paths + a wider RBF bandwidth so the off-diagonal similarities are
    # informative (random walks under a unit-bandwidth GAK are near-orthogonal,
    # which would make the reference block an uninteresting identity).
    Xg = nb.simulate(16, 8, 3, seed=0)
    gak = K.GlobalAlignmentKernel(static_kernel=sk.RBFKernel(bandwidth=5.0))
    G = nb.as_host(gak(Xg))
    out["global_alignment_kernel"] = {
        "meta": meta,
        "demo": {"input": "simulate(16, 8, 3, seed=0)",
                 "kwargs": "static=RBF(bandwidth=5.0)  (normalized, log-space)",
                 "gram_shape": list(G.shape), "gram_block": _block(G),
                 "diag_mean": round(float(np.diag(G).mean()), 6)},
        "scaling": {"axis": "sequence length L  (n=32, d=3)",
                    **_scaling(lambda L: _kernel_call(gak, 32, L, 3),
                               [8, 16, 32, 64, 128], dev)},
    }

    Xe = nb.simulate(24, 20, 3, seed=0)
    Kex = exact_sig(Xe)

    # ---- 4. Random Fourier Signature Features (RFSF-TRP) ------------------
    def make_rfsf(nc):
        return K.SignatureFeatures(
            n_levels=4, order=1, normalize=True,
            static_features=feats.RandomFourierFeatures(
                n_components=nc, random_state=0),
            projection=proj.TensorizedRandomProjection(
                n_components=nc, rank=1, random_state=0))

    conv = []
    for nc in (50, 100, 250, 500):
        r = make_rfsf(nc); r.fit(Xe)
        conv.append({"n_components": nc,
                     "rel_err": round(_rel_err(nb.as_host(r(Xe)), Kex), 4)})
    r = make_rfsf(100); r.fit(Xe)
    G = nb.as_host(r(Xe)); P = nb.as_host(r.transform(Xe))
    out["rfsf_features"] = {
        "meta": meta,
        "demo": {"input": "simulate(24, 20, 3, seed=0)",
                 "kwargs": "n_levels=4, RFF(n_components=100), TRP(rank=1), normalize=True",
                 "gram_shape": list(G.shape), "feature_shape": list(P.shape),
                 "diag_mean": round(float(np.diag(G).mean()), 6),
                 "rel_err_vs_exact": round(_rel_err(G, Kex), 4),
                 "convergence": conv,
                 "note": ("RNG-dependent: exact element values differ across "
                          "backends, but diag==1, symmetry, and rel_err (which "
                          "shrinks with n_components) are portable.")},
        "scaling": {"axis": "n_samples N  (L=50, d=5, n_components=100)",
                    **_scaling(lambda N: _feature_call(lambda: make_rfsf(100),
                                                       N, 50, 5),
                               [64, 128, 256, 512, 1024], dev)},
    }

    # ---- 5. Low-Rank Signature Features (projection only, no RFF) ---------
    def make_lowrank(nc):
        return K.SignatureFeatures(
            n_levels=4, order=1, normalize=True, static_features=None,
            projection=proj.TensorizedRandomProjection(
                n_components=nc, rank=1, random_state=0))

    conv = []
    for nc in (50, 100, 250, 500):
        lr = make_lowrank(nc); lr.fit(Xe)
        conv.append({"n_components": nc,
                     "rel_err": round(_rel_err(nb.as_host(lr(Xe)), Kex), 4)})
    lr = make_lowrank(100); lr.fit(Xe)
    G = nb.as_host(lr(Xe)); P = nb.as_host(lr.transform(Xe))
    out["low_rank_signature"] = {
        "meta": meta,
        "demo": {"input": "simulate(24, 20, 3, seed=0)",
                 "kwargs": "n_levels=4, TRP(n_components=100, rank=1), normalize=True",
                 "gram_shape": list(G.shape), "feature_shape": list(P.shape),
                 "diag_mean": round(float(np.diag(G).mean()), 6),
                 "rel_err_vs_exact": round(_rel_err(G, Kex), 4),
                 "convergence": conv,
                 "note": ("Low-rank random projection of the signature; same "
                          "portability caveat as RFSF.")},
        "scaling": {"axis": "n_samples N  (L=50, d=5, n_components=100)",
                    **_scaling(lambda N: _feature_call(lambda: make_lowrank(100),
                                                       N, 50, 5),
                               [64, 128, 256, 512, 1024], dev)},
    }

    # ---- 6. Random Warping Series ----------------------------------------
    def make_rws(nc):
        return K.RandomWarpingSeries(n_components=nc, stdev=1., max_warp=32,
                                     normalize=True, random_state=0)

    rws = make_rws(100); rws.fit(Xe)
    G = nb.as_host(rws(Xe))
    eig = float(np.linalg.eigvalsh((G + G.T) / 2).min())
    out["random_warping_series"] = {
        "meta": meta,
        "demo": {"input": "simulate(24, 20, 3, seed=0)",
                 "kwargs": "n_components=100, stdev=1.0, max_warp=32, normalize=True",
                 "gram_shape": list(G.shape),
                 "diag_mean": round(float(np.diag(G).mean()), 6),
                 "symmetric_max_abs": round(float(np.abs(G - G.T).max()), 8),
                 "min_eig": round(eig, 6),
                 "note": ("DTW-based, RNG-dependent: element values differ "
                          "across backends; diag==1, symmetry and PSD-ness are "
                          "the portable invariants.")},
        "scaling": {"axis": "n_samples N  (L=50, d=5, n_components=100)",
                    **_scaling(lambda N: _feature_call(lambda: make_rws(100),
                                                       N, 50, 5),
                               [64, 128, 256, 512, 1024], dev)},
    }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}  ({OUT.stat().st_size} bytes)")
    for name, v in out.items():
        if name == "meta":
            continue
        sc = v["scaling"]
        print(f"  {name:24s} x={sc['x']}  t={sc['seconds']}")


if __name__ == "__main__":
    gen()
