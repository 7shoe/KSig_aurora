"""JIT builder/loader for the native SYCL DP kernels (TORCH_PORT Sec. 10.3).

The extension is compiled on first use via ``torch.utils.cpp_extension.load``
with ``sycl_sources`` (icpx ``-fsycl``, Intel/Level-Zero default target — no
nvptx flags on Aurora). The build is cached by torch under
``TORCH_EXTENSIONS_DIR``; ``available()`` swallows any failure so import of
``ksig`` never depends on a working compiler / XPU.
"""
from __future__ import annotations

import functools
import os
import re
import shlex

import torch

_HERE = os.path.dirname(__file__)


def _fix_sycl_host_flags(cpp_ext) -> None:
  """Patch torch's ``_wrap_sycl_host_flags`` for this icpx + torch build.

  When a ``.sycl`` file is compiled, torch hands the *host* half of the mixed
  compile its flags via ``-fsycl-host-compiler-options=<cflags>`` and folds the
  pybind ABI defines (``-DPYBIND11_COMPILER_TYPE=\\"_gcc\\"`` etc.) into that
  string. The ``\\"`` escaping mis-tokenizes on this icpx: the unterminated
  quote swallows every flag that follows it (``-isystem`` torch includes and
  ``-fPIC``), so ``c10/xpu`` headers go missing and the object is non-PIC and
  fails to link into the shared module.

  ``pde_kernels.sycl`` includes no pybind11, so it never needs those defines;
  ``bindings.cpp`` (a normal C++ TU, not routed through this helper) still gets
  them. We therefore strip ``-DPYBIND11_*`` from the host-compiler-options and
  force ``-fPIC``. Idempotent; a no-op if torch's internal helper is absent.
  """
  orig = getattr(cpp_ext, '_wrap_sycl_host_flags', None)
  if orig is None or getattr(orig, '_ksig_patched', False):
    return

  def patched(cflags):
    cflags = re.sub(r'-DPYBIND11_\S+', '', cflags)
    if '-fPIC' not in cflags:
      cflags += ' -fPIC'
    host_cxx = cpp_ext.get_cxx_compiler()
    return [f'-fsycl-host-compiler={host_cxx}',
            shlex.quote(f'-fsycl-host-compiler-options={cflags}')]

  patched._ksig_patched = True
  cpp_ext._wrap_sycl_host_flags = patched


@functools.lru_cache(maxsize=1)
def get_ext():
  """Build (once) and return the compiled ``ksig_sycl`` extension module.

  Raises:
    RuntimeError: If no XPU device is available.
    Exception: Any compiler/toolchain error from ``cpp_extension.load``.
  """
  if not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
    raise RuntimeError('SYCL fast-path requires an available XPU device.')
  import torch.utils.cpp_extension as cpp_ext
  from torch.utils.cpp_extension import load, include_paths
  _fix_sycl_host_flags(cpp_ext)
  # Header-search belt-and-suspenders: surface the torch include dirs via
  # ``CPATH`` (read by the preprocessor independently of the command line) in
  # case any ``-isystem`` flag is dropped during the mixed SYCL compile.
  _cpath = os.pathsep.join(include_paths())
  if os.environ.get('CPATH'):
    _cpath += os.pathsep + os.environ['CPATH']
  os.environ['CPATH'] = _cpath
  # NOTE: torch's ``load()`` (unlike ``load_inline``) has no ``sycl_sources``
  # kwarg — ``.sycl`` files are auto-detected within ``sources`` via
  # ``_is_sycl_file`` and routed to the SYCL compiler (icpx), which already
  # injects ``-fsycl`` and ``-fsycl-targets=spir64``. Intel/Level-Zero is the
  # default SYCL target on Aurora; do NOT pass nvptx/sm_80 flags (those are for
  # Polaris/NVIDIA, a different machine).
  return load(
    name='ksig_sycl',
    sources=[
      os.path.join(_HERE, 'bindings.cpp'),
      os.path.join(_HERE, 'pde_kernels.sycl'),
    ],
    # Kernels carry explicit names (SigPdeKernel/GakLogKernel/RwsDtwKernel) so
    # we need neither -fsycl-unnamed-lambda (which conflicts with torch's
    # -fsycl-host-compiler mixed-compilation path) nor any target flags.
    # '-fPIC' here lands after -fsycl, outside the span the mangled PYBIND11
    # define swallows (see CPATH note above), so the device object is PIC. The
    # .sycl TU is deliberately free of the ATen/torch tensor library (all torch
    # glue is in bindings.cpp), so its host object carries no absolute
    # relocations against c10 symbols and links into the shared module.
    extra_sycl_cflags=['-O3', '-fPIC'],
    extra_cflags=['-O3'],
    verbose=bool(int(os.environ.get('KSIG_SYCL_VERBOSE', '0'))),
  )


@functools.lru_cache(maxsize=1)
def available() -> bool:
  """Whether the SYCL fast-path can be used (XPU present and ext builds)."""
  try:
    get_ext()
    return True
  except Exception:
    return False
