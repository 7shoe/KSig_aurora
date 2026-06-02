"""Optional native SYCL fast-path for the three DP kernels on Aurora XPU.

This package is *optional* and *profile-gated*: the portable torch wavefront in
``ksig.algorithms`` is always the canonical, always-correct path (and the
numerical oracle for these kernels). ``ksig.algorithms._try_sycl`` dispatches
here only when running on an XPU tensor and :func:`loader.available` is true
(the extension built). Anything else falls through to the torch wavefront.

Building requires an Aurora **compute node** with an Intel XPU visible plus the
oneAPI compiler (``icpx -fsycl``); see ``loader`` and ``README.md``.
"""
