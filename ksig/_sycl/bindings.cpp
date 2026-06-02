// Python bindings + torch glue for the native SYCL DP kernels (TORCH_PORT
// Sec. 10.3).
//
// The SYCL device kernels live in pde_kernels.sycl (compiled by icpx -fsycl)
// and are deliberately free of the ATen/torch tensor library so that TU links
// cleanly into the shared module. ALL torch-side work -- tensor reshaping,
// the host-side driver transforms, AT_DISPATCH, and pybind -- lives here, in a
// plain C++ TU. We hand the kernels raw device pointers + sizes; they run on
// torch's current XPU stream. Exposed names must match
// ksig.algorithms._try_sycl: 'sig_pde', 'gak_log', 'rws_dtw'.
#include <torch/extension.h>
#include <cstdint>

// Defined in pde_kernels.sycl. Raw device pointers only; each launches a single
// SYCL kernel on torch's current XPU queue. Overloaded on scalar type so the
// AT_DISPATCH lambdas below resolve to the matching instantiation.
void ksig_sig_pde(const double* M, double* K,
                  int64_t nX, int64_t nY, int64_t lX, int64_t lY);
void ksig_sig_pde(const float* M, float* K,
                  int64_t nX, int64_t nY, int64_t lX, int64_t lY);
void ksig_gak_log(const double* logM, double* logK,
                  int64_t nX, int64_t nY, int64_t lX, int64_t lY);
void ksig_gak_log(const float* logM, float* logK,
                  int64_t nX, int64_t nY, int64_t lX, int64_t lY);
void ksig_rws_dtw(const double* D, double* P, const int64_t* seg,
                  int64_t nX, int64_t nY, int64_t lX,
                  int64_t sumLY, int64_t lY_max);
void ksig_rws_dtw(const float* D, float* P, const int64_t* seg,
                  int64_t nX, int64_t nY, int64_t lX,
                  int64_t sumLY, int64_t lY_max);

at::Tensor sig_pde_launch(const at::Tensor& M_in, bool difference) {
  TORCH_CHECK(M_in.device().is_xpu(), "sig_pde expects an XPU tensor");
  at::Tensor M = M_in;
  const bool is_diag = (M.dim() == 3);
  if (is_diag) M = M.unsqueeze(1);
  if (difference) M = at::diff(at::diff(M, 1, -2), 1, -1);
  M = M.contiguous();
  const int64_t nX = M.size(0), nY = M.size(1), lX = M.size(2), lY = M.size(3);
  // Empty sequence (e.g. a length-1 series after differencing -> lX/lY == 0):
  // there are no DP cells, so the kernel value is the empty product, 1. Return
  // it directly -- a zero-length launch would have wg_size == 0. This matches
  // the torch wavefront oracle.
  if (lX == 0 || lY == 0) {
    at::Tensor K = at::ones({nX, nY}, M.options());
    return is_diag ? K.squeeze(1) : K;
  }
  at::Tensor K = at::empty({nX, nY}, M.options());
  AT_DISPATCH_FLOATING_TYPES(M.scalar_type(), "sig_pde", [&] {
    ksig_sig_pde(M.data_ptr<scalar_t>(), K.data_ptr<scalar_t>(),
                 nX, nY, lX, lY);
  });
  return is_diag ? K.squeeze(1) : K;
}

at::Tensor gak_log_launch(const at::Tensor& M_in) {
  TORCH_CHECK(M_in.device().is_xpu(), "gak_log expects an XPU tensor");
  at::Tensor M = M_in;
  const bool is_diag = (M.dim() == 3);
  if (is_diag) M = M.unsqueeze(1);
  // Driver transform on the host side (torch): M/(2-M) then log(clamp(.,eps)).
  const double eps = (M.scalar_type() == at::kDouble) ? 1e-12 : 1e-7;
  M = M / (2.0 - M);
  at::Tensor logM = at::log(at::clamp_min(M, eps)).contiguous();
  const int64_t nX = logM.size(0), nY = logM.size(1),
                lX = logM.size(2), lY = logM.size(3);
  at::Tensor logK = at::empty({nX, nY}, logM.options());
  AT_DISPATCH_FLOATING_TYPES(logM.scalar_type(), "gak_log", [&] {
    ksig_gak_log(logM.data_ptr<scalar_t>(), logK.data_ptr<scalar_t>(),
                 nX, nY, lX, lY);
  });
  return is_diag ? logK.squeeze(1) : logK;
}

at::Tensor rws_dtw_launch(const at::Tensor& D_in, const at::Tensor& warp_lens) {
  TORCH_CHECK(D_in.device().is_xpu(), "rws_dtw expects an XPU tensor");
  TORCH_CHECK(D_in.dim() == 3, "`D` must have ndim==3");
  at::Tensor D = D_in.contiguous();
  const int64_t nX = D.size(0), lX = D.size(1), sumLY = D.size(2);
  at::Tensor wl = warp_lens.to(at::kLong).contiguous();
  const int64_t nY = wl.size(0);
  // Segment endpoints seg[0..nY] and the max length, on the same device.
  at::Tensor seg = at::empty({nY + 1}, wl.options());
  seg.index_put_({0}, 0);
  seg.narrow(0, 1, nY).copy_(at::cumsum(wl, 0));
  const int64_t lY_max = wl.numel() ? wl.max().item<int64_t>() : 0;
  at::Tensor P = at::empty({nX, nY}, D.options());
  AT_DISPATCH_FLOATING_TYPES(D.scalar_type(), "rws_dtw", [&] {
    ksig_rws_dtw(D.data_ptr<scalar_t>(), P.data_ptr<scalar_t>(),
                 seg.data_ptr<int64_t>(), nX, nY, lX, sumLY, lY_max);
  });
  return P;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sig_pde", &sig_pde_launch,
        "SigPDE kernel (antidiagonal wavefront), single SYCL launch.");
  m.def("gak_log", &gak_log_launch,
        "GAK log-space kernel (antidiagonal wavefront), single SYCL launch.");
  m.def("rws_dtw", &rws_dtw_launch,
        "RWS/DTW kernel (antidiagonal wavefront), single SYCL launch.");
}
