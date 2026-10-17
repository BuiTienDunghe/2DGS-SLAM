"""Diagnostics for T10 (AdamW arithmetic order) and T13 (where the free-direction change comes from)."""
import os, sys, importlib.util, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_obs_decouple as t
DEV = t.DEV
# ---- T10 variants: which formula reproduces torch AdamW (foreach, cuda) bit-exactly?
g = torch.Generator().manual_seed(10)
p = torch.nn.Parameter(torch.zeros(3, device=DEV))
opt = torch.optim.AdamW([p], lr=0.0015, weight_decay=0.01)
m = torch.zeros(3, device=DEV); v = torch.zeros(3, device=DEV)
worst = {k: 0.0 for k in "abcd"}
for it in range(1, 121):
    gr = (torch.randn(3, generator=g) * 0.37).to(DEV)
    p.grad = gr.clone(); opt.step(); ref = p.detach().clone()
    with torch.no_grad(): p.zero_()
    m.lerp_(gr, 0.1); v.mul_(0.999).addcmul_(gr, gr, value=0.001)
    bc1, bc2 = 1 - 0.9 ** it, 1 - 0.999 ** it
    den = (v.sqrt() / bc2 ** 0.5).add_(1e-8)
    ss = -(0.0015 / bc1)
    var = dict(a=(m / den) * ss, b=(m * ss) / den, c=torch.zeros(3, device=DEV).addcdiv(m, den, value=ss),
               d=torch.zeros(3, device=DEV).addcdiv_(m, den, value=ss))
    for k, x in var.items():
        worst[k] = max(worst[k], float(((x - ref).abs() / ref.abs().clamp_min(1e-30)).max()))
print("T10 variants worst rel:", worst)
print("AdamW defaults:", {k: opt.defaults[k] for k in ("foreach", "fused", "capturable")})
# ---- T13: track rotation and the free component through the dec run
n1, n2 = t.unit([0.1, -0.99, 0.1]), t.unit([0.8, 0.0, 0.6])
free = t.unit(np.cross(n1, n2))
d_free, obs_left, dec = t._run_plane_only(True)
print("T13 dec: free change m", d_free, "sum |dxi_r| per axis", dec.sum_abs_dxi_r.tolist(), "sum |dxi_t|", dec.sum_abs_dxi_t.tolist())
