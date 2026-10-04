"""Deformation field phi(x) = sum_k w_k [R_k (x - g_k) + g_k + t_k] and rotation blending (§4.5, §4.9)."""
import roma
import torch

from gaussian_splatting.utils.general_utils import rotmat2quaternion


def node_rotations(omega, R0):
    """R_k = Exp(omega_k) R0_k."""
    return roma.rotvec_to_rotmat(omega) @ R0


def phi(x, idx, w, g, R, t, chunk=262144):
    out = []
    for s in range(0, x.shape[0], chunk):
        i, ww, xx = idx[s:s + chunk], w[s:s + chunk].to(g.dtype), x[s:s + chunk].to(g.dtype)
        gk = g[i]
        y = (R[i] @ (xx[:, None, :] - gk)[..., None]).squeeze(-1) + gk + t[i]
        out.append((ww[..., None] * y).sum(1))
    return torch.cat(out, 0) if out else x.new_zeros((0, 3), dtype=g.dtype)


def quat_of(R):
    """(w,x,y,z) of rotation matrices, positive w."""
    q = rotmat2quaternion(R.double(), normalize=True)
    return torch.where(q[:, :1] < 0, -q, q)


def blend_quat(idx, w, qn, chunk=262144):
    """Weighted quaternion mean with sign alignment to the highest-weight node."""
    out = []
    for s in range(0, idx.shape[0], chunk):
        i, ww = idx[s:s + chunk], w[s:s + chunk].to(qn.dtype)
        q = qn[i]  # [m,K,4]
        ks = ww.argmax(1)
        qs = q[torch.arange(q.shape[0], device=q.device), ks]
        sgn = torch.sign((q * qs[:, None]).sum(-1))
        sgn[sgn == 0] = 1
        qb = (ww[..., None] * sgn[..., None] * q).sum(1)
        out.append(torch.nn.functional.normalize(qb, dim=-1))
    return torch.cat(out, 0) if out else qn.new_zeros((0, 4))


def quat_to_rotmat(q):
    """(w,x,y,z) -> R."""
    return roma.unitquat_to_rotmat(torch.cat([q[:, 1:], q[:, :1]], -1))
