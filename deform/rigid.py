"""Offline replay of the baseline rigid map correction (BackEnd.apply_rigid_correction)."""
import torch

from deform.render_utils import DEV
from gaussian_splatting.utils.general_utils import rotmat2quaternion


def quat_mul(q1, q2):
    """Hamilton product, (w,x,y,z), same formula as GaussianModel.update_after_pgo."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], -1)


def delta_T(dump):
    """{uid: 4x4 float32 world-frame increment P*_j P_j^-1} exactly as the backend computes it."""
    kf = set(int(u) for u in dump["keyframe_uids"])
    done = set()
    out = {}
    for fid in dump["all_cam_ids"]:
        fid = int(fid)
        if fid not in kf:
            continue
        if fid in done:  # backend already replaced T by the optimized pose -> identity
            out[fid] = torch.eye(4, device=DEV)
            continue
        opt_w2c = dump["poses_pgo"][fid].to(DEV).to(torch.float32).inverse()
        pre = dump.get("poses_pre_kf", dump["poses_pre"])
        old_w2c = torch.linalg.inv(pre[fid].double()).float().to(DEV)
        out[fid] = opt_w2c.inverse() @ old_w2c
        done.add(fid)
    return out


def apply_rigid(xyz, rot, tc, dT):
    """xyz [N,3], rot [N,4] (normalized), tc [N] -> rigidly corrected (xyz, rot) on GPU."""
    N = xyz.shape[0]
    U = torch.eye(4, device=DEV).unsqueeze(0).repeat(N, 1, 1)
    tc = tc.to(DEV).reshape(-1)
    for fid, T in dT.items():
        m = tc == fid
        if m.any():
            U[m] = T.unsqueeze(0).repeat(int(m.sum()), 1, 1)
    R, t = U[:, :3, :3], U[:, :3, 3]
    xyz_new = torch.bmm(R, xyz.to(DEV).float().unsqueeze(-1)).squeeze(-1) + t
    q_new = quat_mul(rotmat2quaternion(R), rot.to(DEV).float())
    return xyz_new, torch.nn.functional.normalize(q_new, dim=-1)


def replay(dump, use_online=True):
    """Rigid fix from "pre". Uses the increments the backend really applied (dT_online) when dumped."""
    if use_online and "dT_online" in dump:
        dT = {int(u): T.float().to(DEV) for u, T in dump["dT_online"].items()}
    else:
        dT = delta_T(dump)
    g = dump["gauss_pre"]
    xyz, rot = apply_rigid(g["xyz"], g["rot"], g["tc"], dT)
    return xyz, rot, dT


def quat_angle(q1, q2):
    """Rotation angle between unit quaternions (sign-invariant, accurate near 0)."""
    q1, q2 = q1.double(), q2.double()
    s = torch.sign((q1 * q2).sum(-1, keepdim=True))
    s[s == 0] = 1
    chord = (q1 - s * q2).norm(dim=-1) / 2
    return 4.0 * torch.asin(chord.clamp(max=1.0))
