"""Exercise installed CUDA kernels before running SLAM."""
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path

import torch
from packaging.requirements import Requirement
from simple_knn._C import distCUDA2
from fused_ssim import fused_ssim

root = Path(__file__).resolve().parents[1]
for line in (root / 'requirements.txt').read_text().splitlines():
    line = line.strip()
    if not line or line.startswith(('#', './')):
        continue
    req = Requirement(line)
    installed = metadata.version(req.name)
    assert installed in req.specifier, (req.name, installed, str(req.specifier))
assert torch.cuda.is_available()
modules = {}
for name in ['diff_surfel_rasterization._C', 'simple_knn._C', 'fused_ssim_cuda', 'curope', 'asmk.hamming']:
    modules[name] = importlib.import_module(name).__file__
    assert str(root / '.venv') in modules[name], (name, modules[name])
torch.manual_seed(42)
points = torch.rand(64, 3, device='cuda')
actual = distCUDA2(points)
expected = torch.cdist(points, points).square().topk(4, largest=False).values[:, 1:].mean(1)
assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4), 'KNN differs from brute force'
x = torch.rand(1, 3, 32, 32, device='cuda', requires_grad=True)
score = fused_ssim(x, x.detach()).mean()
assert abs(score.item() - 1) < 1e-5, score.item()
score.backward()
assert torch.isfinite(x.grad).all()
torch.cuda.synchronize()
report = {'torch': torch.__version__, 'cuda': torch.version.cuda,
          'gpu': torch.cuda.get_device_name(), 'extensions': modules,
          'requirements': 'PASS', 'knn_against_brute_force': 'PASS',
          'fused_ssim_identity_and_backward': 'PASS'}
print(json.dumps(report, indent=2))
(root / 'runs/setup/cuda-check.json').write_text(json.dumps(report, indent=2) + '\n')
