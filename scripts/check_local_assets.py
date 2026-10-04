"""Verify local Replica frame alignment and record checkpoint SHA-256 digests."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCENES = ['office0', 'office1', 'office2', 'office3', 'office4', 'room0', 'room1', 'room2']
report = {'scenes': {}, 'checkpoints': {}}
for scene in SCENES:
    root = ROOT / 'datasets' / 'replica' / scene
    rgb = sorted((root / 'results').glob('frame*.jpg'))
    depth = sorted((root / 'results').glob('depth*.png'))
    expected = [f'{i:06d}' for i in range(2000)]
    assert [p.stem[5:] for p in rgb] == expected, (scene, 'RGB indices')
    assert [p.stem[5:] for p in depth] == expected, (scene, 'depth indices')
    poses = np.loadtxt(root / 'traj.txt')
    assert poses.shape == (2000, 16) and np.isfinite(poses).all(), (scene, 'poses')
    for index in [0, 999, 1999]:
        for path in [rgb[index], depth[index]]:
            with Image.open(path) as im:
                im.load()
                assert im.size == (1200, 680), (path, im.size)
    report['scenes'][scene] = {'rgb': len(rgb), 'depth': len(depth), 'poses': len(poses), 'sample_dimensions': [1200, 680]}
    print(scene, 'PASS', flush=True)
for name in ['MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth',
             'MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth',
             'MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl']:
    path = ROOT / 'submodules' / 'dust3r' / 'checkpoints' / name
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    report['checkpoints'][name] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
    print(name, 'hashed', flush=True)
output = ROOT / 'runs' / 'setup' / 'assets.json'
output.write_text(json.dumps(report, indent=2) + '\n')
print(output, flush=True)
