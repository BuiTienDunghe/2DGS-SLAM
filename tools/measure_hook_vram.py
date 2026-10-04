"""P4: extra VRAM and time of correct_map on a real dump (frozen config). usage: python tools/measure_hook_vram.py DUMP CFG"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deform  # noqa: F401,E402
import torch  # noqa: E402

from deform import config as dcfg  # noqa: E402
from deform.dump import EventState, load_dump  # noqa: E402
from deform.pipeline import correct_map, inp_from_dump  # noqa: E402
from utils.loop_dump import load_deform_config  # noqa: E402

d = load_dump(sys.argv[1])
st = EventState(d)
inp = inp_from_dump(d, st.frame)
cfg = dcfg.resolve(load_deform_config(sys.argv[2]), d["config"])
cfg["seed"] = 0
st.frame(st.kf_uids[0])  # warm the dataset reader
torch.cuda.synchronize()
base = torch.cuda.memory_allocated()
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
res = correct_map(inp, cfg, cfg["variant"])
torch.cuda.synchronize()
dt = time.perf_counter() - t0
peak = torch.cuda.max_memory_allocated()
lg = res["log"]
print(f"N={st.N} accepted={res['accepted']} reason={res['reason']} opt_edl={lg.get('edl_opt_init_mm')}->{lg.get('edl_opt_final_mm')} pairs={lg.get('corr_capped')} time={dt:.1f}s "
      f"base={base / 1024**3:.2f}GB peak={peak / 1024**3:.2f}GB extra={(peak - base) / 1024**3:.2f}GB "
      f"stages={ {k: round(v, 1) for k, v in res['log'].get('t_stage_s', {}).items()} }")
