"""plan v6 quick (E5): keyframe 0 as a loop candidate and the revisit-burst bookkeeping (CPU, no SLAM).

usage: python tests/test_v6_kf0.py
  K1 switch on : candidate 0 reaches the relocalisation (revisit and featquery paths); -1 stops
  K2 switch off: candidate 0 stops (upstream `> 0`), candidate 5 reaches the relocalisation
  K3 is_this_loop_necessary with last_loop_id = -1: no index() on -1, throttled until N keyframes exist
  K4 burst bookkeeping: a burst starts after >= 3 non-passing checks; dumps at burst positions 0, 10, 20 (max 3)
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deform  # noqa: F401,E402  (sets up the import paths of the repo)
import json  # noqa: E402

from utils.slam_frontend import FrontEnd  # noqa: E402


class Cam:
    def __init__(self, uid):
        self.uid = uid


def make(allow_kf0, revisit_ret, featquery_ret=-1):
    fe = FrontEnd.__new__(FrontEnd)
    fe.config = {"loop": {"allow_kf0": allow_kf0}}
    fe.enable_loop_closure = True
    fe.enable_revisit_loop = True
    fe.verbose = False
    fe.reloc_method = "mast3r"
    fe.allow_kf0 = allow_kf0
    fe.min_loop_id = 0 if allow_kf0 else 1
    fe.log_loop_checks = False
    fe.revisit_dump = False
    fe.calls = []
    fe.detect_loop_by_revisit = lambda depth, cam: revisit_ret
    fe.detect_loop_by_featquery = lambda img, cam: featquery_ret
    fe.is_this_loop_necessary = lambda loop_id: (True, "test")

    def reloc(cam, img, loop_id, loop_type):
        fe.calls.append((loop_type, loop_id))
        return Cam(loop_id)

    fe.reloc_with_mast3r = reloc
    return fe


def k1_k2():
    fe = make(True, 0)
    out = fe.try_loop_closure(None, None, Cam(100))
    assert fe.calls == [("revisit", 0)] and out.uid == 0, fe.calls
    fe = make(True, -1, 0)
    out = fe.try_loop_closure(None, None, Cam(100))
    assert fe.calls == [("featquery", 0)] and out.uid == 0, fe.calls
    fe = make(True, -1, -1)
    assert fe.try_loop_closure(None, None, Cam(100)) is None and fe.calls == []
    fe = make(False, 0, 0)
    assert fe.try_loop_closure(None, None, Cam(100)) is None and fe.calls == [], fe.calls
    fe = make(False, 5)
    out = fe.try_loop_closure(None, None, Cam(100))
    assert fe.calls == [("revisit", 5)] and out.uid == 5
    print("K1 K2 ok: candidate 0 relocalised only with loop.allow_kf0; -1 always stops")


def k3():
    fe = FrontEnd.__new__(FrontEnd)
    fe.old_than_N_keyframe = 12
    fe.last_loop_at_len_kf = 0
    fe.last_loop_id = -1
    fe.key_frame_ids = list(range(0, 20, 2))  # 10 keyframes
    ok, why = fe.is_this_loop_necessary(0)
    assert ok is False and "no loop yet" in why, (ok, why)
    fe.key_frame_ids = list(range(0, 26, 2))  # 13 keyframes
    ok, _ = fe.is_this_loop_necessary(0)
    assert ok is True
    # after a loop to keyframe 0: same throttle as upstream
    fe.last_loop_id, fe.last_loop_at_len_kf = 0, 13
    fe.key_frame_ids = list(range(0, 30, 2))
    ok, _ = fe.is_this_loop_necessary(4)
    assert ok is False
    # upstream initial state (last_loop_id = 0) gives the same answer as -1 before any loop
    fe.last_loop_id, fe.last_loop_at_len_kf = 0, 0
    fe.key_frame_ids = list(range(0, 20, 2))
    assert fe.is_this_loop_necessary(6)[0] is False
    print("K3 ok: no loop yet -> throttled exactly like upstream, no index(-1)")


def k4():
    with tempfile.TemporaryDirectory() as td:
        fe = FrontEnd.__new__(FrontEnd)
        fe.save_dir = td
        fe.verbose = False
        fe.allow_kf0 = True
        fe.log_loop_checks = True
        fe.revisit_dump = True
        fe.rv_burst_gap, fe.rv_dump_every, fe.rv_dump_max = 3, 10, 3
        fe._rv_nopass, fe._rv_pos, fe._rv_dumps, fe._rv_burst = 10 ** 9, 0, 0, 0
        fe.key_frame_ids = [0]
        sent = []
        fe.request_dump = lambda cam, tag, info: sent.append((cam.uid, tag))
        # frames 0..4 fail, 5..39 pass with two short gaps, 40..44 fail, 45..47 pass
        passing = set(range(5, 40)) - {12, 13, 20} | {45, 46, 47}
        for f in range(48):
            fe._rv_last = {"observed_ratio": 0.6 if f in passing else 0.1, "pass": f in passing, "cand_kf": 0 if f in passing else None}
            fe._note_revisit_check(Cam(f))
        assert sent == [(5, "b01_1"), (15, "b01_2"), (25, "b01_3"), (45, "b02_1")], sent
        rows = [json.loads(l) for l in open(os.path.join(td, "loop_checks.jsonl"))]
        assert len(rows) == 48 and rows[5]["burst"] == 1 and rows[12]["burst"] == 1 and rows[42]["burst"] is None and rows[45]["burst"] == 2
    print("K4 ok: bursts and dump requests", sent)


if __name__ == "__main__":
    k1_k2()
    k3()
    k4()
    print("ALL PASS")
