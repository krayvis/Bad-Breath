import os, sys, csv, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dlg
from scipy.io import wavfile
wav, log = sys.argv[1:3]
sr, x = wavfile.read(wav); x = x.astype(np.float32)/32768
f = dlg.features(x, sr)
user = [(r["type"], float(r["start"]), float(r["end"])) for r in csv.DictReader(open(log))]
W = 186.0
def ov(a,b): return max(0, min(a[2],b[2])-max(a[1],b[1]))
def score(ev, kind, lo, hi):
    mine = [u for u in user if u[0]==kind and lo<=u[1]<hi]
    det = [e for e in ev if e[0]==kind and lo<=e[1]<hi]
    rec = sum(any(ov(u,e)>0 for e in det) for u in mine)
    pre = sum(any(ov(u,e)>0 for u in mine) for e in det)
    return rec, len(mine), pre, len(det)
def f1(r):
    rec,nm,pre,nd = r
    p = pre/nd if nd else 0; q = rec/nm if nm else 0
    return 2*p*q/(p+q) if p+q else 0
res = []
grid = dict(br_loose=[3,5], br_top=[6,12,18], br_harm=[0.55,0.7], br_cent_loose=[2600,3200], br_cent=[2000,2600,3200],
            br_seeds=[3,5,8], br_min=[0.2,0.24], br_pre=[27,35,45], br_gap=[3,5])
keys = list(grid)
for vals in itertools.product(*[grid[k] for k in keys]):
    P = dict(zip(keys, vals))
    PP = {**dlg.DEFAULTS, **P}
    ev = dlg.detect_breaths(f, PP)
    a = score(ev, "breath", 0, 93); b = score(ev, "breath", 93, W)
    res.append((f1(a), f1(b), P, a, b))
res.sort(key=lambda r: -(r[0]))
print("top by first half F1 (train), with second half (test):")
for r in res[:8]:
    print(f"train F1 {r[0]:.2f} {r[3]} | test F1 {r[1]:.2f} {r[4]} | {r[2]}")
best = res[0][2]
ev = dlg.detect_breaths(f, {**dlg.DEFAULTS, **best})
print("best-train on whole:", score(ev,"breath",0,W))
