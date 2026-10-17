#!/usr/bin/env python3
"""Tune detector parameters against your own review log.

    python3 tools/tune.py recording.wav review_log.csv --kind breath \
        --grid br_top=14,18,22 br_min=0.20,0.24,0.30 br_harm=0.5,0.55,0.6

The review log is the CSV from the app's "Save review log" button. Ground truth is every label of the chosen kind you
kept or added by hand. Only the stretch of the recording you reviewed is scored (from 0 to the end of your last
decision, or --window), so a detection there that matches none of your marks, including one you rejected, is a false positive.

Every combination in the grid is scored on the first half of that stretch (train) and the second half (test). The best
train score is reported with its test score, so an overfit shows up as a gap. Copy the winning values into the DETECT
object near the top of the script in index.html, and into DEFAULTS in dlg.py, then run parity_check.py.

Score: a detection matches a mark if they overlap. Recall = marks matched / marks. Precision = detections that match a
mark / all detections in the window. Score is F1,
or F-beta with --beta (beta 2 weights recall twice as much as precision). IoU is the overlap of matched pairs, which
says how well the edges line up. Start and end offsets are detector minus you, in ms (negative = detector is earlier).
"""
import argparse, csv, itertools, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dlg
from scipy.io import wavfile

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("wav"); ap.add_argument("log")
ap.add_argument("--kind", default="breath", choices=["breath", "smack", "dead"])
ap.add_argument("--grid", nargs="*", default=[], help="name=v1,v2,... (parameter names from DEFAULTS in dlg.py)")
ap.add_argument("--window", nargs=2, type=float, metavar=("START", "END"), help="seconds you fully reviewed (default: 0 to your last decision)")
ap.add_argument("--beta", type=float, default=1.0)
ap.add_argument("--top", type=int, default=8)
ap.add_argument("--json", help="write the best parameters (only those that differ from the defaults) to this file")
a = ap.parse_args()

# ---- recording and features (computed once)
sr, x = wavfile.read(a.wav)
if x.ndim > 1: x = x.mean(1)
x = (x.astype(np.float32) / (np.iinfo(x.dtype).max + 1 if x.dtype.kind == "i" else 1.0))
f = dlg.features(x, sr)
print(f"{a.wav}: {len(x)/sr:.1f} s, {sr} Hz | noise floor {f['floor']:.1f} dB, speech level {f['vref']:.1f} dB")

# ---- ground truth from the review log
KIND = {"breath": "breath", "plosive": "plosive", "smack": "smack", "dead": "dead"}
rows = list(csv.DictReader(open(a.log)))
def span(r): return float(r["start"]), float(r["end"])
def decided(r): return r.get("decision", "keep") in ("keep", "reject") or r.get("added") in ("1", 1)
truth = [span(r) for r in rows if r["type"] == a.kind and (r.get("decision", "keep") == "keep" or r.get("added") in ("1", 1))]
rejected = [span(r) for r in rows if r["type"] == a.kind and r.get("decision") == "reject"]
if not truth: sys.exit(f"no kept or added '{a.kind}' labels in {a.log}")
w0, w1 = a.window if a.window else (0.0, max(span(r)[1] for r in rows if decided(r)) + 0.3)
mid = (w0 + w1) / 2
print(f"scoring window {w0:.1f} to {w1:.1f} s | {len(truth)} {a.kind} marks, {len(rejected)} rejected")

def ov(p, q): return max(0.0, min(p[1], q[1]) - max(p[0], q[0]))
def iou(p, q):
    o = ov(p, q); return o / ((p[1] - p[0]) + (q[1] - q[0]) - o + 1e-9)

def run(P):
    P = {**dlg.DEFAULTS, **P}
    br = dlg.detect_breaths(f, P)
    ev = br if a.kind == "breath" else dlg.detect_smacks(f, P, br) if a.kind == "smack" else dlg.detect_dead(f, P)
    return [(s, e) for _, s, e, _ in ev]

def score(det, lo, hi):
    T = [t for t in truth if lo <= t[0] < hi]; D = [d for d in det if lo <= d[0] < hi]
    if not T: return None
    hit = [max(T, key=lambda t: ov(t, d)) if any(ov(t, d) > 0 for t in T) else None for d in D]
    tp = sum(h is not None for h in hit)
    fp = sum(h is None for h in hit)
    rec = sum(any(ov(t, d) > 0 for d in D) for t in T) / len(T)
    prec = tp / (tp + fp) if tp + fp else 0.0
    b2 = a.beta ** 2
    f_b = (1 + b2) * prec * rec / (b2 * prec + rec) if prec + rec else 0.0
    pairs = [(d, h) for d, h in zip(D, hit) if h is not None]
    ious = [iou(d, h) for d, h in pairs]
    ds = [(d[0] - h[0]) * 1000 for d, h in pairs]; de = [(d[1] - h[1]) * 1000 for d, h in pairs]
    return dict(f=f_b, rec=rec, prec=prec, n=len(T), found=int(round(rec * len(T))), det=len(D),
                iou=float(np.median(ious)) if ious else 0.0, ds=float(np.median(ds)) if ds else 0.0, de=float(np.median(de)) if de else 0.0)

def fmt(s):
    return "n/a" if s is None else (f"F {s['f']:.2f} | found {s['found']}/{s['n']} | precision {s['prec']:.2f} | "
                                    f"IoU {s['iou']:.2f} | start {s['ds']:+.0f} ms, end {s['de']:+.0f} ms | {s['det']} detections")

# ---- grid
names, values = [], []
for g in a.grid:
    k, v = g.split("=", 1)
    if k not in dlg.DEFAULTS: sys.exit(f"unknown parameter {k}; see DEFAULTS in tools/dlg.py")
    names.append(k); values.append([float(t) if "." in t or "e" in t.lower() else int(t) for t in v.split(",")])
combos = list(itertools.product(*values)) if names else [()]
if len(combos) > 20000: sys.exit(f"{len(combos)} combinations is too many; narrow the grid")

base = run({})
print("\ndefaults")
print("  train:", fmt(score(base, w0, mid))); print("  test: ", fmt(score(base, mid, w1)))
if not names: sys.exit(0)

res = []
for c in combos:
    P = dict(zip(names, c)); det = run(P)
    res.append((P, score(det, w0, mid), score(det, mid, w1)))
res = [r for r in res if r[1] is not None]
res.sort(key=lambda r: (-r[1]["f"], -r[1]["iou"]))
print(f"\n{len(combos)} combinations, best {min(a.top, len(res))} by train score")
for P, tr, te in res[:a.top]:
    print(" ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in P.items()})
    print("     train:", fmt(tr)); print("     test: ", fmt(te))
best = res[0][0]
if a.json:
    json.dump(best, open(a.json, "w"), indent=2); print("\nwrote", a.json)
print("\nIf the test score is clearly below the train score the grid is overfit: narrow it or review more of the recording.")
