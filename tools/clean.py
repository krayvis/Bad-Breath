# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Winston Moy. See LICENSE.
"""Apply a label file to a recording: duck breaths, soften smacks and plosives, shorten dead air.

  python3 clean.py in.wav labels.txt out.wav [--gap 1.5] [--breath-db -12] [--smack-db -24] [--report]

Labels are Audacity text labels (start<TAB>end<TAB>text). The first word picks the action:
  breath  -> eased down to --breath-db (default silence) over --ramp, back up over --ramp-up (ramps inside the label)
  smack   -> volume lowered by --smack-db with 5 ms ramps
  plosive -> low end (below 120 Hz) filtered out of the region, ramped in and out
  dead    -> faded to silence over --ramp; if longer than --gap seconds the silent middle is cut so --gap seconds remain
Review the labels in the viewer first, then run this on the reviewed file.
"""
import sys, argparse
import numpy as np
from scipy import signal
from scipy.io import wavfile

ap = argparse.ArgumentParser()
ap.add_argument("wav"); ap.add_argument("labels"); ap.add_argument("out")
ap.add_argument("--gap", type=float, default=1.5, help="seconds of dead air to leave (default 1.5)")
ap.add_argument("--breath-db", type=float, default=-60.0, help="depth of the breath dip; -60 is effectively silence")
ap.add_argument("--ramp", type=float, default=0.15, help="seconds to ease breaths and dead air down to silence, after the word before (default 0.15)")
ap.add_argument("--ramp-up", type=float, default=0.05, help="seconds to come back up before the next word, which starts quickly (default 0.05)")
ap.add_argument("--pad", type=float, default=0.02, help="seconds added to each side of a breath (default 0.02)")
ap.add_argument("--smack-db", type=float, default=-24.0)
ap.add_argument("--report", action="store_true")
a = ap.parse_args()

sr, x = wavfile.read(a.wav)
dtype = x.dtype
x = x.astype(np.float64) / (32768.0 if dtype == np.int16 else 1.0)
mono = x.ndim == 1
if mono:
    x = x[:, None]

labs = []
for line in open(a.labels):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 3:
        labs.append((p[2].split()[0].lower(), float(p[0]), float(p[1])))
labs.sort(key=lambda l: l[1])

def ramp_env(n, s, e, gain, r):
    """Gain envelope: 1 outside, `gain` inside [s,e), linear ramps of r samples just outside each edge."""
    g = np.ones(n)
    s, e = max(0, s), min(n, e)
    if e <= s:
        return g
    g[s:e] = gain
    a0 = max(0, s - r)
    if s > a0:
        g[a0:s] = np.linspace(1, gain, s - a0, endpoint=False)
    b1 = min(n, e + r)
    if b1 > e:
        g[e:b1] = np.linspace(gain, 1, b1 - e, endpoint=False)[::1]
    return g

def dip_env(n, s, e, gain, r_down, r_up):
    """Gain envelope: eases down to `gain` over r_down samples from the start of [s,e), holds, and comes back up
    over r_up samples ending at e. Ramps sit inside the label, so audio outside it is untouched.
    If the label is too short for both, the ramps shrink in proportion."""
    g = np.ones(n)
    s, e = max(0, s), min(n, e)
    if e - s < 2:
        return g
    if r_down + r_up > e - s:
        k = (e - s) / (r_down + r_up); r_down, r_up = int(r_down * k), int(r_up * k)
    g[s:e] = gain
    if r_down > 1:
        t = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, r_down, endpoint=False))   # raised cosine, 0 -> 1
        g[s:s + r_down] = 1 - (1 - gain) * t
    if r_up > 1:
        t = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, r_up, endpoint=False))
        g[e - r_up:e] = 1 - (1 - gain) * t[::-1]
    return g

n = len(x)
env = np.ones(n)
for kind, s, e in labs:
    si, ei = int(s * sr), int(e * sr)
    if kind == "breath":
        pad = int(a.pad * sr)
        env = np.minimum(env, dip_env(n, si - pad, ei + pad, 10 ** (a.breath_db / 20), int(a.ramp * sr), int(a.ramp_up * sr)))
    elif kind == "dead":
        env = np.minimum(env, dip_env(n, si, ei, 0.0, int(a.ramp * sr), int(a.ramp_up * sr)))
    elif kind == "smack":
        env = np.minimum(env, ramp_env(n, si, ei, 10 ** (a.smack_db / 20), int(0.005 * sr)))
y = x * env[:, None]

for kind, s, e in labs:
    if kind != "plosive":
        continue
    si, ei = max(0, int(s * sr) - int(0.03 * sr)), min(n, int(e * sr) + int(0.03 * sr))
    sos = signal.butter(4, 120, "highpass", fs=sr, output="sos")
    seg = signal.sosfiltfilt(sos, y[si:ei], axis=0)
    w = np.ones(ei - si); r = min(int(0.03 * sr), (ei - si) // 2)
    w[:r] = np.linspace(0, 1, r); w[-r:] = np.linspace(1, 0, r)
    y[si:ei] = y[si:ei] * (1 - w[:, None]) + seg * w[:, None]

# dead air: cut the middle, keep gap seconds, 40 ms equal-gain crossfade at the join
cuts = []
for kind, s, e in labs:
    if kind == "dead" and (e - s) > a.gap:
        remove = (e - s) - a.gap
        mid = (s + e) / 2
        cuts.append((int((mid - remove / 2) * sr), int((mid + remove / 2) * sr)))
cuts.sort()
xf = int(0.040 * sr)
pieces, pos = [], 0
out = np.zeros((0, y.shape[1]))
for cs, ce in cuts:
    cs = max(cs, pos + xf); ce = max(ce, cs)
    head = y[pos:cs]
    pieces.append(head)
    pos = ce
pieces.append(y[pos:])
if cuts:
    res = pieces[0]
    for k, nxt in enumerate(pieces[1:]):
        m = min(xf, len(res), len(nxt))
        fade = np.linspace(0, 1, m)[:, None]
        blend = res[-m:] * np.cos(fade * np.pi / 2) + nxt[:m] * np.sin(fade * np.pi / 2)
        res = np.concatenate([res[:-m], blend, nxt[m:]])
    y = res
else:
    y = y

if dtype == np.int16:
    out = np.clip(np.round(y * 32768.0), -32768, 32767).astype(np.int16)
else:
    out = y.astype(dtype)
wavfile.write(a.out, sr, out[:, 0] if mono else out)
if a.report:
    cutdur = sum((ce - cs) for cs, ce in cuts) / sr
    cnt = {}
    for k, _, _ in labs: cnt[k] = cnt.get(k, 0) + 1
    print(f"labels {cnt} | dead air removed {cutdur:.1f} s | {n/sr:.1f} s -> {len(y)/sr:.1f} s")
