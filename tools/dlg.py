"""Dialogue cleanup detector, version 2 (tuned on 3 minutes of reviewed labels).

Library use:   f = features(x, sr);  ev = detect(f, params)
Command line:  python3 dlg.py in.wav outdir      -> labels.txt (breath, smack, dead air), plosive_candidates.txt, events.csv
"""
import sys, os, csv
import numpy as np
from scipy import signal, ndimage
from scipy.io import wavfile

HOP_S, WIN_S = 0.010, 0.025

DEFAULTS = dict(
    # breaths: grown from seed frames, long enough, dark enough, quiet before
    br_loose=5, br_top=18, br_harm=0.55, br_cent_loose=2600, br_gap=3, br_cent=2000,
    br_burst=12, br_burst_over=12, br_endwin=0.25, br_burst_keep=0.15, br_trim_start=0.0, br_trim_end=0.0, br_valley=0.30, br_valley_drop=8, br_valley_keep=0.15, br_grow=0.40, br_grow_lo=-1, br_grow_hi=-24.6, br_grow_miss=6, br_seeds=8, br_min=0.24, br_max=1.2, br_pre=45,
    # smacks: strong short HF click in quiet context
    sm_rise=20, sm_level=20, sm_ctx=30, sm_inside_rise=22,
    # dead air: quiet stretches (room tone plus faint noise)
    da_lvl=14, da_min=0.4, da_in_start=0.10, da_in_end=0.01,
    # plosives (no reviewed examples yet): sub-55 Hz blast
    pl_rise=23, pl_band=(15, 55),
)


def frames(sig, w, h):
    n = 1 + (len(sig) - w) // h
    return np.lib.stride_tricks.as_strided(sig, shape=(n, w), strides=(sig.strides[0] * h, sig.strides[0]), writeable=False)


def features(x, sr):
    hop, win = int(HOP_S * sr + 0.5), int(WIN_S * sr)
    F = frames(x, win, hop)
    n = len(F)
    rms = np.empty(n); harm = np.empty(n); flat = np.empty(n); cent = np.empty(n)
    w = np.hanning(win).astype(np.float32)
    nfft = 4096
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    band = (freqs >= 300) & (freqs <= 6000)
    lagmin, lagmax = int(sr / 400), int(sr / 70)
    for s in range(0, n, 2000):
        blk = F[s:s + 2000]; e = s + len(blk)
        rms[s:e] = 20 * np.log10(np.sqrt((blk ** 2).mean(1)) + 1e-9)
        X = np.fft.rfft(blk * w, nfft, axis=1)
        P = X.real ** 2 + X.imag ** 2
        ac = np.fft.irfft(P, nfft, axis=1)
        harm[s:e] = ac[:, lagmin:lagmax].max(1) / (ac[:, 0] + 1e-12)
        Pb = P[:, band] + 1e-20
        flat[s:e] = np.exp(np.log(Pb).mean(1)) / Pb.mean(1)
        cent[s:e] = (Pb * freqs[band]).sum(1) / Pb.sum(1)
    live = rms[rms > -85]   # digital silence would otherwise drag the floor to -180 dB
    floor = max(-80.0, np.percentile(live, 5)) if len(live) > 20 else -80.0
    voiced = (harm > 0.6) & (rms > floor + 25)
    vref = np.percentile(rms[voiced], 50) if voiced.any() else floor + 40
    # 1 ms high-frequency envelope for clicks
    hs = signal.butter(4, 1500, "highpass", fs=sr, output="sos")
    hf = signal.sosfiltfilt(hs, x.astype(np.float64)).astype(np.float32)
    fh = max(1, int(round(0.001 * sr))); fw = 2 * fh   # 1 ms hop, 2 ms window, whatever the sample rate
    HF = frames(hf, fw, fh)
    env = 20 * np.log10(np.sqrt((HF ** 2).mean(1)) + 1e-9)
    env_rise = env - ndimage.median_filter(env, size=101, mode="nearest")
    return dict(x=x, sr=sr, hop=hop, win=win, n=n, rms=rms, harm=harm, flat=flat, cent=cent, floor=floor, vref=vref,
                env=env, env_rise=env_rise, fh=fh, fw=fw)


def _t(f, frame):
    return (frame * f["hop"] + f["win"] / 2) / f["sr"]


def _segments(mask, min_len, max_len, merge_gap):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    segs, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p - 1 > merge_gap:
            segs.append((s, p)); s = i
        p = i
    segs.append((s, p))
    return [(a, b) for a, b in segs if min_len <= (b - a + 1) <= max_len]


def _grow(rms, a, b, P, floor, vref):
    """Extend a detected breath outward over the quiet onset and tail that the strict rule misses.
    Walks while level stays between floor+lo and grow_hi (below speech), bridging short misses."""
    ok = (rms > floor + P["br_grow_lo"]) & (rms < vref + P["br_grow_hi"])
    maxf = int(P["br_grow"] / HOP_S)
    for direction in (-1, 1):
        edge, miss, k = (a if direction < 0 else b), 0, 0
        last_good = edge
        while k < maxf:
            k += 1
            i = (a if direction < 0 else b) + direction * k
            if i < 0 or i >= len(rms):
                break
            if ok[i]:
                last_good, miss = i, 0
            else:
                miss += 1
                if miss > P["br_grow_miss"]:
                    break
        if direction < 0:
            a = min(a, last_good)
        else:
            b = max(b, last_good)
    return a, b


def _end_burst(rms, a, b, P):
    """End a breath just before a sharp onset at its tail (the 't' of 'to', the start of the next word).
    A burst is a frame at least br_burst dB above the mean of the 5 frames before it AND at least br_burst_over dB above
    the breath's median level, within the last br_endwin seconds. Returns the new last frame."""
    body = float(np.median(rms[a:b + 1]))
    for i in range(max(a + 3, b - int(P["br_endwin"] / HOP_S)), b + 1):
        base = rms[max(a, i - 5):i]
        if rms[i] - base.mean() >= P["br_burst"] and rms[i] - body >= P["br_burst_over"]:
            return i - 1 if (i - 1 - a) * HOP_S >= P["br_burst_keep"] else b
    return b


def _valley_start(rms, a, b, P):
    """Move the start of a breath to the amplitude valley after the word's tail.
    Looks at the first br_valley seconds; if the lowest smoothed point sits at least br_valley_drop dB
    below the loudest frame before it, the breath starts at that minimum, as long as enough breath remains."""
    n = int(P["br_valley"] / HOP_S)
    hi = min(b, a + n)
    if hi - a < 4:
        return a
    sm = ndimage.median_filter(rms[a:hi + 1], size=3, mode="nearest")
    m = int(np.argmin(sm))
    if m < 2:
        return a
    if rms[a:a + m].max() - sm[m] < P["br_valley_drop"]:
        return a
    if (b - (a + m)) * HOP_S < P["br_valley_keep"]:
        return a
    return a + m


def detect_breaths(f, P):
    rms, harm, cent, floor, vref = f["rms"], f["harm"], f["cent"], f["floor"], f["vref"]
    loose = (rms > floor + P["br_loose"]) & (rms < vref - P["br_top"]) & (harm < P["br_harm"]) & (cent < P["br_cent_loose"])
    seed = (rms > floor + 8) & (rms < vref - 14) & (harm < 0.5) & (cent < P["br_cent"])
    loose = ndimage.binary_closing(loose, structure=np.ones(2 * P["br_gap"] + 1, bool))
    lab, nl = ndimage.label(loose)
    out = []
    for sl in ndimage.find_objects(lab):
        a, b = sl[0].start, sl[0].stop - 1
        dur = (b - a + 1) * HOP_S
        if not (P["br_min"] <= dur <= P["br_max"]):
            continue
        if seed[a:b + 1].sum() < P["br_seeds"]:
            continue
        if cent[a:b + 1].mean() > P["br_cent"]:
            continue
        pre = rms[max(0, a - 8):a].mean() if a > 0 else floor
        if pre > floor + P["br_pre"]:
            continue
        a, b = _grow(rms, a, b, P, floor, vref)
        b = _end_burst(rms, a, b, P)
        a = _valley_start(rms, a, b, P)
        out.append(("breath", _t(f, a) - 0.0125 + P["br_trim_start"], _t(f, b) + 0.0125 - P["br_trim_end"],
                    dict(level=float(rms[a:b + 1].mean()), cent=float(cent[a:b + 1].mean()), pre=float(pre), dur=dur)))
    return out


def detect_smacks(f, P, breaths):
    env, rise, fh, fw, sr, rms, floor = f["env"], f["env_rise"], f["fh"], f["fw"], f["sr"], f["rms"], f["floor"]
    sc = (rise > P["sm_rise"]) & (env > floor + P["sm_level"])
    out = []
    for a, b in _segments(sc, 2, 40, 5):
        ts, te = (a * fh + fw / 2) / sr, (b * fh + fw / 2) / sr
        fa, fb = int(ts / HOP_S), int(te / HOP_S)
        around = np.concatenate([rms[max(0, fa - 8):max(0, fa - 1)], rms[fb + 2:fb + 9]])
        if len(around) == 0 or around.max() > floor + P["sm_ctx"]:
            continue
        r = float(rise[a:b + 1].max())
        inside = any(bs - 0.02 <= ts <= be + 0.02 for _, bs, be, _ in breaths)
        if inside and r < P["sm_inside_rise"]:
            continue
        out.append(("smack", ts - 0.005, te + 0.01, dict(rise=r, level=float(env[a:b + 1].max()), around=float(around.max()))))
    return out


def detect_dead(f, P):
    rms, floor = f["rms"], f["floor"]
    quiet = rms < floor + P["da_lvl"]
    out = []
    for a, b in _segments(quiet, int(P["da_min"] / HOP_S), 10 ** 7, 5):
        s, e = _t(f, a) + P["da_in_start"], _t(f, b) - P["da_in_end"]
        if e - s < P["da_min"] - 0.1:
            continue
        out.append(("dead", s, e, dict(level=float(rms[a:b + 1].mean()), dur=e - s)))
    return out


def detect_plosives(f, P):
    sr, x, rms, floor, hop, win = f["sr"], f["x"], f["rms"], f["floor"], f["hop"], f["win"]
    sos = signal.butter(6, list(P["pl_band"]), "bandpass", fs=sr, output="sos")
    lf = signal.sosfiltfilt(sos, x.astype(np.float64)).astype(np.float32)
    LF = frames(lf, win, hop)
    lf_db = 20 * np.log10(np.sqrt((LF ** 2).mean(1)) + 1e-9)
    base = ndimage.median_filter(lf_db, size=101, mode="nearest")
    ratio = lf_db - rms
    pc = (lf_db - base > P["pl_rise"]) & (ratio > -22) & (rms > floor + 15)
    out = []
    for a, b in _segments(pc, 3, 25, 2):
        out.append(("plosive", _t(f, a) - 0.02, _t(f, b) + 0.03, dict(lf_rise=float((lf_db - base)[a:b + 1].max()))))
    return out


def detect(f, P=None, plosives=False):
    P = {**DEFAULTS, **(P or {})}
    br = detect_breaths(f, P)
    ev = br + detect_smacks(f, P, br) + detect_dead(f, P)
    if plosives:
        ev += detect_plosives(f, P)
    ev.sort(key=lambda e: e[1])
    return ev


def label_text(k, s, e, ft):
    if k == "plosive":
        return f"plosive {ft['lf_rise']:.0f}dB"
    if k == "smack":
        return f"smack {ft['rise']:.0f}dB"
    if k == "dead":
        return f"dead air {e - s:.1f}s"
    return f"breath {int(round((e - s) * 1000))}ms"


def write_labels(path, events):
    with open(path, "w") as fh:
        for k, s, e, ft in events:
            fh.write(f"{max(0, s):.6f}\t{e:.6f}\t{label_text(k, s, e, ft)}\n")


if __name__ == "__main__":
    wav, outdir = sys.argv[1:3]
    os.makedirs(outdir, exist_ok=True)
    sr, x = wavfile.read(wav)
    x = x.astype(np.float32) / 32768.0
    f = features(x, sr)
    print(f"floor {f['floor']:.1f} dB | voiced ref {f['vref']:.1f} dB")
    ev = detect(f, plosives=True)
    main = [e for e in ev if e[0] != "plosive"]
    write_labels(os.path.join(outdir, "labels.txt"), main)
    write_labels(os.path.join(outdir, "plosive_candidates.txt"), [e for e in ev if e[0] == "plosive"])
    with open(os.path.join(outdir, "events.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["kind", "start", "end", "features"])
        for k, s, e, ft in ev:
            w.writerow([k, f"{s:.4f}", f"{e:.4f}", {a: round(float(b), 2) for a, b in ft.items()}])
    cnt = {}
    for e in ev:
        cnt[e[0]] = cnt.get(e[0], 0) + 1
    print("events:", cnt)
