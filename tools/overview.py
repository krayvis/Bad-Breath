# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Winston Moy. See LICENSE.
import sys
import numpy as np
from scipy.io import wavfile

path = sys.argv[1]
sr, x = wavfile.read(path)
x = x.astype(np.float32) / 32768.0
print("sr", sr, "samples", len(x), "dur_s", round(len(x) / sr, 1))
print("peak dBFS", round(20 * np.log10(np.max(np.abs(x)) + 1e-9), 1))

win, hop = int(0.025 * sr), int(0.010 * sr)
n = 1 + (len(x) - win) // hop
rms = np.array([np.sqrt(np.mean(x[i * hop:i * hop + win] ** 2)) for i in range(n)])
db = 20 * np.log10(rms + 1e-9)
for p in (5, 10, 25, 50, 75, 90, 99):
    print(f"frame RMS p{p}: {np.percentile(db, p):.1f} dBFS")

floor = np.percentile(db, 10)
speech = db > (floor + 15)
print("est noise floor dBFS", round(floor, 1), "| frames above floor+15dB:", round(100 * speech.mean(), 1), "%")

# run-length of silence/pauses (frames below threshold)
runs, cur = [], 0
for s in speech:
    if not s:
        cur += 1
    else:
        if cur:
            runs.append(cur * 0.010)
        cur = 0
runs = np.array(runs)
print("pauses:", len(runs), "| >150ms:", int((runs > 0.15).sum()), "| >500ms:", int((runs > 0.5).sum()))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(n) * 0.010
    fig, ax = plt.subplots(2, 1, figsize=(14, 6))
    ax[0].plot(t, db, lw=0.5)
    ax[0].axhline(floor + 15, color="r", lw=0.6)
    ax[0].set_ylabel("frame RMS dBFS")
    ax[0].set_xlim(0, t[-1])
    seg = x[: sr * 20]
    ax[1].specgram(seg, NFFT=1024, Fs=sr, noverlap=768, cmap="magma", vmin=-120, vmax=-30)
    ax[1].set_ylim(0, 10000)
    ax[1].set_title("first 20 s spectrogram")
    fig.tight_layout()
    fig.savefig(out_png, dpi=80)
    print("wrote", out_png)
except Exception as e:
    print("plot skipped:", e)
