#!/usr/bin/env python3
"""Check that the browser detector (index.html) and the Python detector (dlg.py) produce the same labels.

    python3 tools/parity_check.py recording.wav [index.html]

Run it after changing any detection parameter or detector logic in either place. It opens the page in headless
Chromium (pip install playwright numpy scipy; playwright install chromium), analyzes the recording, saves the labels
with "Include unreviewed" on, runs dlg.py on the same file, and compares per kind: the counts and the largest boundary
difference. Expect identical counts and differences under about a millisecond.
"""
import os, re, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dlg
from scipy.io import wavfile
from playwright.sync_api import sync_playwright

wav = os.path.abspath(sys.argv[1])
page = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "index.html"))

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_context(accept_downloads=True).new_page()
    pg.goto("file://" + page); pg.wait_for_timeout(600)
    pg.set_input_files("#fileAudio", wav)
    pg.wait_for_function("t => document.querySelector('#audioName').textContent.includes(t)", arg=os.path.basename(wav), timeout=120000)
    pg.click("#btnAnalyze")
    pg.wait_for_function("document.querySelector('#status').textContent.startsWith('Found')", timeout=600000)
    pg.check("#inclPending")
    with pg.expect_download() as dl:
        pg.click("#btnSave")
    path = os.path.join(tempfile.mkdtemp(), "labels.txt"); dl.value.save_as(path)
    b.close()

def kind(text):
    t = text.lower()
    return "dead" if t.startswith("dead") else t.split()[0]
js = {}
for line in open(path):
    a = line.rstrip("\n").split("\t")
    if len(a) >= 3: js.setdefault(kind(a[2]), []).append((float(a[0]), float(a[1])))

sr, x = wavfile.read(wav)
if x.ndim > 1: x = x.mean(1)
x = x.astype(np.float32) / (np.iinfo(x.dtype).max + 1 if x.dtype.kind == "i" else 1.0)
py = {}
for k, s, e, _ in dlg.detect(dlg.features(x, sr)):
    py.setdefault(k, []).append((max(0.0, s), e))

ok = True
for k in ("breath", "smack", "dead"):
    a, b_ = sorted(js.get(k, [])), sorted(py.get(k, []))
    if len(a) != len(b_):
        print(f"{k:7s} browser {len(a)}  python {len(b_)}  COUNT MISMATCH"); ok = False; continue
    d = max((max(abs(p[0] - q[0]), abs(p[1] - q[1])) for p, q in zip(a, b_)), default=0.0) * 1000
    print(f"{k:7s} browser {len(a)}  python {len(b_)}  largest boundary difference {d:.2f} ms")
    ok = ok and d < 5
print("PARITY OK" if ok else "PARITY FAILED")
sys.exit(0 if ok else 1)
