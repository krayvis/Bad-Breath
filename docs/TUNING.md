# Tuning Bad Breath for your own recordings

The detector is a set of threshold rules over a few audio features, not a trained model. The defaults were fitted to one narrator in one room, so on your voice and your microphone expect to adjust them. This guide is for two readers: someone comfortable with audio analysis, and an AI agent doing the tuning for a person. It is written so either can follow it without asking questions.

The result of following it: new default values in `DETECT` (in `index.html`) and `DEFAULTS` (in `tools/dlg.py`), checked against recordings you reviewed by hand, with the browser and Python detectors still agreeing.

## How the pieces fit

| File | Role |
| --- | --- |
| `index.html` | The app. Contains the browser detector (`DETECT`, `detectFeatures`, `detectBreaths`, `detectSmacks`, `detectDead`, `sensParams`) and the renderer that applies labels. |
| `tools/dlg.py` | The same detector in Python (`DEFAULTS`, `features`, `detect_*`). This is where you experiment. |
| `tools/tune.py` | Scores the detector against a review log and searches a parameter grid. |
| `tools/parity_check.py` | Proves `index.html` and `dlg.py` produce the same labels on a recording. |
| `tools/clean.py` | Applies a label file to a WAV from the command line. |
| `tools/overview.py` | Prints level statistics for a recording. |

The loop is: label by hand in the app, save the review log, tune in Python, copy the new values into the browser, check parity, listen.

## Setup

```
pip install numpy scipy playwright
playwright install chromium     # only needed for parity_check.py
```

## Step 1: Know your recording

Open it in the app and press **Analyze**. The Detection settings bar shows the measured noise floor and speech level. Everything is relative to those two numbers:

- **Noise floor**: the 5th percentile of frame level, ignoring digital silence (below -85 dB).
- **Speech level** (`vref`): the median level of voiced frames (periodicity above 0.6 and at least 25 dB above the floor).

The two recordings the defaults were tuned on have a speech level 45 to 48 dB above the floor. The smaller that gap (noisy room, quiet voice), the less room there is between a breath or click and the noise, and detection gets harder. I have not tested below about 40 dB, so treat a small gap as a warning. Fix the recording first if you can. `python3 tools/overview.py recording.wav` prints percentiles of frame level for a second opinion.

## Step 2: Collect ground truth

The detector can only be as good as the marks it is checked against.

1. Open the recording in the app, press **Analyze**, and review a stretch from the beginning. Work through the flags with the down arrow: **K** keeps one, **D** rejects it. Fix boundaries by dragging the label edges. Add what it missed: drag across the waveform and press **B** (breath) or **G** (dead air), or place the playhead and press **S** (smack).
2. Review a contiguous stretch from the start of the file and be thorough in it. Three minutes (about 30 breaths) is a floor; ten is better. Unreviewed flags are ignored by the scorer, but a breath you did not mark inside the reviewed stretch counts against precision, so do not stop halfway through a section.
3. Press **Save review log**. Keep the CSV next to the recording.
4. Do this for at least two recordings from different days or rooms. Tune on one, hold the other out to test.

The review log is a CSV with one row per label:

```
type,start,end,duration,text,decision,added,moved,orig_start,orig_end
breath,10.0299,10.3696,0.3397,"breath",keep,1,0,10.0299,10.3696
```

- `type`: `breath`, `plosive`, `smack`, `dead` or `other`.
- `start`, `end`: final boundaries in seconds.
- `decision`: `keep`, `reject` or `unreviewed`.
- `added`: 1 if you drew it yourself. `moved`: 1 if you changed a detector label's edges. `orig_start`, `orig_end` are the detector's original edges.

The ground truth for a kind is every row of that type with `decision=keep` or `added=1`. Rows with `decision=reject` are known false positives.

## Step 3: Score and search

```
python3 tools/tune.py recording.wav review_log.csv --kind breath
```

prints the defaults' score on the first half of the reviewed stretch (train) and the second half (test). Then give it a grid:

```
python3 tools/tune.py recording.wav review_log.csv --kind breath \
    --grid br_top=14,18,22 br_min=0.20,0.24,0.30 br_harm=0.5,0.55,0.6 --json best.json
```

It scores every combination, lists the best by train score, and shows each one's test score. A test score clearly below train means the grid is overfit; narrow it, or review more audio. Run `--kind smack` and `--kind dead` the same way.

What the numbers mean:

- A detection **matches** a mark when they overlap. **Recall** is the share of your marks matched. **Precision** is matched detections over all detections in the reviewed stretch; a detection that matches no mark (including one you rejected) is a false positive. **F** combines them (`--beta 2` favors recall, which is usually right here because a false flag costs a click to reject and a miss costs a breath left in).
- **IoU** is the overlap of matched pairs. It shows how well the edges line up, which recall does not.
- **start / end offsets** are detector minus you in milliseconds, medians over matched pairs. A consistent +40 ms on the end means the detector runs late; shift it with a parameter, not by hand.

Tune one kind at a time and one group of parameters at a time, three or four values each. The grid grows fast and a wide grid on thirty marks will overfit.

## Parameter reference

All levels are in dB. "Floor" is the noise floor and "speech" is the speech level measured per recording, so thresholds expressed relative to them carry over between recordings of different loudness. Names are identical in `DEFAULTS` (Python) and `DETECT` (browser).

### Breaths

A breath is a run of frames that is quiet, unpitched and dull, then trimmed at its edges.

| Parameter | Default | What it does |
| --- | --- | --- |
| `br_loose` | 5 | Candidate frames must be at least this far above the floor. |
| `br_top` | 18 | Candidate frames must be at least this far below speech. Lower it to admit louder breaths. |
| `br_harm` | 0.55 | Candidate frames must have periodicity below this (0 is noise, 1 is a steady pitch). |
| `br_cent_loose` | 2600 | Candidate frames must have a spectral centroid below this, in Hz. |
| `br_gap` | 3 | Gaps up to twice this many frames (10 ms each) inside a candidate run are bridged. |
| `br_cent` | 2000 | Centroid ceiling for the run's seed frames and for its mean. |
| `br_seeds` | 8 | A run needs this many seed frames. Seeds are stricter: floor +8 dB, speech -14 dB, periodicity under 0.5, centroid under `br_cent`. The +8, -14 and 0.5 are fixed in the code. |
| `br_min`, `br_max` | 0.24, 1.2 | Allowed run length in seconds. |
| `br_pre` | 45 | The 80 ms before the run may not be louder than floor plus this. Stops breaths from starting inside a loud word. |
| `br_grow` | 0.40 | Seconds a breath may grow outward on each side to cover its quiet onset and tail. |
| `br_grow_lo` | -1 | Growth continues while level stays above floor plus this. |
| `br_grow_hi` | -24.6 | Growth continues while level stays below speech plus this. |
| `br_grow_miss` | 6 | Frames of failing the growth test that are tolerated before it stops. |
| `br_valley`, `br_valley_drop`, `br_valley_keep` | 0.30, 8, 0.15 | Start the breath at the lowest point of its first `br_valley` seconds, if that is at least `br_valley_drop` dB below the loudest earlier frame and at least `br_valley_keep` seconds of breath remain. Keeps the trailing "t" or "s" of the previous word out. |
| `br_burst`, `br_burst_over`, `br_endwin`, `br_burst_keep` | 12, 12, 0.25, 0.15 | End the breath before a sharp onset in its last `br_endwin` seconds: a frame at least `br_burst` dB above the mean of the five before it and `br_burst_over` dB above the breath's median level. At least `br_burst_keep` seconds must remain. This is the "t" of "to". |
| `br_trim_start`, `br_trim_end` | 0, 0 | Python only. Shift the label edges by this many seconds after everything else. |

### Lip smacks

A smack is a short jump in the high-frequency envelope (above 1.5 kHz, 1 ms steps) over its rolling median, in a quiet context.

| Parameter | Default | What it does |
| --- | --- | --- |
| `sm_rise` | 20 | Minimum jump above the rolling median. |
| `sm_level` | 20 | The high-frequency envelope must be at least this far above the floor. |
| `sm_ctx` | 30 | The 80 ms either side may not be louder than floor plus this. Stops it from flagging consonants inside speech. |
| `sm_inside_rise` | 22 | Inside a breath, a smack needs at least this much jump. |

### Dead air

| Parameter | Default | What it does |
| --- | --- | --- |
| `da_lvl` | 14 | Frames below floor plus this count as quiet. |
| `da_min` | 0.4 | Shortest stretch to flag, in seconds. |
| `da_in_start`, `da_in_end` | 0.10, 0.01 | Seconds trimmed off each end of the stretch so speech and its decay are untouched. |

### Plosives (Python only)

`pl_rise` (23) and `pl_band` (15, 55 Hz) find low-frequency pops. They were never tuned because no reviewed plosives existed, so they produce candidates only. `tools/dlg.py` writes them to `plosive_candidates.txt`. Mark plosives by hand in the app; if you collect enough, tune these the same way and port the detector to the browser.

### Fixed constants

These are not parameters; change them in code if you need to. 10 ms hop and 25 ms window for level, periodicity and centroid. Periodicity is the largest normalized autocorrelation over lags for 70 to 400 Hz. Centroid is over 300 to 6000 Hz. Stereo is averaged to mono. Label times are the frame center minus or plus 12.5 ms.

### The sensitivity sliders

The sliders in the app move several parameters together (`sensParams` in `index.html`). With breath sensitivity `b` (steps from -2 to +2): `br_min` is 0.24 - 0.03b, `br_seeds` is max(3, 8 - 1.5b) rounded, `br_cent` is 2000 + 150b, `br_cent_loose` is 2600 + 200b and `br_harm` is 0.55 + 0.03b. With smack sensitivity `s`: `sm_rise` and `sm_level` are both 20 - 3s. The dead air sliders set `da_lvl` and `da_min` directly. If your tuned defaults make the sliders feel off, adjust `sensParams`.

## Step 4: Move the values into the browser

1. Put the winning values in `DEFAULTS` in `tools/dlg.py` and in `DETECT` in `index.html`. Use the same numbers; the names match.
2. Run the parity check on every recording you tuned on:
   ```
   python3 tools/parity_check.py recording.wav
   ```
   It must print `PARITY OK`: the same number of labels per kind and boundaries within a millisecond. A mismatch means the two detectors differ somewhere (a parameter only changed in one place, or logic was edited in one of them).
3. Run `tune.py` once more on the held-out recording with no grid to confirm the new defaults hold up there.

## Step 5: Listen

Scores say where the edges are, not whether it sounds right. In the app, tick **Hear processed** and step through flagged breaths with the down arrow. Listen for a breath cut that clips the end of a word, a trailing "s" or "t" swallowed, and a dead air cut that makes the pause too tight. The export dialog has the ramp lengths (down 150 ms, up 50 ms by default), the breath padding and the gap length; those are taste, not detection, and are separate from everything above.

## Notes for an AI agent doing this for a person

- You need the person's help for ground truth. Ask them to label in the app and send back the review log and the WAV (or paths to both). Do not tune against unreviewed detector output; agreeing with yourself proves nothing.
- Do not change the feature definitions (hop, window, bands, percentile choices) until you have exhausted the parameters. They are shared between the Python and browser detectors and every change has to be made in both.
- Keep a train and a test split by time, and a held-out recording if there is one. Report both scores. Do not report a train score alone.
- Prefer recall over precision when they conflict: a false flag costs the person one keypress to reject, a miss costs a breath in the final audio. Say which you optimized.
- Thresholds that are relative to floor or speech should stay relative. Absolute dB thresholds break when the same voice is recorded louder or quieter.
- Expect noise to be the main failure. Added room noise hides lip smacks (the clicks vanish under high-frequency noise) and thins breath recall. If the person's room is noisy, try the sensitivity sliders first, then `sm_rise`, `sm_level` and `br_loose` in the grid, and say plainly if the recording is too noisy for the rules to work.
- After changing anything, run `parity_check.py` and `tune.py` with no grid on every reviewed recording, and show the person the before and after numbers.
- Ask the person to listen to a few of the changed flags with **Hear processed** before calling it done. You cannot hear the result.
- Do not publish or send their audio anywhere. The app is built so audio never leaves the browser; keep it that way.

## Going further

If threshold rules stop being enough, the review logs are a labeled dataset: `dlg.features(x, sr)` returns per-frame level, periodicity, centroid and flatness at 10 ms steps, and the log gives the frames that are breath. A small classifier over those (logistic regression or a shallow tree) would be a natural next step. The constraint to keep is that the browser has to run it without a server, so ship it as a few numbers in a JSON object or a handful of lines of code rather than a framework.
