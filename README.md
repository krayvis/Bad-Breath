# Bad Breath

Finds breaths, dead air and lip smacks in a voiceover or narration recording, lets you review what it flagged, and exports a cleaned WAV. Free, runs in your browser, nothing to install, nothing uploaded anywhere. Your audio never leaves your machine.

It is built for one job: quietly reduce the non-speaking parts of a read so it stays natural, and leave about a second or two of space between sentences.

**Try it: https://krayvis.github.io/Bad-Breath/**

**This is a demo, tuned to one voice.** The detection thresholds were fitted to one narrator's recordings in one room. On other voices and rooms it will miss breaths and flag the wrong things, so treat every flag as a suggestion and review before exporting. [`docs/TUNING.md`](docs/TUNING.md) explains how to tune it for your own setup, written for someone with audio analysis experience or for an AI agent doing it for you.

## The tool

Open the page above, or download `index.html` and open it in a browser. It is a single file and works offline (Firefox, Chrome and Safari).

1. **Open audio** and pick a WAV. It keeps the file's own sample rate and bit depth.
2. **Analyze** flags breaths, dead air and lip smacks. A 5-minute recording takes about 7 seconds.
3. Review them. Down arrow steps through the labels, K keeps one, D rejects it. Drag across the waveform and press B (breath) or G (dead air) to add one, or click and press S (smack) or P (plosive). Tick **Hear processed** to listen with the flagged parts silenced.
4. **Export clean WAV** applies everything except rejected labels and saves `<name>_clean.wav`.

What export does: breaths and dead air ease down to silence over 150 ms and come back up over 50 ms, with the ramps inside the label. The slow fade down matters because sentence endings trail off; the quick fade up matters because words start fast. Breaths get 20 ms of padding. Lip smacks are ducked 24 dB. Plosive labels lose their low end. Dead air longer than 1.5 s is shortened to 1.5 s by cutting the silent middle. All of these are settings in the export dialog.

**Detection settings** has sliders for breath and lip smack sensitivity and for how quiet and how long dead air must be. Moving one redoes the detection instantly. Thresholds are measured against the recording's own noise floor and speech level, so a quieter or louder recording behaves the same.

**Viewing**: scroll or pinch to zoom around the pointer, Shift+scroll or the arrow keys to pan, F zooms to the selected label, 0 fits the whole recording. The spectrogram appears when the view is under 30 s.

**Save labels** writes an Audacity-style label file. **Save review log** records every keep, reject and move, which is what the detector gets tuned against.

## Label types

- **breath**: broadband inhale between phrases. Duck it, do not remove it.
- **plosive**: low-frequency pop on p, b, t, k. Reduce the low end only, so the consonant keeps its energy. Not detected automatically; mark these by hand.
- **smack**: short click, usually just before speech starts.
- **dead air**: stretch of room tone only.

## How breaths are found

A breath is a short stretch of unpitched, fairly quiet, low-frequency noise between speech. Levels are measured in 10 ms steps (25 ms window) against two numbers measured per recording: the noise floor (5th percentile of frame level) and the speech level (median level of clearly voiced frames).

1. **Candidates:** level at least 5 dB above the floor and at least 18 dB below speech level; periodicity score below 0.55; spectral centroid under 2600 Hz. Gaps up to 60 ms are bridged.
2. **Must look like a breath:** 0.24 to 1.2 s long; at least 80 ms of a stricter version of the test (8 dB above the floor, 14 dB below speech, periodicity under 0.5, centroid under 2000 Hz); mean centroid under 2000 Hz; the 80 ms before it no louder than 45 dB above the floor.
3. **Edges:** it grows outward up to 0.4 s per side over the quiet onset and tail. It starts at the amplitude valley after the previous word's tail if that valley is at least 8 dB below the preceding frames. It ends before a sharp onset in its last 0.25 s (12 dB or more above the preceding frames and the breath's own level), such as the "t" of "to".

Smacks are short high-frequency envelope jumps of 20 dB or more in quiet context. Dead air is a stretch at least 0.4 s long where the level stays within 14 dB of the noise floor.

## Scripts

- `tools/dlg.py`: the detector, as a library and command line. `python3 tools/dlg.py in.wav outdir` writes `labels.txt`, `plosive_candidates.txt` and `events.csv`. The browser tool has the same detector built in and the two agree to within a millisecond. Needs numpy and scipy.
- `tools/clean.py`: applies a label file to a WAV without the browser. `python3 tools/clean.py in.wav labels.txt out.wav --report`. Run it on the original recording, since time stamps move after dead air is cut.
- `tools/tune.py wav review_log.csv --kind breath --grid ...`: scores the detector against your reviewed labels and searches a parameter grid, scoring the first and second halves separately to catch overfitting.
- `tools/parity_check.py wav`: checks that the browser and Python detectors produce the same labels.
- `tools/overview.py`: level and noise floor summary for a recording.

## How well does it work

The thresholds were tuned against 58 hand-made marks in the first three minutes of one narrator's recording: 35 of 35 breaths found, 12 of 12 dead air regions, 10 of 11 smacks. A second recording of the same narrator (7.5 minutes) behaves similarly, but its flags have not been reviewed by ear.

Limits:
- Tuned on one voice and one room. Other voices and rooms will need the sensitivity sliders, and may need re-tuning with `tools/tune.py` (see `docs/TUNING.md`).
- Added room noise hides lip smacks and some breaths. In one test, smack detection fell from 10 of 11 to 3.
- Plosives are not detected automatically.
- Not tested on recordings over about an hour.
- The ramp lengths and breath edges are by-ear judgments; adjust them to taste in the export dialog.

The way to improve it is reviewed recordings: save a review log (**Save review log**) from your own recording and tune against it, or open an issue with what the detector got wrong. Only share a review log if you are happy for the timestamps to be public; it contains no audio.

## Privacy and terms

**Your audio stays on your computer.** The app runs entirely in your browser. A recording you open is read from your disk into the page's memory, analyzed there, and the cleaned file is made there and saved by your browser. Nothing is uploaded. There is no server, account, analytics or tracking in this project. This was checked by recording every network request while opening a recording, analyzing it and exporting it: after the page loads there are none.

**What the page does load.** The page itself, from GitHub Pages (GitHub keeps ordinary web server logs; see GitHub's privacy statement), and two fonts from Google Fonts, so Google sees your IP address and browser details like any site that uses its fonts. Nothing else. A downloaded `index.html` works offline and falls back to your system fonts.

**What is saved in your browser.** Your export and detection settings, and a draft of your label decisions (label times and the recording's file name, never any audio), kept in this site's local storage so a reload doesn't lose your review. Clear it with your browser's site data settings.

**Terms.** Free software under the AGPL-3.0, provided "as is" with no warranty (sections 15 and 16 of the license). The tool never changes your original file; export saves a new one. Keep your originals and listen to the result before you publish it. The detection is tuned to one voice (see above), so it will make mistakes on others.

## License

GNU Affero General Public License v3 (AGPL-3.0-only), copyright Winston Moy. See `LICENSE`.

In short: you are free to use, study, modify and share it. If you distribute a modified version, or let people use one over a network (for example hosting a changed copy of the page), you must offer them the complete source of your version under the same license. You may not make it proprietary or relicense it more restrictively. The license does not ban selling copies, but anyone who sells one must give the buyer the source and the same freedoms, so it stays free to everyone. This is a summary, not legal advice; the license text governs.
