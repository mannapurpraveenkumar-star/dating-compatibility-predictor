# THE DATING GAME v2.0
*Beauty Is in the Eye of the Beholder*

An academic machine learning project. Upload a group photo and the app:

1. validates the image (JPEG, PNG or WebP, up to 10 MB and 40 megapixels),
2. detects every face with the pretrained **YuNet** CNN detector,
3. reads each person's features in the background: apparent age, expression,
   outfit colour, height in the photo, head tilt and position,
4. scores every possible pair, n × (n − 1) / 2 of them, on **similarity**
   (alike in age, height, outfit, expression) and **togetherness**
   (close, side by side, leaning in),
5. shows **the most likely pairs**, picked best-first without using anyone twice (a photo with two couples gives two pairs), each with a match score out of 100,
6. lets you download a report and start again.

## How the match score works

| Part | Feature | Weight |
|---|---|---|
| Similarity | Estimated age (same age = full marks, 15+ years apart = zero) | 40 |
| Similarity | Height in the photo (only when both faces are at a similar distance) | 5 |
| Similarity | Outfit colour (CIELAB colour distance) | 10 |
| Togetherness | Distance apart (full marks at 1.2 face-widths or closer, zero at 4) | 20 |
| Togetherness | Next to each other among the adults (children in between don't count) | 10 |
| Togetherness | Head tilted towards the other person | 10 |

Age carries the most weight because partners' ages are strongly correlated in
population studies (around r ≈ 0.8), far more than height (around r ≈ 0.2).
Smiles are measured and logged but **not scored**: almost everyone smiles in a
posed photo, so how big a smile is says nothing about a relationship.

**Man–woman rule.** Pairs must be one man and one woman by *apparent* gender
(from the InsightFace model). Testing showed why: without it, two young women in
matching white tops were paired over the real couples next to them. The rule
applies only when the model is at least 85% sure about **both** people. In
testing it read one grandfather as female with 70% confidence, so he was
correctly not blocked. Turn the rule off with `GENDER_RULE_ENABLED = False` in
`app\config.py`. It means same-sex couples are never predicted.

The score is the weighted average of the features that could be measured. If a
feature can't be measured (e.g. no outfit visible), it is skipped and the other
weights are rescaled. Pairs are then chosen **jointly**: the set of
non-overlapping pairs with the highest total of (score − cut-off) wins. Choosing
jointly matters in family photos, where grabbing the single best pair first
can leave the others wrongly paired, and subtracting the cut-off means a strong pair is
never broken up to make two weak ones. Only pairs at or above the cut-off count; if
none do, the app reports "No likely pair found".

The cut-off is **35**. Levels: 75+ Strong, 60–74 Good, 50–59 Possible, 35–49 Weak.
It was 50 until the man–woman rule removed the main source of false matches.

The similarity idea comes from *assortative mating*: partners tend to resemble
each other in traits such as age and height. The weights are a documented
starting point, **not learned from data**. `app/research.py` and
`research/README.md` show how to test them against consented, labelled photos.

### Check ages (your answers override the models)
Under the result there's a **Check ages** panel with every face and an
**Adult / Under 18** choice. Your answers always win: anyone you mark under 18 is
never matched, and anyone you mark adult is matched using their estimated age.
When faces are too small for the models (under 40 px or blurry), the panel
opens automatically and the app asks you instead of guessing. In testing,
enhancing a 239 px photo (4× upscale, denoise, sharpen) still rated a young boy
as 25, so enhancement can't replace this check. The API takes the same answers as
optional form fields: `adults=Person A,Person C` and `children=Person B`.

### Safeguards
- **Skin colour is never measured or used.** Matching on it would amount to matching by ethnicity.
- **People who may be under 18 are never matched.** Someone counts as an adult only if (1) their face is at least 40 px wide and sharp enough to read, (2) their combined age estimate (average of the InsightFace regressor and the age-band model) is 21 or more, **and** (3) the age-band model's average chance of "age 12 or under" across 4 views is below 35%. Otherwise they're left out. Testing showed why two models are needed: the regressor read young children as 23–28, and the band model read adults 10–20 years too young. Each catches the other's mistakes. Faces under 40 px are refused outright, and so are faces too blurry to read (sharpness under 90). This also stops someone enlarging a small thumbnail to get past the size check.
- **No identification.** No face recognition, no names.
- **Honest output.** The result is labelled as an experimental prediction, not proof of a relationship.

---

## 1. Setup on Windows (PowerShell, no execution-policy change needed)

All commands run from the project folder and call the venv's Python directly,
so `Activate.ps1` is never used.

```powershell
cd F:\THE_DATING_GAME

# Only if .venv does not exist yet:
py -3.11 -m venv .venv

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Check the environment
```powershell
.\.venv\Scripts\python.exe scripts\diagnose_env.py
```
Expected ending: `Everything looks good.` If it reports problems, it prints
the exact commands to fix them.

### Troubleshooting OpenCV (`module 'cv2' has no attribute 'CascadeClassifier'`)
That error almost always has one of these causes. `diagnose_env.py` checks all of them.

| Cause | Fix |
|---|---|
| Two or more OpenCV packages installed (`opencv-python`, `opencv-contrib-python`, `opencv-python-headless`) overwrite each other's `cv2` folder | Uninstall all of them, then reinstall one (below) |
| OpenCV **5.x** is installed: `CascadeClassifier` moved to the contrib modules and Haar XML files are no longer bundled | Use the pinned 4.10 package (below). This app doesn't use cascades anyway |
| A file or folder named `cv2.py` / `cv2` in your project hides the real package | Rename it |

Clean reinstall:
```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless
.\.venv\Scripts\python.exe -m pip install --no-cache-dir opencv-python-headless==4.10.0.84
.\.venv\Scripts\python.exe -c "import cv2; print(cv2.__version__, cv2.__file__, hasattr(cv2, 'FaceDetectorYN'))"
```
Expected: `4.10.0 F:\THE_DATING_GAME\.venv\Lib\site-packages\cv2\__init__.py True`

## 2. Run the website

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```
Open **http://127.0.0.1:8000** in your browser. Stop the server with `Ctrl+C`.

Health check: http://127.0.0.1:8000/api/health should show `"status": "ok"` and `"detector_ready": true`.
API docs (from FastAPI): http://127.0.0.1:8000/docs

## 3. Testing

### Automatic
```powershell
.\.venv\Scripts\python.exe -m pytest
```
107 tests covering valid group photos in all three formats, invalid and corrupted
files, unsupported formats, empty uploads, no faces, one face, many faces,
left-to-right ordering, pair counts, duplicate-pair prevention, file-size and
pixel limits (including a decompression-bomb header), detector failures, API
error handling, report generation, feature reading, pair scoring, the under-18
safeguard and the research harness.

### Manual (browser)
1. Open the site. The upload box should say "JPEG, PNG or WebP, up to 10 MB".
2. Drag `tests\fixtures\group_four.jpg` onto the box (or press it and choose the file). A preview appears.
3. Press **Analyse Photo**. Expect "4 faces found, 6 pairs compared" and **Person B & Person C, 74/100, Good match**, highlighted in pink on the photo.
4. Press **Download report**. Open the `.html` file. It shows the predicted pair, the photo and the limitations.
5. Press **Upload another photo**, choose `tests\fixtures\single_face.jpg`. Expect "No pair predicted".
6. Rename any `.txt` file to `.jpg` and upload it. Expect a red message about an unsupported file type.
7. Try a real group photo of people who agreed to it.
8. Press Tab through the page to check every control can be reached and shows a blue focus ring.
9. Narrow the window to phone width. The layout stacks.

## 4. Project structure

```
THE_DATING_GAME/
├── app/
│   ├── __init__.py      version
│   ├── config.py        all limits and thresholds in one place
│   ├── main.py          FastAPI routes: /, /api/health, /api/analyse
│   ├── vision.py        validation, YuNet detection, crops, annotation
│   ├── attributes.py    per-person features: age, expression, outfit colour
│   ├── matching.py      pair scoring and the single best-pair prediction
│   ├── analysis.py      runs the pipeline and builds the result
│   ├── report.py        Markdown and self-contained HTML report
│   └── research.py      offline experiment harness (not used by the website)
├── models/              YuNet, age and expression ONNX models + notes
├── research/            protocol and CSV template for consented data
├── scripts/
│   ├── diagnose_env.py  environment and OpenCV diagnostics
│   └── download_model.py
├── static/              index.html, style.css, script.js, favicon.svg
├── tests/               pytest suite + fixtures
├── pytest.ini
├── requirements.txt
└── README.md
```
Changes from the suggested structure, and why:
- **config.py**: limits used by the code and tests live in one place.
- **report.py**: keeps report formatting separate from analysis so each can be tested on its own.
- **research.py + research/**: the ML research component the brief asks for, kept out of the website.
- **models/**: the detector needs a model file. It ships with the project, so no download step is needed.
- **scripts/**: diagnostics for the OpenCV error you hit.
- Extra test files (`test_api.py`, `test_report.py`, `test_research.py`) cover each module separately.

## 5. API

`POST /api/analyse`: multipart form with a field named `photo`.

| Status | Code | Meaning |
|---|---|---|
| 200 | | Analysis result (also for zero faces) |
| 400 | `empty_file`, `missing_file`, `bad_request` | Nothing usable was sent |
| 413 | `file_too_large`, `too_many_pixels` | Over 10 MB or 40 MP |
| 415 | `unsupported_format` | Not JPEG/PNG/WebP (checked from file bytes, not the name) |
| 422 | `corrupted_image`, `image_too_small` | Can't be decoded, or under 64 px |
| 500 | `internal_error`, `detection_failed` | Unexpected failure; the server keeps running |
| 503 | `detector_unavailable` | Model file or OpenCV feature missing |

Errors look like `{"detail": {"code": "...", "message": "..."}}`.

The JSON result includes `prediction` (the predicted `pair`, `match_score`,
`level`, `reasons`, `message`, `excluded_people`), `faces` (with the features
read for each person), `pair_scores` (every pair's score and components, for
research), `limitations`, `annotated_image` and `crops` (base64 data URLs),
`report_markdown` and `report_html`. The website shows only the prediction.

## 6. How the measurements work

| Measurement | Method |
|---|---|
| Face boxes, 5 landmarks, confidence | YuNet via `cv2.FaceDetectorYN`, threshold 0.6, NMS 0.3 |
| Head tilt | Angle of the line between the two eye landmarks |
| Sharpness | Variance of the Laplacian after resizing the face to 128 px wide; below 60 is flagged |
| Brightness / contrast | Mean / standard deviation of grey levels; under 50 or over 210 flagged, contrast under 25 flagged |
| Small face | Shorter side under 40 px |
| Pair distance | Distance between face centres divided by the average face width |
| Size ratio | Smaller face area ÷ larger face area (mostly reflects distance from the camera) |
| Estimated age | Average of `genderage.onnx` (InsightFace; median of 6 views) and `age_googlenet.onnx` (Adience bands; median expected age over 4 views) |
| Child check | `age_googlenet.onnx`: average probability of age 0–12 across 4 views (2 crop sizes, mirrored) |
| Expression | `emotion-ferplus-8.onnx` (FER+): "happiness" share of 8 expressions. Logged only, not scored |
| Apparent gender | `genderage.onnx` (InsightFace) gender output, averaged over 6 views; used only for the man–woman rule when 85%+ confident |
| Outfit colour | Dominant k-means colour (CIELAB) of the area 1.3 to 2.1 face-heights below the top of the face |
| Height in photo | Difference in face-top position, in face-heights; compared only when face sizes are within 70% of each other |

These thresholds are simple, documented heuristics, not validated measures.
Detection runs on a copy scaled to at most 1600 px; boxes are mapped back to the original.
At most 40 faces (780 pairs) are analysed. If more are found, the most confident 40 are kept and the report says so.

## 7. Privacy

- Uploads are decoded from memory and **never written to disk**. The multipart
  form is closed straight after reading, so no temporary file remains.
- Nothing is stored between requests; responses are sent with `Cache-Control: no-store`.
- Server logs contain only counts, sizes, timings and error codes. They never contain filenames or image data.
- **Debug log:** each analysis appends its measurements to `logs\analysis_log.jsonl` on this computer: face boxes, age estimates from both models, child signal, smile score, outfit colour, every pair's score breakdown and the result. It never stores the photo, face crops or the filename. Read it with `scripts\show_log.py` (add `--last 5`, or `--csv people.csv` to open it in Excel). Turn it off with `DEBUG_LOG_ENABLED = False` in `app\config.py`.
- No identification: no face recognition, no database, no names. No inference of gender, ethnicity, skin colour or attractiveness.
- The page loads two fonts from Google Fonts. Without internet access it falls back to system fonts. Your photo is never sent anywhere except this local server.

## 8. Limitations

- Faces that are small, in profile, covered, blurred or in poor light may be missed. Occasional false detections are possible.
- The quality thresholds are heuristics chosen for this project.
- The match score is an untested prediction: its weights were chosen, not learned, and it has not been compared with real couples yet.
- Friends, family and colleagues often pose close together, so a high score is not proof of a relationship.
- The age model is often several years out and less accurate for some groups. "Height in the photo" is not real height.
- Couples who stand apart in a photo (e.g. parents at opposite ends of a family portrait) are usually missed: nothing in the photo links them except their ages. Check `scripts\show_log.py` to see why any pair was or wasn't predicted.
- The test "group" photo is a composite of one public-domain portrait (four copies of the same person), so similarity ties and pose decides. Real-world accuracy has not been measured.

## 9. Final checklist

- [ ] `scripts\diagnose_env.py` ends with "Everything looks good."
- [ ] `python -m pytest` passes
- [ ] Website opens at http://127.0.0.1:8000
- [ ] Group photo: one predicted pair with a score, highlighted on the photo
- [ ] Report downloads as HTML and Markdown
- [ ] No-face, invalid and oversized files show clear messages
- [ ] "Upload another photo" resets the page
- [ ] Keyboard navigation and phone-width layout work
