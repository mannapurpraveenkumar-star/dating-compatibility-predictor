# Research protocol: testing the question properly

> Can a Machine Learning Model predict whether two people are meant to be together?

The website answers "Unable to determine" for every pair because one photo holds
no valid evidence about relationships. This folder shows how the question could
be **tested** rather than assumed.

## 1. Define the label before collecting data
"Meant to be together" cannot be measured. Pick something observable and
reported by the participants themselves, for example *"the two people say they
are currently a couple"*. Never derive the label from the photo.

## 2. Consent and ethics
- Written, informed consent from **every** person in every photo, including
  consent to their face measurements being used for research.
- Ethics approval from your institution before collecting anything.
- Store participant IDs as random codes, never names. Keep photos separate
  from labels and delete photos once features are extracted.
- Do not collect or infer sensitive attributes (gender, ethnicity, age, sexuality).

## 3. Data format
One row per pair in `pairs_template.csv`. Feature columns are produced by the
app (`measurements` in the API response) and listed in
`app/analysis.py: PAIR_FEATURE_NAMES`.

```
photo_id,person_a_id,person_b_id,centre_distance_face_widths,size_ratio,vertical_offset_face_heights,adjacent_in_row,box_overlap_iou,label
```

## 4. Evaluation
```powershell
.\.venv\Scripts\python.exe -m app.research research\your_data.csv
```
- **Grouped train/test split**: rows sharing a photo or a person are kept on the
  same side, so the model cannot memorise people (prevents leakage).
- **Baselines**: majority class and a stratified random guesser.
- **Metrics**: accuracy, balanced accuracy, ROC AUC, confusion matrix.
- A model that does not clearly beat the baselines on held-out groups has not
  shown predictive ability.

## 5. Expected confounds (write these up)
Couples tend to stand next to each other *because they are a couple*, so
"distance in the frame" may predict "is a couple" without saying anything about
compatibility. That is a photography convention, not a property of the people.
Report any positive result in exactly those terms.
