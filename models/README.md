# Face detection model

`face_detection_yunet_2023mar.onnx` is **YuNet**, a lightweight CNN face detector
from the [OpenCV Model Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).

- Licence: MIT (see the OpenCV Model Zoo repository).
- Size: 232,589 bytes
- SHA-256: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`
- Loaded with `cv2.FaceDetectorYN.create(...)`, available in OpenCV 4.8 and later.
- Outputs per face: bounding box, 5 landmarks (eyes, nose tip, mouth corners) and a confidence score.

If the file is missing or corrupted, run:

```powershell
.\.venv\Scripts\python.exe scripts\download_model.py --force
```

## Age and expression models (optional)

| File | Source | Purpose |
|---|---|---|
| `age_googlenet.onnx` (23.9 MB) | [ONNX Model Zoo, age_gender](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/age_gender) | Age bands; used only as a second check for any sign that a person is a child. |
| `genderage.onnx` (1.3 MB) | [InsightFace](https://github.com/deepinsight/insightface) model, file from [yakhyo/facial-analysis](https://github.com/yakhyo/facial-analysis) | Numeric age estimate. **Its age output gives the age estimate; its gender output is used only for the man–woman pairing rule (and only when confident).** InsightFace's pretrained models are licensed for non-commercial research only. |
| `emotion-ferplus-8.onnx` (35.0 MB) | [ONNX Model Zoo, emotion_ferplus](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus) | Facial expression; the "happiness" share is used as a smile score |

Checksums are in `app/config.py`. Without these files the app still runs, but nobody can be confirmed as an adult, so no pairs are predicted.
Check each model's licence and training data on its source page before any use beyond this academic project.
