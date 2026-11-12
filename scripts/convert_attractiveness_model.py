"""Convert the SCUT-FBP5500 attractiveness model (PyTorch) to ONNX, once.

The app runs models with OpenCV, which reads ONNX files, but the published
attractiveness weights are PyTorch. Run this ONCE, in Google Colab (free) or any
computer with PyTorch, then copy the output file into the project's models/ folder.

GOOGLE COLAB (about 5 minutes)
  1. Open https://colab.research.google.com and create a new notebook.
  2. In the first cell, run:
         !pip install -q torch torchvision onnx huggingface_hub
  3. Upload this file (left panel: Files > Upload), then in a new cell run:
         !python convert_attractiveness_model.py
  4. Download the file it creates, attractiveness_resnet18.onnx, from the Files panel.
  5. Put it in F:\\THE_DATING_GAME\\models\\ and restart the app.
     The website then shows the "Similar attractiveness" factor and each person's rating.

WHAT IT DOES
  * Downloads the ResNet-18 weights trained on SCUT-FBP5500 (Liang et al., ICPR 2018)
    from https://huggingface.co/Gustrd/SCUT-FBP5500-PyTorch-Model
  * Wraps the network so the colour normalisation is INSIDE the ONNX file: the app
    sends plain 0-255 RGB pixels at 224 x 224 (see app/attributes.py).
  * Exports with ONNX opset 11, which OpenCV 4.10 reads.
  * Prints ratings for a few test faces so you can check they look sensible (1-5).

CHECKING THE RESULT
  The dataset paper doesn't publish its colour normalisation. This script uses
  the standard ImageNet values that torchvision models use. If every test face
  comes out at almost the same rating, or outside 1-5, run again with
  --no-normalise and compare; keep whichever gives a spread of sensible ratings.

LICENCE: SCUT-FBP5500 and models derived from it are for non-commercial research only.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request

import numpy as np

REPO = "Gustrd/SCUT-FBP5500-PyTorch-Model"
WEIGHTS = "resnet18_py3.pth"
OUT = "attractiveness_resnet18.onnx"
TEST_FACES = {   # public-domain portraits used only to sanity-check the output range
    "astronaut": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Eileen_Collins.jpg/480px-Eileen_Collins.jpg",
}


def main() -> int:
    import torch
    import torchvision
    from huggingface_hub import hf_hub_download

    ap = argparse.ArgumentParser()
    ap.add_argument("--no-normalise", action="store_true", help="feed 0-1 RGB without ImageNet mean/std")
    args = ap.parse_args()

    path = hf_hub_download(REPO, WEIGHTS)
    net = torchvision.models.resnet18(num_classes=1)
    state = torch.load(path, map_location="cpu", encoding="latin1")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    net.load_state_dict(state)
    net.eval()

    class Wrapped(torch.nn.Module):
        """0-255 RGB in, 1-5 rating out."""

        def __init__(self, inner, normalise: bool):
            super().__init__()
            self.inner = inner
            self.normalise = normalise
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, x):
            x = x / 255.0
            if self.normalise:
                x = (x - self.mean) / self.std
            return self.inner(x)

    model = Wrapped(net, normalise=not args.no_normalise).eval()
    dummy = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(model, dummy, OUT, input_names=["input"], output_names=["score"], opset_version=11)
    print(f"Saved {OUT}")

    # Sanity check on real faces
    try:
        from PIL import Image
        for name, url in TEST_FACES.items():
            req = urllib.request.Request(url, headers={"User-Agent": "dating-game-converter"})
            img = Image.open(urllib.request.urlopen(req, timeout=30)).convert("RGB")
            w, h = img.size
            side = min(w, h)
            img = img.crop(((w - side) // 2, 0, (w + side) // 2, side)).resize((256, 256)).crop((16, 16, 240, 240))
            x = torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1)[None]
            with torch.no_grad():
                print(f"Test rating for {name}: {float(model(x)):.2f} (expected roughly 2-4 on a 1-5 scale)")
    except Exception as exc:  # the check is optional
        print(f"(Skipped the test rating: {exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
