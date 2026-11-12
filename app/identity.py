"""Same person in several photos: count them once.

In multi-photo mode the same person can appear in two uploads. Without this
check they would be two "people" and could be paired twice (or with themselves).

How it works
------------
1. SFace (OpenCV's face-recognition model) aligns each face using the five
   landmarks YuNet found and turns it into 128 numbers (an "embedding").
2. Every face is compared with every face from a DIFFERENT photo (cosine
   similarity). Two faces in the same photo are always different people.
3. Pairs above SAME_PERSON_COSINE are merged, most similar first, and a group may
   never contain two faces from the same photo.
4. In each group the largest face is kept as the person; the others are marked
   "same person as ..." and are never paired.

Privacy: faces are compared only with each other within one upload. Embeddings
are not stored or logged, and nobody is identified by name.
Switch it off with the environment variable SAME_PERSON_CHECK=0.
"""

from __future__ import annotations

import logging
import threading

import cv2
import numpy as np

from app import config
from app.vision import Face

log = logging.getLogger(__name__)


class _Recognizer:
    def __init__(self, path=config.SFACE_MODEL_PATH):
        self.path = path
        self._rec = None
        self._failed = False
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return (config.SAME_PERSON_CHECK_ENABLED and hasattr(cv2, "FaceRecognizerSF")
                and self.path.is_file() and not self._failed)

    def embed(self, image: np.ndarray, face: Face) -> np.ndarray | None:
        if not self.available or len(face.landmarks) < 5 or min(face.w, face.h) < config.SAME_PERSON_MIN_FACE:
            return None
        row = np.array([[face.x, face.y, face.w, face.h,
                         *[c for pt in face.landmarks[:5] for c in pt], face.score]], dtype=np.float32)
        with self._lock:
            try:
                if self._rec is None:
                    self._rec = cv2.FaceRecognizerSF.create(str(self.path), "")
                    log.info("Same-person model loaded")
                aligned = self._rec.alignCrop(image, row)
                feat = np.asarray(self._rec.feature(aligned), dtype=np.float64).ravel()
            except cv2.error:
                log.warning("Same-person model failed; check disabled")
                self._failed = True
                return None
        norm = np.linalg.norm(feat)
        return feat / norm if norm else None


recognizer = _Recognizer()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def find_same_people(faces: list[Face], embeddings: dict[str, np.ndarray | None],
                     threshold: float = config.SAME_PERSON_COSINE) -> dict:
    """Group faces from different photos that show the same person.

    Returns {"duplicate_of": {label: kept_label}, "similarity": {label: cosine},
    "groups": [[kept_label, other_label, ...], ...]}.
    """
    by_label = {f.label: f for f in faces}
    candidates = []
    for i, a in enumerate(faces):
        for b in faces[i + 1:]:
            if a.photo == b.photo:
                continue                      # same photo: always two different people
            ea, eb = embeddings.get(a.label), embeddings.get(b.label)
            if ea is None or eb is None:
                continue
            sim = cosine(ea, eb)
            if sim >= threshold:
                candidates.append((sim, a.label, b.label))
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

    group_of = {f.label: {f.label} for f in faces}
    best_sim: dict[frozenset, float] = {}
    for sim, a, b in candidates:
        ga, gb = group_of[a], group_of[b]
        if ga is gb:
            continue
        photos_a = {by_label[x].photo for x in ga}
        if photos_a & {by_label[x].photo for x in gb}:
            continue                          # would put two faces of one photo in one person
        merged = ga | gb
        for x in merged:
            group_of[x] = merged
        best_sim[frozenset((a, b))] = sim

    duplicate_of, similarity, groups, seen = {}, {}, [], set()
    for f in faces:
        g = group_of[f.label]
        if len(g) < 2 or id(g) in seen:
            continue
        seen.add(id(g))
        # Keep the clearest (largest) face; when sizes are close, keep the earliest photo's,
        # so same-photo factors such as height stay available for that person.
        biggest = max(by_label[x].area for x in g)
        members = sorted(g, key=lambda x: (by_label[x].area < 0.8 * biggest, by_label[x].photo,
                                           -by_label[x].area, by_label[x].index))
        keep = members[0]
        for other in members[1:]:
            duplicate_of[other] = keep
            sims = [s for pair, s in best_sim.items() if other in pair]
            similarity[other] = round(max(sims), 3) if sims else None
        groups.append(members)
    return {"duplicate_of": duplicate_of, "similarity": similarity, "groups": groups}


def same_people(photos_images: list[np.ndarray], faces: list[Face]) -> dict:
    """Embed every face (multi-photo uploads only) and group repeats of the same person."""
    empty = {"duplicate_of": {}, "similarity": {}, "groups": [], "checked": False}
    if len({f.photo for f in faces}) < 2 or not recognizer.available:
        return empty
    embeddings = {f.label: recognizer.embed(photos_images[f.photo], f) for f in faces}
    return {**find_same_people(faces, embeddings), "checked": recognizer.available}
