import json
from pathlib import Path

import cv2
import numpy as np


def l2_distance(embedding, database):
    embedding = np.asarray(embedding, dtype=np.float32).ravel()
    database = np.asarray(database, dtype=np.float32)
    if database.ndim == 1:
        database = database.reshape(1, -1)
    if database.ndim != 2 or database.shape[0] == 0:
        raise ValueError("Database must contain one or more embedding vectors")
    if database.shape[1] != embedding.size:
        raise ValueError("Database vectors must have the same size as the embedding")

    scores = np.linalg.norm(database - embedding, axis=1)
    return float(scores[0]) if len(scores) == 1 else scores.tolist()


class FaceDatabase:
    MATCH_THRESHOLD = 1.15
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(self, json_path):
        self.path = Path(json_path)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def build(self, image_dir, detector, aligner, embedder):
        image_dir = Path(image_dir)
        data = {}
        images = sorted(p for p in image_dir.rglob("*")
                        if p.suffix.lower() in self.IMAGE_EXTENSIONS)
        if not images:
            raise ValueError(f"No registration images found in {image_dir}")

        for path in images:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Cannot read registration image: {path}")
            faces = detector.detect(image)
            if len(faces) != 1:
                raise ValueError(f"Expected one face in {path}, found {len(faces)}")
            box, kps = faces[0]
            vector = embedder.embed(aligner.align(image, box, kps))
            relative = path.relative_to(image_dir)
            person = relative.parts[0] if len(relative.parts) > 1 else path.stem
            data.setdefault(person, []).append(vector.tolist())

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data))
        self.data = data

    def search(self, embedding):
        if not self.data:
            raise ValueError("Face database is empty; build it from registration images first")

        best_person, best_score = "UNKNOWN", float("inf")
        for person, vectors in self.data.items():
            score = float(np.min(l2_distance(embedding, vectors)))
            if score < best_score:
                best_person, best_score = person, score

        if best_score > self.MATCH_THRESHOLD:
            best_person = "UNKNOWN"
        return best_person, best_score
