import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from face_onnx import FaceAlign, FaceAnti, FaceDatabase, FaceDetector, FaceEmbedder, l2_distance


def test_face_det():
    root = Path(__file__).resolve().parents[1]
    fixtures = Path(__file__).parent / "fixtures"
    image_path = fixtures / "frames" / "inputs" / "frame_000001.jpg"
    output_path = fixtures / "frames" / "out_det" / "frame_000001.jpg"
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    visualized = image.copy()
    for box, kps in FaceDetector(root / "weights" / "onnx" / "scrfd_10g_bnkps.onnx").detect(image):
        x1, y1, x2, y2 = np.round(box).astype(int)
        cv2.rectangle(visualized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for x, y in np.round(kps).astype(int):
            cv2.circle(visualized, (x, y), 3, (0, 0, 255), -1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), visualized)


def test_face_align():
    root = Path(__file__).resolve().parents[1]
    fixtures = Path(__file__).parent / "fixtures"
    image_path = fixtures / "frames" / "inputs" / "frame_000001.jpg"
    output_dir = fixtures / "frames" / "out_align" / "frame_000001"
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    faces = FaceDetector(root / "weights" / "onnx" / "scrfd_10g_bnkps.onnx").detect(image)
    output_dir.mkdir(parents=True, exist_ok=True)
    aligner = FaceAlign()
    for i, (box, kps) in enumerate(faces, 1):
        cv2.imwrite(str(output_dir / f"face_{i:02d}.jpg"), aligner.align(image, box, kps))


def test_face_embed():
    root = Path(__file__).resolve().parents[1]
    fixtures = Path(__file__).parent / "fixtures"
    crop_path = fixtures / "frames" / "out_align" / "frame_000001" / "face_01.jpg"
    output_path = fixtures / "frames" / "out_emb" / "frame_000001.face_01.json"
    face_crop = cv2.imread(str(crop_path))
    if face_crop is None:
        raise FileNotFoundError(crop_path)

    embedding = FaceEmbedder(root / "weights" / "onnx" / "arcface_r100.onnx").embed(face_crop)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(embedding.tolist()))


def test_face_compare():
    root = Path(__file__).resolve().parents[1]
    fixtures = Path(__file__).parent / "fixtures"
    detector = FaceDetector(root / "weights" / "onnx" / "scrfd_10g_bnkps.onnx")
    aligner = FaceAlign()
    embedder = FaceEmbedder(root / "weights" / "onnx" / "arcface_r100.onnx")

    def get_embedding(image_path):
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        faces = detector.detect(image)
        if not faces:
            raise ValueError(f"No face detected in {image_path}")
        box, kps = max(faces, key=lambda face: (face[0][2] - face[0][0]) * (face[0][3] - face[0][1]))
        return embedder.embed(aligner.align(image, box, kps))

    embedding1 = get_embedding(fixtures / "nv2.jpg")
    embedding2 = get_embedding(fixtures / "nv4.jpg")
    score = l2_distance(embedding1, embedding2)
    threshold = 1.15
    print(f"L2 distance: {score:.6f}")
    print(f"Result (threshold {threshold}): {'MATCH' if score <= threshold else 'NO MATCH'}")


def test_face_anti():
    root = Path(__file__).resolve().parents[1]
    fixtures = Path(__file__).parent / "fixtures"
    image_path = fixtures / "frames" / "inputs" / "frame_000001.jpg"
    output_path = fixtures / "frames" / "out_anti" / "frame_000001.jpg"
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    detector = FaceDetector(root / "weights" / "onnx" / "scrfd_10g_bnkps.onnx")
    anti = FaceAnti(root / "weights" / "onnx" / "MiniFASNetV2.onnx",
                    root / "weights" / "onnx" / "MiniFASNetV1SE.onnx")
    visualized = image.copy()
    for box, _ in detector.detect(image):
        crop_v2 = FaceAnti.crop(image, box, FaceAnti.V2_SCALE)
        crop_v1se = FaceAnti.crop(image, box, FaceAnti.V1SE_SCALE)
        result = anti.check(crop_v2, crop_v1se)
        x1, y1, x2, y2 = np.round(box).astype(int)
        color = (0, 255, 0) if result["is_real"] else (0, 0, 255)
        label = "REAL" if result["is_real"] else "FAKE"
        score = result["real_score"] if result["is_real"] else result["spoof_score"]
        cv2.rectangle(visualized, (x1, y1), (x2, y2), color, 2)
        cv2.putText(visualized, f"{label} {score:.2f}", (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), visualized)


def _load_face_pipeline(root):
    detector = FaceDetector(root / "weights" / "onnx" / "scrfd_10g_bnkps.onnx")
    aligner = FaceAlign()
    embedder = FaceEmbedder(root / "weights" / "onnx" / "arcface_r100.onnx")
    anti = FaceAnti(root / "weights" / "onnx" / "MiniFASNetV2.onnx",
                    root / "weights" / "onnx" / "MiniFASNetV1SE.onnx")
    database = FaceDatabase(root / "tests" / "db_features.json")
    return detector, aligner, embedder, anti, database


def _process_pipeline_frame(image_path, output_path, pipeline):
    detector, aligner, embedder, anti, database = pipeline
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    visualized = image.copy()
    for box, kps in detector.detect(image):
        crops = (
            FaceAnti.crop(image, box, FaceAnti.V2_SCALE),
            FaceAnti.crop(image, box, FaceAnti.V1SE_SCALE),
        )
        anti_result = anti.check(*crops)
        x1, y1, x2, y2 = np.round(box).astype(int)

        if anti_result["is_real"]:
            vector = embedder.embed(aligner.align(image, box, kps))
            person, distance = database.search(vector)
            color = (0, 255, 0) if person != "UNKNOWN" else (0, 255, 255)
            label = f"REAL: {person} L2={distance:.2f}"
        else:
            color = (0, 0, 255)
            label = f"FAKE {anti_result['spoof_score']:.2f}"

        cv2.rectangle(visualized, (x1, y1), (x2, y2), color, 2)
        cv2.putText(visualized, label, (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), visualized)


def test_face_pipeline():
    root = Path(__file__).resolve().parents[1]
    fixtures = Path(__file__).parent / "fixtures"
    image_path = fixtures / "frames" / "inputs" / "frame_000001.jpg"
    output_path = fixtures / "frames" / "out_pipeline" / "frame_000001.jpg"
    _process_pipeline_frame(image_path, output_path, _load_face_pipeline(root))


@pytest.mark.slow
def test_face_pipeline_all():
    root = Path(__file__).resolve().parents[1]
    fixtures = Path(__file__).parent / "fixtures" / "frames"
    image_paths = sorted((fixtures / "inputs").glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(fixtures / "inputs")

    pipeline = _load_face_pipeline(root)
    for image_path in image_paths:
        output_path = fixtures / "out_pipeline" / image_path.name
        _process_pipeline_frame(image_path, output_path, pipeline)
