import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pytest.importorskip("openvino")

from face_onnx import FaceAnti as OnnxFaceAnti
from face_onnx import FaceAlign as OnnxFaceAlign
from face_onnx import FaceDetector as OnnxFaceDetector
from face_onnx import FaceEmbedder as OnnxFaceEmbedder
from face_openvino import FaceAnti, FaceAlign, FaceDetector, FaceEmbedder


ONNX = ROOT / "weights" / "onnx"
FRAME = ROOT / "tests" / "fixtures" / "frames" / "inputs" / "frame_000001.jpg"


@pytest.fixture(scope="module")
def backends():
    openvino_detector = FaceDetector(ONNX / "scrfd_10g_bnkps.onnx")
    openvino_embedder = FaceEmbedder(ONNX / "arcface_r100.onnx")
    openvino_anti = FaceAnti(
        ONNX / "MiniFASNetV2.onnx",
        ONNX / "MiniFASNetV1SE.onnx",
    )
    try:
        yield {
            "onnx_detector": OnnxFaceDetector(ONNX / "scrfd_10g_bnkps.onnx"),
            "onnx_embedder": OnnxFaceEmbedder(ONNX / "arcface_r100.onnx"),
            "onnx_anti": OnnxFaceAnti(
                ONNX / "MiniFASNetV2.onnx", ONNX / "MiniFASNetV1SE.onnx"
            ),
            "openvino_detector": openvino_detector,
            "openvino_embedder": openvino_embedder,
            "openvino_anti": openvino_anti,
        }
    finally:
        openvino_detector.close()
        openvino_embedder.close()
        openvino_anti.close()


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-6)


def _load_frame():
    image = cv2.imread(str(FRAME))
    if image is None:
        raise FileNotFoundError(FRAME)
    return image


def _largest_face(faces):
    if not faces:
        raise AssertionError("No face detected in the OpenVINO parity fixture")
    return max(
        faces,
        key=lambda face: (face[0][2] - face[0][0]) * (face[0][3] - face[0][1]),
    )


def test_openvino_detector_matches_onnx(backends):
    image = _load_frame()
    expected = backends["onnx_detector"].detect(image)
    actual = backends["openvino_detector"].detect(image)

    assert len(actual) == len(expected)
    for expected_box, expected_landmarks in expected:
        actual_box, actual_landmarks = max(actual, key=lambda face: _iou(expected_box, face[0]))
        assert _iou(expected_box, actual_box) > 0.98
        assert np.mean(np.linalg.norm(expected_landmarks - actual_landmarks, axis=1)) < 1.0


def test_openvino_embedder_matches_onnx(backends):
    image = _load_frame()
    box, landmarks = _largest_face(backends["onnx_detector"].detect(image))
    onnx_crop = OnnxFaceAlign().align(image, box, landmarks)
    openvino_crop = FaceAlign().align(image, box, landmarks)
    assert np.array_equal(onnx_crop, openvino_crop)

    expected = backends["onnx_embedder"].embed(onnx_crop)
    actual = backends["openvino_embedder"].embed(openvino_crop)
    cosine = float(np.dot(expected, actual) / (np.linalg.norm(expected) * np.linalg.norm(actual)))
    assert cosine > 0.999

    with ThreadPoolExecutor(max_workers=2) as pool:
        concurrent = list(pool.map(backends["openvino_embedder"].embed, [openvino_crop] * 2))
    for vector in concurrent:
        assert float(np.dot(actual, vector)) > 0.999


def test_openvino_anti_spoof_matches_onnx(backends):
    image = _load_frame()
    box, _ = _largest_face(backends["onnx_detector"].detect(image))
    crop_v2 = OnnxFaceAnti.crop(image, box, OnnxFaceAnti.V2_SCALE)
    crop_v1se = OnnxFaceAnti.crop(image, box, OnnxFaceAnti.V1SE_SCALE)

    expected = backends["onnx_anti"].check(crop_v2, crop_v1se)
    actual = backends["openvino_anti"].check(crop_v2, crop_v1se)
    assert abs(expected["real_score"] - actual["real_score"]) < 0.01
    assert expected["is_real"] == actual["is_real"]
