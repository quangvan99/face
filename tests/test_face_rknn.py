import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from face_onnx import FaceAnti as OnnxFaceAnti
from face_onnx import FaceAlign, FaceDetector as OnnxFaceDetector
from face_onnx import FaceEmbedder as OnnxFaceEmbedder
from face_rknn import FaceAnti as RknnFaceAnti
from face_rknn import FaceAlign as RknnFaceAlign
from face_rknn import FaceDetector as RknnFaceDetector
from face_rknn import FaceEmbedder as RknnFaceEmbedder


ONNX = ROOT / "weights" / "onnx"
RKNN = ROOT / "weights" / "rknn"
SAMPLES = (
    ROOT / "tests/fixtures/frames/inputs/frame_000001.jpg",
    ROOT / "tests/fixtures/frames/inputs/frame_000100.jpg",
    ROOT / "tests/fixtures/frames/inputs/frame_000300.jpg",
    ROOT / "tests/db/an/an.jpg",
    ROOT / "tests/db/tu/tu.jpg",
    *(ROOT / "tests/fixtures" / f"nv{i}.jpg" for i in range(1, 5)),
)


def _images():
    for path in SAMPLES:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        yield path.name, image


def _largest_face(faces, name):
    if not faces:
        raise AssertionError(f"No face detected in {name}")
    return max(faces, key=lambda face: (face[0][2] - face[0][0]) * (face[0][3] - face[0][1]))


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-6)


def _benchmark(function, repeats=10):
    for _ in range(3):
        function()
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    return (time.perf_counter() - start) * 1000 / repeats


def _print_speed(name, onnx_ms, rknn_ms):
    print(f"{name}: ONNX {onnx_ms:.2f} ms, RKNN {rknn_ms:.2f} ms, speedup {onnx_ms / rknn_ms:.2f}x")


def test_detector_matches_onnx_and_benchmarks_speed():
    onnx_detector = OnnxFaceDetector(ONNX / "scrfd_10g_bnkps.onnx")
    rknn_detector = RknnFaceDetector(RKNN / "scrfd_10g_bnkps.rknn")
    try:
        images = list(_images())
        for name, image in images:
            expected = onnx_detector.detect(image)
            actual = rknn_detector.detect(image)
            assert len(actual) == len(expected), f"face count differs for {name}"
            for box, landmarks in expected:
                match = max(actual, key=lambda face: _iou(box, face[0]))
                assert _iou(box, match[0]) > 0.98, f"bbox differs for {name}"
                assert np.mean(np.linalg.norm(landmarks - match[1], axis=1)) < 2.0, f"landmarks differ for {name}"

        name, image = images[0]
        onnx_ms = _benchmark(lambda: onnx_detector.detect(image))
        rknn_ms = _benchmark(lambda: rknn_detector.detect(image))
        _print_speed("SCRFD detect", onnx_ms, rknn_ms)
    finally:
        rknn_detector.close()


def test_embedder_matches_onnx_and_benchmarks_speed():
    detector = OnnxFaceDetector(ONNX / "scrfd_10g_bnkps.onnx")
    aligner = FaceAlign()
    rknn_aligner = RknnFaceAlign()
    onnx_embedder = OnnxFaceEmbedder(ONNX / "arcface_r100.onnx")
    rknn_embedder = RknnFaceEmbedder(RKNN / "arcface_r100.rknn")
    try:
        crops = []
        for name, image in _images():
            box, landmarks = _largest_face(detector.detect(image), name)
            expected_crop = aligner.align(image, box, landmarks)
            actual_crop = rknn_aligner.align(image, box, landmarks)
            assert np.array_equal(expected_crop, actual_crop), f"alignment differs for {name}"
            crops.append((name, expected_crop))
        for name, crop in crops:
            expected = onnx_embedder.embed(crop)
            actual = rknn_embedder.embed(crop)
            cosine = float(np.dot(expected, actual) / (np.linalg.norm(expected) * np.linalg.norm(actual)))
            assert cosine > 0.999, f"embedding differs for {name}: cosine={cosine:.6f}"

        onnx_ms = _benchmark(lambda: onnx_embedder.embed(crops[0][1]))
        rknn_ms = _benchmark(lambda: rknn_embedder.embed(crops[0][1]))
        _print_speed("ArcFace R100", onnx_ms, rknn_ms)
    finally:
        rknn_embedder.close()


def test_anti_spoof_matches_onnx_and_benchmarks_speed():
    detector = OnnxFaceDetector(ONNX / "scrfd_10g_bnkps.onnx")
    onnx_anti = OnnxFaceAnti(ONNX / "MiniFASNetV2.onnx", ONNX / "MiniFASNetV1SE.onnx")
    rknn_anti = RknnFaceAnti(RKNN / "MiniFASNetV2.rknn", RKNN / "MiniFASNetV1SE.rknn")
    try:
        crops = []
        for name, image in _images():
            box, _ = _largest_face(detector.detect(image), name)
            crop_v2 = OnnxFaceAnti.crop(image, box, OnnxFaceAnti.V2_SCALE)
            crop_v1se = OnnxFaceAnti.crop(image, box, OnnxFaceAnti.V1SE_SCALE)
            assert np.array_equal(crop_v2, RknnFaceAnti.crop(image, box, RknnFaceAnti.V2_SCALE))
            assert np.array_equal(crop_v1se, RknnFaceAnti.crop(image, box, RknnFaceAnti.V1SE_SCALE))
            crops.append((
                name,
                crop_v2,
                crop_v1se,
            ))
        for name, crop_v2, crop_v1se in crops:
            expected = onnx_anti.check(crop_v2, crop_v1se)
            actual = rknn_anti.check(crop_v2, crop_v1se)
            assert abs(expected["real_score"] - actual["real_score"]) < 0.01, f"anti score differs for {name}"
            assert expected["is_real"] == actual["is_real"], f"anti decision differs for {name}"

        _, crop_v2, crop_v1se = crops[0]
        onnx_ms = _benchmark(lambda: onnx_anti.check(crop_v2, crop_v1se))
        rknn_ms = _benchmark(lambda: rknn_anti.check(crop_v2, crop_v1se))
        _print_speed("MiniFASNet ensemble", onnx_ms, rknn_ms)
    finally:
        rknn_anti.close()
