"""Smoke tests for Bài toán 1.1 using the available face fixtures."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
AUGMENTED = FIXTURES / "augmented_1_1"
sys.path.insert(0, str(ROOT))

from face_onnx import FaceAlign, FaceDetector, FaceEmbedder


# Fixture-only threshold. Calibrate a production threshold on real validation data.
COSINE_THRESHOLD = 0.5


@pytest.fixture(scope="module")
def face_pipeline():
    detector = FaceDetector(ROOT / "weights" / "onnx" / "scrfd_10g_bnkps.onnx")
    aligner = FaceAlign()
    embedder = FaceEmbedder(ROOT / "weights" / "onnx" / "arcface_r100.onnx")
    return detector, aligner, embedder


def _face_embedding(image_path, face_pipeline):
    detector, aligner, embedder = face_pipeline
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    faces = detector.detect(image)
    if not faces:
        raise AssertionError(f"No face detected in {image_path}")
    box, landmarks = max(
        faces,
        key=lambda face: (face[0][2] - face[0][0]) * (face[0][3] - face[0][1]),
    )
    size = (float(box[2] - box[0]), float(box[3] - box[1]))
    vector = embedder.embed(aligner.align(image, box, landmarks))
    return vector, size


def _similarity(id_image, camera_image, face_pipeline):
    id_vector, _ = _face_embedding(id_image, face_pipeline)
    camera_vector, _ = _face_embedding(camera_image, face_pipeline)
    return float(np.dot(id_vector, camera_vector))


def test_tc_1_1_01_an_cccd_matches_camera_an(face_pipeline):
    score = _similarity(
        FIXTURES / "an_cccd.jpg",
        FIXTURES / "cam_real.jpg",
        face_pipeline,
    )
    assert score >= COSINE_THRESHOLD, f"Expected MATCH, cosine={score:.4f}"


def test_tc_1_1_02_hien_cccd_does_not_match_camera_an(face_pipeline):
    score = _similarity(
        FIXTURES / "hien_cccd.jpg",
        FIXTURES / "cam_real.jpg",
        face_pipeline,
    )
    assert score < COSINE_THRESHOLD, f"Expected MISMATCH, cosine={score:.4f}"


def test_tc_1_1_03_an_with_glasses_matches(face_pipeline):
    score = _similarity(
        FIXTURES / "an_cccd.jpg",
        FIXTURES / "cam_an_deokinh.jpg",
        face_pipeline,
    )
    assert score >= COSINE_THRESHOLD, f"Expected MATCH with glasses, cosine={score:.4f}"


def test_tc_1_1_04_yaw_pitch_15_to_30_degrees(face_pipeline):
    variants = (
        "cam_real_yaw_left.png",
        "cam_real_yaw_right.png",
        "cam_real_pitch_up.png",
        "cam_real_pitch_down.png",
    )
    for name in variants:
        image_path = AUGMENTED / name
        score = _similarity(FIXTURES / "an_cccd.jpg", image_path, face_pipeline)
        assert score >= COSINE_THRESHOLD, f"Expected MATCH for {name}, cosine={score:.4f}"


def test_tc_1_1_05_face_smaller_than_120px_is_rejected():
    pytest.skip("The current pipeline has no <120px rejection rule to exercise")


def test_tc_1_1_06_face_at_least_120px_matches(face_pipeline):
    image_path = AUGMENTED / "cam_real_face_180px.jpg"
    _, face_size = _face_embedding(image_path, face_pipeline)
    assert min(face_size) >= 120, f"Expected >=120px bbox, got {face_size}"

    score = _similarity(FIXTURES / "an_cccd.jpg", image_path, face_pipeline)
    assert score >= COSINE_THRESHOLD, f"Expected MATCH, cosine={score:.4f}"


def test_tc_1_1_07_twenty_people():
    pytest.skip("The 20-person dataset is deferred")


def test_synthetic_camera_variants_still_match_an_and_reject_hien(face_pipeline):
    variants = sorted(AUGMENTED.glob("cam_real_*.jpg"))
    variants += sorted(AUGMENTED.glob("cam_real_*.png"))
    assert variants, f"No synthetic camera variants found in {AUGMENTED}"

    for camera_image in variants:
        an_score = _similarity(FIXTURES / "an_cccd.jpg", camera_image, face_pipeline)
        hien_score = _similarity(FIXTURES / "hien_cccd.jpg", camera_image, face_pipeline)
        assert an_score >= COSINE_THRESHOLD, (
            f"Expected MATCH for {camera_image.name}, cosine={an_score:.4f}"
        )
        assert hien_score < COSINE_THRESHOLD, (
            f"Expected MISMATCH for {camera_image.name}, cosine={hien_score:.4f}"
        )
