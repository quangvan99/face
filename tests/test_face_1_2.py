"""Smoke tests for Bài toán 1.2 using real and synthetic attack fixtures."""

import sys
from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
ATTACKS = FIXTURES / "augmented_1_2"
sys.path.insert(0, str(ROOT))

from face_onnx import FaceAnti, FaceDetector


@pytest.fixture(scope="module")
def anti_pipeline():
    onnx = ROOT / "weights" / "onnx"
    return (
        FaceDetector(onnx / "scrfd_10g_bnkps.onnx"),
        FaceAnti(onnx / "MiniFASNetV2.onnx", onnx / "MiniFASNetV1SE.onnx"),
    )


def _check(image_path, anti_pipeline):
    detector, anti = anti_pipeline
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    faces = detector.detect(image)
    if not faces:
        raise AssertionError(f"No face detected in {image_path}")
    box, _ = max(
        faces,
        key=lambda face: (face[0][2] - face[0][0]) * (face[0][3] - face[0][1]),
    )
    crop_v2 = FaceAnti.crop(image, box, FaceAnti.V2_SCALE)
    crop_v1se = FaceAnti.crop(image, box, FaceAnti.V1SE_SCALE)
    return anti.check(crop_v2, crop_v1se)


def test_tc_1_2_01_real_person_is_live(anti_pipeline):
    assert _check(FIXTURES / "cam_real.jpg", anti_pipeline)["is_real"]


def test_tc_1_2_02_color_print_is_spoof(anti_pipeline):
    assert not _check(ATTACKS / "cam_print_color.png", anti_pipeline)["is_real"]


def test_tc_1_2_03_black_and_white_print_is_spoof(anti_pipeline):
    assert not _check(ATTACKS / "cam_print_bw.png", anti_pipeline)["is_real"]


def test_tc_1_2_04_screen_replay_is_spoof(anti_pipeline):
    assert not _check(FIXTURES / "cam_fake.jpg", anti_pipeline)["is_real"]


def test_tc_1_2_05_tablet_video_replay():
    pytest.skip("A still image cannot test video replay behavior")


def test_tc_1_2_06_demo_photo_card_is_spoof(anti_pipeline):
    assert not _check(ATTACKS / "cam_card_demo.png", anti_pipeline)["is_real"]


def test_tc_1_2_07_twenty_live_and_twenty_attacks():
    pytest.skip("The 20 live + 20 attack dataset is deferred")
