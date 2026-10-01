# Face

Face anti, detection, alignment, embedding, and comparison utilities.

## Installation

```bash
# Default (ONNX Runtime)
uv sync

# RK3588 Board Support (RKNN Runtime)
uv sync --extra rknn

# Server CPU Support (OpenVINO Runtime)
uv sync --extra openvino
```

## Convert ONNX to RKNN

Place `.onnx` models into `weights/onnx/`, then run:

```bash
bash convert_to_rknn.sh
```
*Outputs `.rknn` models and logs to `weights/rknn/`.*

## Testing

```bash
# Test ONNX models
uv run pytest tests/test_face_onnx.py

# Test RKNN models & benchmark speed (on RK3588 board)
uv run pytest tests/test_face_rknn.py

# Test OpenVINO CPU backend against ONNX Runtime
uv sync --extra openvino
uv run pytest tests/test_face_openvino.py

# Run all tests
uv run pytest

# Optional: process all input frames (excluded by default)
uv run pytest -m slow tests/test_face_onnx.py::test_face_pipeline_all
```

## Test-case coverage

The full scenario matrix is in [docs/testcases.md](docs/testcases.md).

```bash
# BÀI TOÁN 1.1: CCCD-to-camera face verification
uv run pytest tests/test_face_1_1.py

# BÀI TOÁN 1.2: face anti-spoofing / liveness
uv run pytest tests/test_face_1_2.py
```

Current fixture coverage:

- **Bài toán 1.1:** TC-1.1.01–04 and TC-1.1.06 pass with the available fixtures. TC-1.1.05 is skipped because the pipeline does not yet reject faces smaller than 120 px. TC-1.1.07 is deferred until the 20-person dataset is available. Yaw/pitch samples are AI-generated smoke-test images.
- **Bài toán 1.2:** TC-1.2.01–03 and TC-1.2.06 pass with the available fixtures. The screen-replay sample for TC-1.2.04 also passes, but it is a tablet, not a smartphone. TC-1.2.05 is skipped because a still image cannot test video replay; TC-1.2.07 is deferred until the larger dataset is available. The TC-1.2.06 fixture is a clearly marked demo card, not a real CCCD.

Generated images are preliminary smoke-test data; real captures are still needed for acceptance testing.
