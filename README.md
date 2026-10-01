# Face

Face anti, detection, alignment, embedding, and comparison utilities.

## Installation

```bash
# Default (ONNX Runtime)
uv sync

# RK3588 Board Support (RKNN Runtime)
uv sync --extra rknn
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

# Run all tests
uv run pytest

# Optional: process all input frames (excluded by default)
uv run pytest -m slow tests/test_face_onnx.py::test_face_pipeline_all
```
