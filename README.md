# face-onnx

face detection, alignment, embedding, and comparison utilities.

## Install

```bash
uv sync
```

## Test

Run all tests with `uv run pytest`. Run an individual module test with:

```bash
uv run pytest tests/test_face_onnx.py::test_face_det
uv run pytest tests/test_face_onnx.py::test_face_align
uv run pytest tests/test_face_onnx.py::test_face_embed
uv run pytest tests/test_face_onnx.py::test_face_compare
```
