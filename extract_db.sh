#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DB_DIR="$PROJECT_ROOT/db"
DETECTOR_MODEL="$PROJECT_ROOT/weights/onnx/scrfd_10g_bnkps.onnx"
EMBEDDER_MODEL="$PROJECT_ROOT/weights/onnx/arcface_r100.onnx"

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is required to build the face database." >&2
    exit 1
fi

if [[ ! -d "$DB_DIR" ]]; then
    echo "Error: registration image directory not found: $DB_DIR" >&2
    exit 1
fi

for model in "$DETECTOR_MODEL" "$EMBEDDER_MODEL"; do
    if [[ ! -f "$model" ]]; then
        echo "Error: required model not found: $model" >&2
        exit 1
    fi
done

cd "$PROJECT_ROOT"
uv run python - "$PROJECT_ROOT" <<'PY'
import sys
from pathlib import Path

from face_onnx import FaceAlign, FaceDetector, FaceEmbedder
from utils import FaceDatabase

project_root = Path(sys.argv[1])
db_dir = project_root / "db"
database = FaceDatabase(db_dir / "features.json")
detector = FaceDetector(project_root / "weights/onnx/scrfd_10g_bnkps.onnx")
aligner = FaceAlign()
embedder = FaceEmbedder(project_root / "weights/onnx/arcface_r100.onnx")

database.build(db_dir, detector, aligner, embedder)

image_count = sum(len(embeddings) for embeddings in database.data.values())
print(f"Wrote {database.path} ({len(database.data)} people, {image_count} images)")
PY
