#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RKNN_DIR="$ROOT_DIR/weights/rknn"
mkdir -p "$RKNN_DIR/logs"

UV_CMD=(
  uv run --no-project --python 3.10
  --with 'setuptools<81'
  --with 'cmake<4'
  --with 'numpy==1.26.4'
  --with 'rknn-toolkit2==2.3.2'
)

convert_model() {
  local model_name="$1"
  local input_name="$2"
  local input_shape="$3"
  local log_path="$RKNN_DIR/logs/${model_name}.log"

  if "${UV_CMD[@]}" python - "$model_name" "$input_name" "$input_shape" >"$log_path" 2>&1 <<'PY'
import sys
from pathlib import Path

from rknn.api import RKNN

name, input_name, shape = sys.argv[1:]
root = Path.cwd()
source = root / "weights" / "onnx" / f"{name}.onnx"
output = root / "weights" / "rknn" / f"{name}.rknn"
input_size = [list(map(int, shape.split(",")))]

if not source.exists():
    raise FileNotFoundError(source)

output.unlink(missing_ok=True)
print(f"Converting {source} -> {output}", flush=True)
converter = RKNN(verbose=True)
try:
    def check(result):
        if result != 0:
            raise RuntimeError(f"RKNN conversion returned error code {result}")

    check(converter.config(target_platform="rk3588"))
    check(converter.load_onnx(model=str(source), inputs=[input_name], input_size_list=input_size))
    check(converter.build(do_quantization=False))
    check(converter.export_rknn(str(output)))
    print(f"SUCCESS: {output}", flush=True)
finally:
    converter.release()
PY
  then
    echo "Converted $model_name (log: $log_path)"
  else
    local status=$?
    echo "Conversion failed for $model_name (exit $status); see $log_path" >&2
    exit "$status"
  fi
}

convert_model scrfd_10g_bnkps input.1 1,3,640,640
convert_model arcface_r100 input.1 1,3,112,112
convert_model MiniFASNetV2 input 1,3,80,80
convert_model MiniFASNetV1SE input 1,3,80,80
