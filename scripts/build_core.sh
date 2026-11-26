#!/usr/bin/env bash
# Build in the project; never install or modify the host toolchain.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_mode="${1:-portable}"
build_dir="${2:-${root_dir}/build}"
build_jobs="${BUILD_JOBS:-4}"
cmake_args=(-S "$root_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON)

case "$build_mode" in
  portable) cmake_args+=(-DPOWER11_ENABLE_MMA=OFF -DPOWER11_BUILD_CAMERA=OFF) ;;
  power) cmake_args+=(-DPOWER11_ENABLE_MMA=ON -DPOWER11_BUILD_CAMERA=OFF) ;;
  camera) cmake_args+=(-DPOWER11_ENABLE_MMA=OFF -DPOWER11_BUILD_CAMERA=ON) ;;
  power-camera) cmake_args+=(-DPOWER11_ENABLE_MMA=ON -DPOWER11_BUILD_CAMERA=ON) ;;
  *) echo 'Usage: build_core.sh portable|power|camera|power-camera [build-directory]' >&2; exit 2 ;;
esac
if [[ "$build_mode" == *camera ]]; then
  : "${ONNXRUNTIME_ROOT:?Set ONNXRUNTIME_ROOT to a reviewed C/C++ SDK (include/ and lib/)}"
  cmake_args+=("-DONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT")
  if [[ -n "${OpenCV_DIR:-}" ]]; then cmake_args+=("-DOpenCV_DIR=$OpenCV_DIR"); fi
fi
mkdir -p "$build_dir"
cmake "${cmake_args[@]}" 2>&1 | tee "$build_dir/configure.log"
cmake --build "$build_dir" --parallel "$build_jobs" 2>&1 | tee "$build_dir/build.log"
ctest --test-dir "$build_dir" --output-on-failure 2>&1 | tee "$build_dir/ctest.log"
"$build_dir/rfdetr_gemm" --capability | tee "$build_dir/mma-capability.json"
python3 - "$root_dir" "$build_dir" <<'PY'
from pathlib import Path
import sys

root, target = map(Path, sys.argv[1:])
sys.path.insert(0, str(root / 'scripts'))
from run_record import artifact, atomic_json, identity

sources = [root / 'CMakeLists.txt', *sorted((root / 'core').rglob('*.hpp')),
           *sorted((root / 'core').rglob('*.cpp'))]
products = [target / name for name in ('rfdetr_gemm', 'rfdetr_adapter', 'rfdetr_camera',
                                      'configure.log', 'build.log', 'ctest.log', 'compile_commands.json')]
atomic_json(target / 'build-manifest.json', {
    **identity('core_build'), 'status': 'completed',
    'sources': [artifact(path) for path in sources],
    'products': [artifact(path) for path in products if path.is_file()],
    'scope': 'Build and portable correctness tests; MMA runtime execution only when capability permits'})
PY
