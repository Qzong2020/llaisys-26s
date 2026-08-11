#!/bin/bash
# Run all NVIDIA operator tests on MetaX C500.
# C500-specific: (1) Linear F32 GEMM needs allow_tf32=False so PyTorch reference
# matches LLAISYS's full-precision cuBLAS math mode; (2) Self-Attention reference
# creates CPU mask without device=, so we set torch default device to CUDA.
set -euo pipefail
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH="/opt/maca/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"

SHA=$(sha256sum python/llaisys/libllaisys/libllaisys.so | cut -d' ' -f1)
echo "=== C500 NVIDIA operator regression ==="
echo "Build: $SHA"
echo

failed=0
total=0

run_op_test() {
    local name="$1"; local script="$2"
    total=$((total + 1))
    printf "%-20s " "${name}:"
    local out
    out=$(python -c "
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_default_device('cuda')
import sys, os
sys.path.insert(0, 'test')
sys.argv = ['${script}', '--device', 'nvidia']
import runpy
runpy.run_path('${script}', run_name='__main__')
" 2>&1)
    if echo "$out" | grep -q 'Test passed'; then
        echo "PASS"
    else
        echo "FAIL"
        echo "$out" | tail -5
        failed=$((failed + 1))
    fi
}

# Runtime test doesn't need torch fixups
echo -n "Runtime:             "
if PYTHONPATH="$PWD/python:${PYTHONPATH:-}" LD_LIBRARY_PATH="/opt/maca/lib:${LD_LIBRARY_PATH:-}" \
   python test/test_runtime.py --device nvidia 2>&1 | grep -q 'Test passed'; then
    echo "PASS"
else
    echo "FAIL"
    failed=$((failed + 1))
fi
total=$((total + 1))

run_op_test "Add"        "test/ops/add.py"
run_op_test "Argmax"     "test/ops/argmax.py"
run_op_test "Embedding"  "test/ops/embedding.py"
run_op_test "Linear"     "test/ops/linear.py"
run_op_test "RMSNorm"    "test/ops/rms_norm.py"
run_op_test "RoPE"       "test/ops/rope.py"
run_op_test "Self-Attn"  "test/ops/self_attention.py"
run_op_test "SwiGLU"     "test/ops/swiglu.py"

echo
echo "=== Result: $((total - failed))/$total passed ==="
exit $failed
