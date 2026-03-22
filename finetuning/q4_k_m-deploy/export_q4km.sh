#!/bin/bash
# Export QLoRA adapter to GGUF Q4_K_M format
# Run this on Vast.ai where the adapter and GPU are available.
#
# Usage:
#   bash export_q4km.sh /path/to/adapter/dir
#
# Example:
#   bash export_q4km.sh /root/OS_AI_model/finetuning/output/qlora-run

set -euo pipefail

ADAPTER_DIR="${1:-/root/OS_AI_model/finetuning/output/qlora-run}"
MERGED_DIR="${ADAPTER_DIR}/merged"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}"
LLAMA_CPP_DIR="/root/llama.cpp"

echo "============================================================"
echo "QLoRA → GGUF Q4_K_M Export"
echo "============================================================"
echo "Adapter:  ${ADAPTER_DIR}"
echo "Merged:   ${MERGED_DIR}"
echo "Output:   ${OUTPUT_DIR}"
echo "============================================================"

# Step 1: Merge LoRA adapter into base model
echo ""
echo "[1/3] Merging LoRA adapter with base model..."
python3 -c "
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

print('Loading adapter...')
model = AutoPeftModelForCausalLM.from_pretrained(
    '${ADAPTER_DIR}',
    torch_dtype='auto',
    device_map='cpu',
    trust_remote_code=True
)

print('Merging and unloading LoRA weights...')
model = model.merge_and_unload()

print('Saving merged model to ${MERGED_DIR}...')
model.save_pretrained('${MERGED_DIR}')

tokenizer = AutoTokenizer.from_pretrained('${ADAPTER_DIR}', trust_remote_code=True)
tokenizer.save_pretrained('${MERGED_DIR}')

print('Merge complete.')
"

# Step 2: Clone and build llama.cpp if not present
if [ ! -d "${LLAMA_CPP_DIR}" ]; then
    echo ""
    echo "[2/4] Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "${LLAMA_CPP_DIR}"
    pip install -r "${LLAMA_CPP_DIR}/requirements.txt" 2>/dev/null || true
else
    echo ""
    echo "[2/4] llama.cpp already present at ${LLAMA_CPP_DIR}"
fi

# Build llama-quantize if not already built
if [ ! -f "${LLAMA_CPP_DIR}/build/bin/llama-quantize" ]; then
    echo ""
    echo "[2/4] Building llama.cpp (needed for quantization)..."
    cd "${LLAMA_CPP_DIR}"
    cmake -B build -DGGML_CUDA=ON 2>/dev/null || cmake -B build
    cmake --build build --config Release -j$(nproc) --target llama-quantize
    cd -
fi

# Step 3: Convert HF model to GGUF F16 (intermediate format)
F16_GGUF="${OUTPUT_DIR}/qwen3.5-4b-os-f16.gguf"
echo ""
echo "[3/4] Converting to GGUF F16 (intermediate)..."
python3 "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" \
    "${MERGED_DIR}" \
    --outfile "${F16_GGUF}" \
    --outtype f16

# Step 4: Quantize F16 → Q4_K_M
Q4_GGUF="${OUTPUT_DIR}/qwen3.5-4b-os-q4km.gguf"
echo ""
echo "[4/4] Quantizing F16 → Q4_K_M..."
"${LLAMA_CPP_DIR}/build/bin/llama-quantize" \
    "${F16_GGUF}" \
    "${Q4_GGUF}" \
    Q4_K_M

# Clean up intermediate F16 file (large, ~8GB)
echo ""
echo "Cleaning up intermediate F16 file..."
rm -f "${F16_GGUF}"

echo ""
echo "============================================================"
echo "Export complete!"
echo "GGUF file: ${Q4_GGUF}"
echo "Size: $(du -h "${Q4_GGUF}" | cut -f1)"
echo ""
echo "Transfer to local machine:"
echo "  croc send ${Q4_GGUF}"
echo "============================================================"
