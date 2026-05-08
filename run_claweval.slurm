#!/bin/bash
#
# claw-eval SLURM evaluation script
#
# Runs Qwen3-32B (model under evaluation) + GPT OSS 120B (judge + user-agent)
# via two local vLLM servers. Evaluates general + multi_turn splits only
# (multimodal M* tasks skipped — they require Docker).
#
# Usage:
#   sbatch run_claweval.slurm
#
# Override variables at submit time:
#   MODEL=Qwen/Qwen3-32B JUDGE_MODEL=<hf-id> sbatch run_claweval.slurm
#
# ── Variable reference ────────────────────────────────────────────────────────
#
# MODEL
#   HuggingFace model ID for the agent under evaluation.
#   Default: Qwen/Qwen3-32B
#
# MODEL_TP
#   Tensor parallel size for the agent model (number of GPUs).
#   Qwen3-32B in BF16 is ~64GB — fits on 1x 80GB GPU.
#   Default: 1
#
# MODEL_PORT
#   Port for the agent model vLLM server.
#   Default: 8000
#
# JUDGE_MODEL
#   HuggingFace model ID for the judge + user-agent model.
#   gpt-oss-120b is a sparse MoE — fits on 1 GPU.
#   Supported natively in vLLM >= 0.20.1.
#   Default: openai/gpt-oss-120b
#
# JUDGE_TP
#   Tensor parallel size for the judge model.
#   gpt-oss-120b is sparse MoE — fits on 1x 140GB GPU. TP=2 triggers allreduce_rms
#   custom CUDA kernel that segfaults on CUDA 12.4 driver. Use TP=1.
#   Default: 1
#
# JUDGE_PORT
#   Port for the judge model vLLM server.
#   Default: 8001
#
# PARALLEL
#   Number of parallel claw-eval workers (tasks running concurrently).
#   Each worker starts its own set of mock services on offset ports.
#   Default: 8
#
# TRIALS
#   Number of independent trials per task (for pass^k metric).
#   Default: 3
#
# FILTER_GENERAL
#   Substring filter for the general (T*) split.
#   Default: tasks/t
#
# FILTER_MULTI_TURN
#   Substring filter for the multi_turn (C*) split.
#   Default: tasks/c
#   Both filters are run sequentially into the same OUTPUT_DIR so results
#   are merged and the final summary covers all 199 non-multimodal tasks.
#
# OUTPUT_DIR
#   Trace output directory.
#   Default: traces/<model basename>
#
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=claweval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --nodelist=smc-pod-[3,5,8]
#SBATCH --output=logs/claweval_%j.out
#SBATCH --error=logs/claweval_%j.out

set -euo pipefail
cd $SLURM_SUBMIT_DIR

source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

SITE_PACKAGES=$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="${SITE_PACKAGES}/nvidia/cudnn/lib:${SITE_PACKAGES}/nvidia/cusparselt/lib:${SITE_PACKAGES}/nvidia/nccl/lib:${SITE_PACKAGES}/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if python3 -c "import flashinfer_cubin" 2>/dev/null; then
    CUBIN_DIR=$(.venv/bin/python -c "import site; print(site.getsitepackages()[0])")/flashinfer_cubin
    echo "Removing flashinfer_cubin to force JIT mode..."
    rm -rf "${CUBIN_DIR}" "${CUBIN_DIR}"-*.dist-info 2>/dev/null || true
fi

mkdir -p logs

free_port() { python3 -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)"; }

MODEL=${MODEL:-"Qwen/Qwen3-32B"}
MODEL_TP=${MODEL_TP:-"1"}
MODEL_PORT=${MODEL_PORT:-$(free_port)}
ENABLE_THINKING=${ENABLE_THINKING:-"true"}

JUDGE_MODEL=${JUDGE_MODEL:-"openai/gpt-oss-120b"}
JUDGE_TP=${JUDGE_TP:-"1"}
JUDGE_PORT=${JUDGE_PORT:-$(free_port)}

PARALLEL=${PARALLEL:-"8"}
TRIALS=${TRIALS:-"3"}
LANGUAGE=${LANGUAGE:-"en"}
OUTPUT_DIR=${OUTPUT_DIR:-"traces/$(echo ${MODEL} | awk -F/ '{print $NF}')"}

# GPU UUID → index mapping
UUIDS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n')
declare -a VISIBLE_DEVICES=()
if [ ! -z "$UUIDS" ]; then
    for UUID in $UUIDS; do
        if [[ $UUID == GPU-* ]] || [[ $UUID == MIG-* ]]; then
            echo "Processing UUID: $UUID"
            ID=$(nvidia-smi --id=$UUID --query-gpu=index --format=csv,noheader)
            if [ ! -z "$ID" ]; then
                VISIBLE_DEVICES+=($ID)
                echo "Mapped UUID $UUID to GPU ID: $ID"
            else
                echo "Could not map UUID $UUID to index. Keeping UUID."
                VISIBLE_DEVICES+=($UUID)
            fi
        else
            VISIBLE_DEVICES+=($UUID)
        fi
    done
    VISIBLE_DEVICES_STR=$( IFS=$','; echo "${VISIBLE_DEVICES[*]}" )
    echo "Visible devices mapped: $VISIBLE_DEVICES_STR"
    export CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES_STR
fi

if [ ${#VISIBLE_DEVICES[@]} -eq 0 ] && [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra VISIBLE_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
fi

GPU0=${VISIBLE_DEVICES[0]:-0}
GPU1=${VISIBLE_DEVICES[1]:-1}
GPU2=${VISIBLE_DEVICES[2]:-2}

THINKING_FLAG=""
if [[ "$(echo "$ENABLE_THINKING" | tr '[:upper:]' '[:lower:]')" == "false" ]]; then
    THINKING_FLAG="--chat-template-kwargs {\"enable_thinking\": false}"
fi

# Start agent model vLLM server
echo "Starting agent vLLM server: ${MODEL} on port ${MODEL_PORT} (TP=${MODEL_TP})..."
CUDA_VISIBLE_DEVICES=${GPU0} .venv/bin/vllm serve "${MODEL}" \
    --port "${MODEL_PORT}" \
    --tensor-parallel-size "${MODEL_TP}" \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --attention-backend FLASH_ATTN \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    ${THINKING_FLAG} &
MODEL_PID=$!

# Wait for agent model vLLM to be ready
echo "Waiting for agent model vLLM server to be ready..."
until curl -sf "http://localhost:${MODEL_PORT}/health" > /dev/null 2>&1 && kill -0 "${MODEL_PID}" 2>/dev/null; do
    if ! kill -0 "${MODEL_PID}" 2>/dev/null; then
        echo "ERROR: vLLM server (PID ${MODEL_PID}) died before becoming ready. Exiting."
        exit 1
    fi
    sleep 5
done
echo "Agent model vLLM server ready."

# Start judge model vLLM server
echo "Starting judge vLLM server: ${JUDGE_MODEL} on port ${JUDGE_PORT} (TP=${JUDGE_TP})..."
CUDA_VISIBLE_DEVICES=${GPU1} .venv/bin/vllm serve "${JUDGE_MODEL}" \
    --port "${JUDGE_PORT}" \
    --tensor-parallel-size "${JUDGE_TP}" \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --max-model-len 32768 \
    --served-model-name "$(echo ${JUDGE_MODEL} | sed 's|openai/||')" &
JUDGE_PID=$!

# Wait for judge model vLLM to be ready
echo "Waiting for judge model vLLM server to be ready..."
until curl -sf "http://localhost:${JUDGE_PORT}/health" > /dev/null 2>&1 && kill -0 "${JUDGE_PID}" 2>/dev/null; do
    if ! kill -0 "${JUDGE_PID}" 2>/dev/null; then
        echo "ERROR: vLLM server (PID ${JUDGE_PID}) died before becoming ready. Exiting."
        exit 1
    fi
    sleep 5
done
echo "Judge model vLLM server ready."

# Rewrite config with actual dynamic ports
python3 - <<PYEOF
import yaml, pathlib
cfg = yaml.safe_load(pathlib.Path('config_qwen3_32b.yaml').read_text())
cfg['model']['base_url'] = 'http://localhost:${MODEL_PORT}/v1'
cfg['judge']['base_url'] = 'http://localhost:${JUDGE_PORT}/v1'
cfg['judge']['model_id'] = cfg['judge']['model_id'].replace('openai/', '')
cfg['user_agent_model']['base_url'] = 'http://localhost:${JUDGE_PORT}/v1'
cfg['user_agent_model']['model_id'] = cfg['user_agent_model']['model_id'].replace('openai/', '')
pathlib.Path('config_qwen3_32b.yaml').write_text(yaml.dump(cfg, default_flow_style=False))
print(f"Config updated: model={cfg['model']['base_url']} judge={cfg['judge']['base_url']} judge_model={cfg['judge']['model_id']}")
PYEOF

# Run claw-eval
echo "Running claw-eval batch (general split: T* tasks)..."
echo "  Model   : ${MODEL}"
echo "  Judge   : ${JUDGE_MODEL}"
echo "  Language: ${LANGUAGE:-all}"
echo "  Parallel: ${PARALLEL}  Trials: ${TRIALS}"
echo "  Output  : ${OUTPUT_DIR}"

LANGUAGE_ARGS=()
if [ -n "${LANGUAGE}" ]; then
    LANGUAGE_ARGS=("--language" "${LANGUAGE}")
fi

.venv/bin/claw-eval batch \
    --config config_qwen3_32b.yaml \
    --model "${MODEL}" \
    --filter "tasks/t" \
    "${LANGUAGE_ARGS[@]}" \
    --parallel "${PARALLEL}" \
    --trials "${TRIALS}" \
    --trace-dir "${OUTPUT_DIR}" \
    --continue "${OUTPUT_DIR}"

EXIT_CODE_GENERAL=$?

echo ""
echo "Running claw-eval batch (multi_turn split: C* tasks)..."

.venv/bin/claw-eval batch \
    --config config_qwen3_32b.yaml \
    --model "${MODEL}" \
    --filter "tasks/c" \
    "${LANGUAGE_ARGS[@]}" \
    --parallel "${PARALLEL}" \
    --trials "${TRIALS}" \
    --trace-dir "${OUTPUT_DIR}" \
    --continue "${OUTPUT_DIR}"

EXIT_CODE_MULTI=$?

# Use the worse exit code
if [ $EXIT_CODE_GENERAL -ne 0 ]; then
    EXIT_CODE=$EXIT_CODE_GENERAL
elif [ $EXIT_CODE_MULTI -ne 0 ]; then
    EXIT_CODE=$EXIT_CODE_MULTI
else
    EXIT_CODE=0
fi

# Shut down vLLM servers
echo "Shutting down vLLM servers..."
kill "${MODEL_PID}" "${JUDGE_PID}" 2>/dev/null || true
wait "${MODEL_PID}" "${JUDGE_PID}" 2>/dev/null || true

# Print results summary
echo ""
echo "=== Results Summary ==="
.venv/bin/python score_summary.py "${OUTPUT_DIR}" 2>/dev/null || true

exit $EXIT_CODE
