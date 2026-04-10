#!/usr/bin/env bash
set -euo pipefail

TEMPLATE_ID="8nnxhdhryv"
DEFAULT_GPU="NVIDIA A4000"

usage() {
  echo "Usage: $(basename "$0") --config <path> [--gpu <gpu type>] [--name <pod name>]"
  echo ""
  echo "Options:"
  echo "  --config   Path to config yaml (e.g. configs/sft_qwen_1.5b.yaml)"
  echo "  --gpu      GPU type (default: '$DEFAULT_GPU')"
  echo "  --name     Pod name (default: derived from config name)"
  echo ""
  echo "Examples:"
  echo "  ./run_experiment.sh --config configs/sft_qwen_1.5b.yaml"
  echo "  ./run_experiment.sh --config configs/grpo_qwen_1.5b_math.yaml --gpu 'NVIDIA A100 80GB PCIe'"
}

CONFIG=""
GPU="$DEFAULT_GPU"
NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config|-c) CONFIG="$2"; shift 2 ;;
    --gpu|-g) GPU="$2"; shift 2 ;;
    --name|-n) NAME="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "Error: --config is required."
  usage
  exit 1
fi

if [[ -z "$NAME" ]]; then
  NAME=$(basename "$CONFIG" .yaml)
fi

echo "[run_experiment.sh] Launching pod..."
echo "  Template: $TEMPLATE_ID"
echo "  GPU:      $GPU"
echo "  Config:   $CONFIG"
echo "  Name:     $NAME"

runpodctl create pod \
  --name "$NAME" \
  --templateId "$TEMPLATE_ID" \
  --gpuType "$GPU" \
  --args "--config $CONFIG"

echo "[run_experiment.sh] Pod launched."
