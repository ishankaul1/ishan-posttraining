#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") --config <path> [extra args forwarded to train.py...]"
  echo "Example:"
  echo "  ./train.sh --config configs/sft_qwen_1.5b.yaml"
  echo ""
  echo "Required env vars:"
  echo "  GCS_BUCKET            - GCS bucket name for checkpoint sync"
  echo "  GCS_KEY_B64           - Base64-encoded GCS service account JSON (from RunPod secret)"
  echo "  WANDB_API_KEY         - Weights & Biases API key (from RunPod secret)"
  echo "  HF_TOKEN              - HuggingFace token for gated models (from RunPod secret)"
}

CONFIG=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config|-c) CONFIG="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "Error: --config is required."
  usage
  exit 1
fi

# ── Credentials ──────────────────────────────────────────────────────────────

if [[ -n "${GCS_KEY_B64:-}" ]]; then
  echo "[train.sh] Decoding GCS credentials..."
  echo "$GCS_KEY_B64" | base64 -d > /tmp/gcs-key.json
  export GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcs-key.json
  gcloud auth activate-service-account --key-file=/tmp/gcs-key.json --quiet
else
  echo "[train.sh] WARNING: GCS_KEY_B64 not set — checkpoint sync will be skipped"
fi

# ── Experiment folder ─────────────────────────────────────────────────────────

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
CONFIG_STEM=$(basename "$CONFIG" .yaml)
EXPERIMENT_FOLDER="experiments/${CONFIG_STEM}/${CURRENT_TIME}"
OUTPUT_DIR="checkpts/"
mkdir -p "$OUTPUT_DIR"

echo "[train.sh] Config:      $CONFIG"
echo "[train.sh] Output dir:  $OUTPUT_DIR"
echo "[train.sh] GCS path:    gs://${GCS_BUCKET:-<not set>}/${EXPERIMENT_FOLDER}"

# ── Background GCS sync ───────────────────────────────────────────────────────

if [[ -n "${GCS_BUCKET:-}" && -n "${GCS_KEY_B64:-}" ]]; then
  echo "[train.sh] Starting background sync (every 5 min) to gs://${GCS_BUCKET}/${EXPERIMENT_FOLDER}"
  (
    while true; do
      gcloud storage rsync "$OUTPUT_DIR" "gs://${GCS_BUCKET}/${EXPERIMENT_FOLDER}/" --recursive 2>/dev/null || true
      sleep 300
    done
  ) &
  SYNC_PID=$!
  trap "kill $SYNC_PID 2>/dev/null; wait $SYNC_PID 2>/dev/null" EXIT
fi

# ── Train ─────────────────────────────────────────────────────────────────────

echo "[train.sh] Launching training..."
accelerate launch train.py --config "$CONFIG" "${EXTRA_ARGS[@]}"

# ── Final sync ────────────────────────────────────────────────────────────────

if [[ -n "${GCS_BUCKET:-}" && -n "${GCS_KEY_B64:-}" ]]; then
  echo "[train.sh] Training complete. Running final sync..."
  gcloud storage rsync "$OUTPUT_DIR" "gs://${GCS_BUCKET}/${EXPERIMENT_FOLDER}/" --recursive
  echo "[train.sh] Done. Checkpoints at gs://${GCS_BUCKET}/${EXPERIMENT_FOLDER}"
fi
