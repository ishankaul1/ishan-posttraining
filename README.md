# ishan-posttraining

TRL-based repo for running SFT and RL (GRPO etc.) experiments on RunPod.

## Next session TODO

- [ ] Grab credentials: `WANDB_API_KEY`, GCS service account JSON, `HF_TOKEN`
- [ ] Base64-encode GCS key: `base64 -i gcs-service-account-key.json | pbcopy`
- [ ] `uv sync` and do a local test run: `./train.sh --config configs/sft_qwen_1.5b.yaml`
- [ ] Build and smoke-test the Docker image
- [ ] Create RunPod template, add secrets (`GCS_KEY_B64`, `WANDB_API_KEY`, `HF_TOKEN`, `GCS_BUCKET`)
- [ ] Fire first run at an A4000 with `configs/sft_qwen_1.5b.yaml`
- [ ] Once SFT works end-to-end, try `configs/grpo_qwen_1.5b_math.yaml`
