# ishan-posttraining

TRL-based repo for running SFT and RL (GRPO etc.) experiments on RunPod.

## Next session TODO

- [x] Grab credentials: `WANDB_API_KEY`, GCS service account JSON, `HF_TOKEN`
- [x] Base64-encode GCS key: `base64 -i gcs-service-account-key.json | pbcopy`
- [x] `uv sync` and do a local test run
- [x] Build and push Docker image (`./docker_push.sh`)
- [x] Create RunPod template, add secrets
- [ ] Fire first run at an A4000: `./run_experiment.sh --config configs/sft_qwen_1.5b.yaml`
- [ ] Once SFT works end-to-end, try `./run_experiment.sh --config configs/grpo_qwen_1.5b_math.yaml`
