# RoboTwin Evaluation for Evo-1

Full setup and run instructions live in the top-level
[README → 🧪 RoboTwin Benchmark](../README.md#-robotwin-benchmark).

This directory ships the Evo-1 **policy plugin** for RoboTwin:

```
policy/Evo1/
  deploy_policy.py   # RoboTwin policy interface: obs -> server -> smoothed 14-D action chunk
  deploy_policy.yml  # default config (horizon=37)
  eval.sh            # single-task launcher
```

Install it into a RoboTwin checkout:

```bash
cp -r policy/Evo1  /path/to/RoboTwin/policy/Evo1
```

The Evo-1 server reads `arm_key` / `dataset_key` from each client request, so **no server-side
edit is needed for RoboTwin** — the client sends `arm_key=aloha_joint` and the per-task
`dataset_key=robotwin_<task>` (RoboTwin `norm_stats.json` is keyed per task, 50 keys under
`aloha_joint`).

## ⚠️ Evaluation recipe (all three matter — dropping any one roughly halves success rate)

1. **`horizon=37`** — actions executed per inference call (default in `eval.sh` / `deploy_policy.yml`).
2. **`num_inference_timesteps=50`** in `Evo_1/scripts/Evo1_server.py` (not 32; 32 causes action jitter).
3. **Gaussian action smoothing, kernel=9** — hardcoded in `deploy_policy.py` `eval()`.

Verified on `place_burger_fries` (2/2 success at horizon=37) with the RoboTwin multitask checkpoint.
