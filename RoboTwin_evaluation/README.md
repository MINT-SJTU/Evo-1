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

## Inference Settings

The policy is served with:

1. **`horizon=37`** — actions executed per inference call (default in `eval.sh` / `deploy_policy.yml`).
2. **`num_inference_timesteps=50`** in `Evo_1/scripts/Evo1_server.py`.
3. **Gaussian action smoothing, kernel=9** — in `deploy_policy.py`.

## Results — RoboTwin-50 (`demo_clean`)

A **single multi-task policy trained on clean data only** — 50 RoboTwin tasks,
**50 `demo_clean` demonstrations per task** (no randomized / augmented data). At **test time**
each task is rolled out **100 times** (50 tasks × 100 = 5000 evaluation episodes),
`horizon=37`, using the released `MINT-SJTU/Evo1_RoboTwin` checkpoint.

**Overall success rate: 48.0% (2399 / 5000 evaluation rollouts)**

| Task | SR | | Task | SR |
|---|---|---|---|---|
| grab_roller | 100% | | blocks_ranking_rgb | 48% |
| lift_pot | 98% | | click_bell | 48% |
| place_burger_fries | 95% | | handover_block | 35% |
| place_cans_plasticbox | 94% | | place_object_stand | 35% |
| shake_bottle | 94% | | blocks_ranking_size | 34% |
| handover_mic | 93% | | stack_blocks_two | 34% |
| shake_bottle_horizontally | 93% | | place_dual_shoes | 33% |
| adjust_bottle | 90% | | open_microwave | 28% |
| place_empty_cup | 86% | | place_a2b_left | 27% |
| press_stapler | 84% | | place_fan | 26% |
| place_container_plate | 80% | | stamp_seal | 25% |
| click_alarmclock | 79% | | place_object_scale | 23% |
| place_bread_skillet | 78% | | place_mouse_pad | 18% |
| stack_bowls_two | 78% | | turn_switch | 14% |
| move_playingcard_away | 73% | | hanging_mug | 10% |
| place_phone_stand | 72% | | place_a2b_right | 10% |
| beat_block_hammer | 70% | | place_shoe | 9% |
| pick_diverse_bottles | 67% | | move_stapler_pad | 5% |
| move_pillbottle_pad | 61% | | rotate_qrcode | 5% |
| move_can_pot | 59% | | put_object_cabinet | 4% |
| open_laptop | 59% | | place_object_basket | 3% |
| pick_dual_bottles | 58% | | scan_object | 3% |
| dump_bin_bigbin | 56% | | place_can_basket | 0% |
| stack_bowls_three | 55% | | put_bottles_dustbin | 0% |
| place_bread_basket | 50% | | stack_blocks_three | 0% |

Results were independently reproduced.

## Demos

Inference examples:

<table>
  <tr>
    <td width="50%"><video src="https://github.com/user-attachments/assets/8c4bac58-93fe-4187-bcf0-700c0a1a46b3" controls muted width="100%"></video></td>
    <td width="50%"><video src="https://github.com/user-attachments/assets/526c0599-feaf-4c8c-b3f2-6fe710fcdd6f" controls muted width="100%"></video></td>
  </tr>
</table>


