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

**Overall success rate: 65.7% (3287 / 5000 evaluation rollouts)**

| Task | SR | | Task | SR |
|---|---|---|---|---|
| adjust_bottle | 100% | | place_phone_stand | 73% |
| handover_mic | 100% | | place_object_stand | 72% |
| stack_blocks_two | 100% | | open_microwave | 70% |
| click_alarmclock | 98% | | place_cans_plasticbox | 70% |
| grab_roller | 98% | | move_can_pot | 69% |
| shake_bottle_horizontally | 98% | | put_bottles_dustbin | 69% |
| place_container_plate | 96% | | place_bread_basket | 63% |
| click_bell | 95% | | place_bread_skillet | 63% |
| dump_bin_bigbin | 95% | | blocks_ranking_size | 58% |
| stack_bowls_two | 93% | | place_can_basket | 50% |
| shake_bottle | 91% | | pick_diverse_bottles | 49% |
| place_burger_fries | 90% | | place_object_scale | 49% |
| stack_blocks_three | 87% | | place_a2b_left | 48% |
| stack_bowls_three | 85% | | put_object_cabinet | 39% |
| press_stapler | 83% | | place_a2b_right | 38% |
| beat_block_hammer | 82% | | place_fan | 34% |
| open_laptop | 82% | | place_shoe | 33% |
| lift_pot | 81% | | rotate_qrcode | 32% |
| move_playingcard_away | 81% | | scan_object | 32% |
| place_object_basket | 79% | | stamp_seal | 28% |
| place_empty_cup | 77% | | turn_switch | 28% |
| blocks_ranking_rgb | 76% | | place_mouse_pad | 17% |
| pick_dual_bottles | 76% | | hanging_mug | 9% |
| move_pillbottle_pad | 75% | | place_dual_shoes | 2% |
| handover_block | 74% | | move_stapler_pad | 0% |

## Demos

Inference examples:

<table>
  <tr>
    <td width="50%"><video src="https://github.com/user-attachments/assets/8c4bac58-93fe-4187-bcf0-700c0a1a46b3" controls muted width="100%"></video></td>
    <td width="50%"><video src="https://github.com/user-attachments/assets/526c0599-feaf-4c8c-b3f2-6fe710fcdd6f" controls muted width="100%"></video></td>
  </tr>
</table>


