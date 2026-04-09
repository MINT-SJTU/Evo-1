# Detailed Dataset Configuration Guide

## Step 1: Define or Choose a Suite
Different robot datasets (e.g., Franka, ALOHA, DROID) vary in state and action keys, dimensions, and gripper control logic. We provide pre-defined "Suites" to handle these differences. Similar datasets can share the same suite. You just need to specify the corresponding suite name in the `process_suite` field of your configuration file. For details, check [`Evo_1/dataset/dataset_process_suite.py`](dataset_process_suite.py).

### 1. How to choose an existing Suite?
You can directly use the following registered Suites based on your dataset format:
- `franka_ee_pose`: For End-Effector Pose-based Franka datasets (e.g., LIBERO, austin_buds). It computes relative actions (deltas) for the first 6 dimensions (pose) while strictly preserving the absolute value of the gripper (7th dimension) to prevent logic breaking.
- `franka_joint_angle`: For joint angle-controlled Franka datasets (e.g., RLBench). Computes relative actions for the first 7 dimensions (joint angles) and keeps the gripper absolute.
- `droid_eef`: Customized for DROID datasets. Extacts and concatenates 6D Cartesian coordinates and 1D gripper data. Applies `1 - value` inversion to the gripper to align with our standard (0=closed, 1=open), and only computes relative actions for the first 6 coordinate dimensions.
- `aloha_joint_angle`: For ALOHA dual-arm datasets (e.g., RoboTwin). Handles 14-dimensional data (7 dim left + 7 dim right) and automatically skips gripper indices (6 and 13) when computing relative actions.
- `metaworld`: For MetaWorld datasets. Since the action data is usually already in relative format (delta xyz + gripper), this Suite does not compute additional deltas.
- `default`: Basic processing logic. Reads `observation.state` and `action`, and attempts to compute relative actions for `[0:6]` and `[7:13]`.

### 2. What if keys or dimensions don't match exactly?
No need to modify code. All built-in Suites support overriding default parameters via the `suite_config` dictionary in your config file. For example, if your state key is `env.state` instead of `observation.state`, simply specify `state_key: env.state` in `suite_config`. You can modify key names, gripper indices (`gripper_indices`), and the dimensions to compute deltas for (`ee_pose_dims` / `joint_dims`, etc.).

### 3. What if existing Suites don't fit at all?
If your dataset is highly customized, you have three options:
- **Method A**: Use the `custom` Suite. This is completely driven by the configuration file. In `suite_config`, you can pass `action_concat_keys` (to concatenate multiple action fields) and `relative_dims` (to specify which dimensions to convert to relative actions).
- **Method B**: Directly modify the `custom` Suite code to meet your needs.
- **Method C**: Write a new Suite. Open `dataset_process_suite.py`, create a new class inheriting from `BaseProcessSuite`, implement `extract_state`, `extract_actions`, and `_convert_to_relative`, then register it via `register_suite("your_suite_name", YourSuiteClass)`.

## Step 2: Modify config.yaml
After deciding on a suite, edit the dataset configuration file `dataset/config.yaml`. (Location: [`Evo_1/dataset/config.yaml`](config.yaml))

Example configuration:
```yaml
max_action_dim: 24
max_state_dim: 24
max_views: 3
normalization_type: bounds # options: bounds, normal, bounds_q99
datasets_manifest: datasets_manifest.pkl # Example: dataset_manifest_eef_libero_10_lerobot21.pkl

decoder_cache_size: 16             # video decoder cache size
cache_shard_size_bytes: 268435456  # shard size for caching decoded videos, default 256MB

data_groups:
  franka_ee_pose_delta:
    libero_10_no_noops_lerobot:
      path: /share/taolin/datasets/libero/libero_10_no_noops_1.0.0_lerobot
      view_map:
        image_1: observation.images.image
        image_2: observation.images.wrist_image
      use_delta_action: False
      process_suite: franka_ee_pose # must set
      suite_config:                 # optional, below is an example of suite_config
        state_key: observation.state
        action_key: action
        gripper_indices: [6]
        ee_pose_dims: [0, 6]
        
    libero_goal_no_noops_lerobot:
      path: /share/taolin/datasets/libero/libero_goal_no_noops_1.0.0_lerobot
      view_map:
        image_1: observation.images.image
        image_2: observation.images.wrist_image
      use_delta_action: False
      process_suite: franka_ee_pose
        
  franka_ee_abs_pose:
    droid_101_eef_lerobotv21:
      path: /share/taolin/datasets/pretraining_data/droid_1.0.1
      view_map:
        image_1: observation.images.exterior_1_left
        image_2: observation.images.wrist_left
        image_3: observation.images.exterior_2_left
      use_delta_action: True
      process_suite: droid_eef
      suite_config:
        state_key: observation.state.cartesian_position
        action_key: action.cartesian_position
        action_gripper_key: action.gripper_position
        state_gripper_key: observation.state.gripper_position
```
Parameter details:
- `max_action_dim`: Max action dimensions accepted by the model. Do NOT modify.
- `max_state_dim`: Max state dimensions accepted by the model. Do NOT modify.
- `max_views`: Max number of image views supported. Fixed at 3. Do NOT modify.
- `normalization_type`: Normalization method (`bounds`, `normal`, or `bounds_q99`). `bounds` is tested and recommended.
- `datasets_manifest`: Name of the `.pkl` checklist file that tracks dataset cache locations. Cached in `/dataset/manifest/`. If a file with this name exists, it skips generation and reuses it. **Always change this filename** (e.g., to `manifest_libero_10.pkl`) when modifying the dataset combination in training to avoid loading stale cache.
- `data_groups`: Top-level dictionary grouping datasets by robot hardware and action space format (e.g., `franka_ee_pose_delta`). Each group contains specific datasets (e.g., `libero_10`, `libero_spatial`).

Dataset-specific parameters:
- `path`: Absolute storage path of the dataset.
- `view_map`: Maps image column names in parquet files (e.g., `observation.images.rgb_static`) to unified view IDs (`image_1`, `image_2`, `image_3`).
- `use_delta_action`: Whether to compute relative actions (`True` or `False`). If the dataset already contains relative actions (e.g., Libero), set to `False`. Defaults to `False`.
- `process_suite`: Specifies the code suite for parsing states and actions. Must exactly match a registered name in `dataset_process_suite.py` (e.g., `franka_ee_pose`, `droid_eef`).
- `suite_config`: (Optional) Advanced dictionary to override specific suite parameters (e.g., `state_key`, `action_key`, `gripper_indices`, `ee_pose_dims`). Omitted keys will fall back to their default values defined in the `process_suite`.

Once you write the suite and configure `config.yaml` as shown, dataset preparation is complete!