# LIBERO-Plus Evaluation

This directory provides the Evo-1 evaluation scripts for LIBERO-Plus. The evaluation covers seven perturbation categories: background, camera, language, layout, light, noise, and robot.

## Evaluation Results

The table below reports the Evo-1 success rate on the four LIBERO suites:

| Evo-1-Libero | background | camera | language | layout | light | noise | robot | avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Spatial | 86.43% | 38.03% | 68.97% | 67.53% | 86.64% | 75.21% | 49.14% | 67.42% |
| Object | 91.13% | 60.35% | 78.81% | 71.46% | 88.22% | 76.07% | 49.50% | 73.65% |
| Goal | 83.63% | 50.74% | 38.29% | 52.47% | 89.61% | 66.49% | 48.41% | 61.38% |
| 10 | 80.97% | 30.31% | 68.41% | 59.94% | 67.88% | 63.92% | 50.64% | 60.30% |
| Avg. | 85.54% | 44.86% | 63.62% | 62.85% | 83.09% | 70.42% | 49.42% | 65.69% |

## 1. Install System Dependencies

Tested environment: Ubuntu 22.04, Python 3.10, and x86_64 Linux.

```bash
sudo apt-get update
sudo apt-get install -y \
  libexpat1 libfontconfig1-dev libpython3-stdlib \
  libmagickwand-dev libgl1 libegl1 libosmesa6-dev ffmpeg unzip
```

EGL is recommended when an NVIDIA GPU is available. Use OSMesa for CPU-only offscreen rendering.

## 2. Clone and Patch LIBERO-Plus

Keeping LIBERO-Plus inside this directory is recommended:

```bash
cd /path/to/libero-plus-eval
git clone https://github.com/LinqingZhong/LIBERO-plus.git
cd LIBERO-plus

git apply --check ../libero-plus-eval.patch
git apply ../libero-plus-eval.patch
```

The patch contains the Python packaging, PyTorch 2.6, and fog image-size compatibility fixes required by the evaluation.

## 3. Download Assets

The LIBERO-Plus repository includes the source code, BDDL files, and initial states. The `assets/` directory must be downloaded separately. Run the following commands from the `LIBERO-plus` root directory:

```bash
python -m pip install huggingface_hub
hf download Sylvest/LIBERO-plus assets.zip \
  --repo-type dataset \
  --local-dir /tmp/libero-plus-assets

ASSETS_STAGE="$(mktemp -d /tmp/libero-plus-assets/extracted.XXXXXX)"
unzip /tmp/libero-plus-assets/assets.zip -d "$ASSETS_STAGE"

ASSETS_SOURCE="$(
  find "$ASSETS_STAGE" -type d -path '*/LIBERO-plus-0/assets' \
    -print -quit
)"

test -n "$ASSETS_SOURCE"
test -d "$ASSETS_SOURCE/textures"
test ! -e ./libero/libero/assets

mv "$ASSETS_SOURCE" ./libero/libero/assets
find "$ASSETS_STAGE" -depth -type d -empty -delete
```

If these commands fail, download [assets.zip](https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip) manually and extract its `assets` directory to `LIBERO-plus/libero/libero/assets`.

## 4. Create the Evaluation Environment

Run the following commands from the `LIBERO-plus` root directory:

```bash
conda create -n libero_plus python=3.10 -y
conda activate libero_plus

python -m pip install --upgrade pip
python -m pip install -r ../requirements-eval.txt
python -m pip install -e . --no-deps
```

After installation, verify from outside the repository that the `libero` package is importable:

```bash
(cd /tmp && python -c "import libero; print('LIBERO import OK:', list(libero.__path__))")
```

## 5. Configure Data Paths

Use a separate configuration directory for LIBERO-Plus:

```bash
export LIBERO_CONFIG_PATH="$HOME/.libero-plus"
mkdir -p "$LIBERO_CONFIG_PATH"
cp ../config.yaml.example "$LIBERO_CONFIG_PATH/config.yaml"
```

Edit `$LIBERO_CONFIG_PATH/config.yaml` and replace every occurrence of `/absolute/path/to/LIBERO-plus` with the absolute path to your LIBERO-Plus clone. The `datasets` path does not need to exist for this evaluation.

After opening a new terminal, activate the environment and set the configuration path:

```bash
conda activate libero_plus
export LIBERO_CONFIG_PATH="$HOME/.libero-plus"
```

Adding the export command to the `libero_plus` Conda environment activation script is recommended so that the variable is set automatically.

## 6. Verify the Installation

```bash
python - <<'PY'
from pathlib import Path
from libero.libero import benchmark, get_libero_path

for key in ("benchmark_root", "assets", "bddl_files", "init_states"):
    path = Path(get_libero_path(key))
    assert path.exists(), f"Missing {key}: {path}"
    print(f"{key}: {path}")

print(
    "libero_spatial tasks:",
    benchmark.get_benchmark_dict()["libero_spatial"]().n_tasks,
)
PY
```

## 7. Run the Evaluation

Return to the evaluation directory and select one suite. The launcher runs all seven clients sequentially:

```bash
conda activate libero_plus
export LIBERO_CONFIG_PATH="$HOME/.libero-plus"
cd /path/to/libero-plus-eval

bash test_libero_plus.sh libero_goal \
  --server-url ws://MODEL_SERVER_IP:9003 \
  --output-dir /target/output/path \
  --horizon 15 \
  --num-episodes 1 \
  --seed 0
```

Replace `MODEL_SERVER_IP` and `/target/output/path` with the actual values.

Available suites and their default maximum step counts:

| suite | max steps |
| --- | ---: |
| `libero_spatial` | 660 |
| `libero_object` | 840 |
| `libero_goal` | 900 |
| `libero_10` | 1560 |

Common options:

| Option | Default | Description |
| --- | --- | --- |
| `--server-url` | `ws://127.0.0.1:9003` | Model server WebSocket URL |
| `--output-dir` | `libero-plus-eval/logs/evo1_libero_plus` | Root directory for logs and videos |
| `--horizon` | `15` | Number of actions returned by each model prediction |
| `--num-episodes` | `1` | Number of episodes per task |
| `--seed` | `0` | Random seed |
| `--max-steps` | Suite-specific default | Override the maximum number of execution steps |
| `--mujoco-gl` | `egl` | Rendering backend; use `osmesa` for CPU-only rendering |

The server URL and output directory can also be set through environment variables:

```bash
export LIBERO_PLUS_SERVER_URL=ws://MODEL_SERVER_IP:9003
export LIBERO_PLUS_OUTPUT_DIR=/target/output/path
bash test_libero_plus.sh libero_spatial
```

To evaluate only one perturbation category, invoke its client directly:

```bash
python -m evo_libero_plus_clients.camera --suite libero_goal
```

Use the following command to list all available options:

```bash
python -m evo_libero_plus_clients.camera --help
```

By default, each task runs for one episode. `test_libero_plus.sh` executes the seven clients sequentially and stops if any client fails.
