## Installation

1. Prepare the environment for Evo-1

```bash
# Clone this repo
git clone https://github.com/DorayakiLin/Evo_1_clean.git


# Create a Conda environment
conda create -n Evo1 python=3.10 -y
conda activate Evo1

# Install requirements
cd Evo_1
pip install -r requirements.txt
MAX_JOBS=64 pip install -v flash-attn --no-build-isolation

```

## Simulation Benchmark

### Meta-World Benchmark

#### 1. Prepare the environment for Meta-World

```bash
# Start a new terminal and create a Conda environment for Meta-World
conda create -n metaworld python=3.10 -y
conda activate metaworld

# Install requirements
pip install mujoco
pip install metaworld
pip install websockets
pip install opencv-python
pip install packaging
```

#### 2. Run the weight and code

##### 2.1 Download Model Weight

[Link to Model Weight for Meta-Wolrd](https://huggingface.co/yinxinyuchen/evo1_metaworld/tree/main/step_65000)

##### 2.2 Modify config

Modify the checkpoint dir to where you download the model weight:
[Modify the checkpoint dir](Evo_1/scripts/Evo1_server.py#L149)

Modify the server port (Optional,default 9000):
[Modify the server port](Evo_1/scripts/Evo1_server.py#L152)

Modify the client port (Optional,default 9000):
[Modify the client port](MetaWorld_evaluation/mt50_evo1_client_prompt.py#L40)

#### 3. Run the simulation evaluation

```bash
# Start Evo-1 server (In terminal 1)
conda activate Evo1
cd Evo_1
python scripts/evo1_server.py

# Start Meta-World client (In terminal 2)
conda activate metaworld
cd MetaWorld_evaluation
python mt50_evo1_client_prompt.py

```

### LIBERO Benchmark
#### 1. Prepare the environment for LIBERO

```bash
conda create -n libero python=3.8.13
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```

#### 2. Run the weight and code
##### 2.1 Download Model Weight
[Link to Model Weight for LIBERO](https://huggingface.co/liujiting/evo1_libero/tree/main)

##### 2.2 Adjust Server

Evo_1_clean/miravla/scripts/evo1_server_json.py

Modify the checkpoint dir to where you download the model weight:
[Modify the checkpoint dir](miravla/scripts/evo1_server_json.py#L149)


#### 3. Run the simulation evaluation
##### Download Model Weight


```bash
# Start Evo-1 server (In terminal 1)
conda activate Evo1
cd Evo_1
python scripts/evo1_server_json.py

# Start LIBERO client (In terminal 2)
conda activate libero
cd LIBERO_evaluation
python libero_client_4tasks.py

```