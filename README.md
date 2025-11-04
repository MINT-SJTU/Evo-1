

## Installation
1. Prepare the environment for Evo-1
    ```bash
    # Clone this repo
    git clone https://github.com/DorayakiLin/Evo_1_clean.git
    
    
    # Create a Conda environment
    conda create -n Evo1 python=3.10 -y
    conda activate Evo1
    
    # Install requirements
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
```

#### 2. Run the weight and code
##### 2.1 Download Model Weight
[Link to Model Weight for Meta-Wolrd](https://huggingface.co/yinxinyuchen/evo1_metaworld/tree/main/step_65000)

##### 2.2 Adjust Server

Evo_1_clean/miravla/scripts/evo1_server_json.py

Modify the checkpoint dir of the 149 line to where you download the model weight:
[this file](miravla/scripts/evo1_server_json.py#L149)
```bash
ckpt_dir = "/home/dell/checkpoints/Evo1_700m/evo1_metaworld/step_65000/"
```

#### 2. Run the simulation evaluation
##### Download Model Weight


```bash

cd miravla
python scripts/evo1_server_json.py

# Start Meta-World client (In terminal 2)
cd metaworld
python mt50_evo1_client_prompt.py

```