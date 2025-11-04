

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
1. Prepare the environment for Meta-World

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
2. Run the simulation evaluation

    ```bash
    # Start Evo-1 server (In terminal 1)
    cd miravla
    python scripts/evo1_server_json.py
    
    # Start Meta-World client (In terminal 2)
    cd metaworld
    python mt50_evo1_client_prompt.py
    
    ```