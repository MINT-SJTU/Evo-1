## Run Aloha (Real Robot)
This example demonstrates how to run EVO1 with a real robot using an ALOHA setup. 

### Environment Requirements

| Component       | Version/Configuration |
|-----------------|-----------------------|
| Operating System| Ubuntu 20.04 LTS      |
| ROS Distribution| ROS1 Noetic           |
| Python Version  | 3.8+ (recommended)    |



## Steps

1. Follow the [hardware installation instructions](https://github.com/tonyzhaozh/aloha?tab=readme-ov-file#hardware-installation) in the ALOHA repo.


2. Modify the network configuration in `Evo1_client_aloha.py`:
   - Update the `IP` and `PORT` parameter to match your real robot's IP address
   - Ensure that the Inference server end and Aloha are in the same network segment



2. Run Evo1_server in Terminal window 1: 
    ```bash
    python Evo1_server.py
    ```

3. Run Evo1_client_aloha.py in Terminal window 2: 
    ```bash
    python Evo1_client_aloha.py
    ```
