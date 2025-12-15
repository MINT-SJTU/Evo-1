
# Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment

[![📄 Paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2511.04555)  

[![🤗 HuggingFace Models](https://img.shields.io/badge/HuggingFace-Evo1_MetaWorld_Model-yellow)](https://huggingface.co/MINT-SJTU/Evo1_MetaWorld/tree/main)  

[![🤗 HuggingFace Models](https://img.shields.io/badge/HuggingFace-Evo1_LIBERO_Model-yellow)](https://huggingface.co/MINT-SJTU/Evo1_LIBERO/tree/main) 

[![📦 Dataset](https://img.shields.io/badge/HuggingFace-Dataset_MetaWorld-orange)](https://huggingface.co/datasets/MINT-SJTU/Evo1_MetaWorld_Dataset/tree/main)  


[![🌍 Website](https://img.shields.io/badge/Github-Website-green)](https://mint-sjtu.github.io/Evo-1.io/)  



## 📰 News  
- 🗓️ **2025-12-15** — Added Evo-1 inference code in Aloha dual arm (Implemented by community user @meijie-jesse)
- 🗓️ **2025-11-15** — Added Evo-1 inference in the LeRobot framework for SO100/SO101
- 🗓️ **2025-11-10** — Released inference script in xarm6
- 🗓️ **2025-11-06** — Released Meta-World & LIBERO evaluation scripts  
- 🗓️ **2025-11-06** — Uploaded model weights to HuggingFace  
- 🗓️ **2025-11-06** — Released official code  




## ✅ To-Do List  

- ✅ Release inference script in xarm6 
- ✅ Add Evo-1 to the LeRobot framework for SO100/SO101   
- ⬜ Release instructions for deploying Evo-1 on Jetson Orin
- ⬜ Release results of all 50 RoboTwin tasks
- ⬜ Release RoboTwin evaluation script  
  



## ⚙️ Installation

Prepare the environment for Evo-1

```bash
# Clone this repo
git clone https://github.com/MINT-SJTU/Evo-1.git

cd Evo-1/

# Create a Conda environment
conda create -n Evo1 python=3.10 -y

conda activate Evo1

# Install requirements
cd Evo_1

pip install -r requirements.txt

# You may need to reduce MAX_JOBS to suit your computer
# (!!! This is a critical step — skipping it may cause lower success rate or unstable robot motion !!!)
MAX_JOBS=64 pip install -v flash-attn --no-build-isolation
```

##  Simulation Benchmark

### 🧪 Meta-World Benchmark

### 1️⃣ Prepare the environment for Meta-World

```bash
conda create -n metaworld python=3.10 -y
conda activate metaworld
pip install mujoco
pip install metaworld
pip install websockets
pip install opencv-python
pip install packaging
pip install huggingface_hub
```

### 2️⃣ Model Preparation

### 📥 2.1 Download Model Weight

```bash
hf download MINT-SJTU/Evo1_MetaWorld --local-dir /path/to/save/checkpoint/
```


### ✏️ 2.2 Modify config

Modify checkpoint dir: [Evo1_server.py#L149](Evo_1/scripts/Evo1_server.py#L149)  
(Optional) Modify server port: [Evo1_server.py#L152](Evo_1/scripts/Evo1_server.py#L152)  
(Optional) Modify client port: [mt50_evo1_client_prompt.py#L40](MetaWorld_evaluation/mt50_evo1_client_prompt.py#L40)


### 3️⃣ Run Meta-World Evaluation

```bash
# Terminal 1
conda activate Evo1

cd Evo_1

python scripts/Evo1_server.py
```

```bash
# Terminal 2
conda activate metaworld

cd MetaWorld_evaluation

python mt50_evo1_client_prompt.py
```

---

### 🧪 LIBERO Benchmark

### 1️⃣ Prepare the environment for LIBERO

```bash
conda create -n libero python=3.8.13 -y

conda activate libero

cd LIBERO_evaluation/

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git

cd LIBERO

pip install -r requirements.txt

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -e .

pip install websockets

pip install huggingface_hub
```

### 2️⃣ Model Preparation

### 📥 2.1 Download Model Weight

```bash
hf download MINT-SJTU/Evo1_LIBERO --local-dir /path/to/save/checkpoint/
```



### ✏️ 2.2 Modify config
Modify checkpoint dir: [Evo1_server.py#L149](Evo_1/scripts/Evo1_server.py#L149)  
Modify ckpt name: [libero_client_4tasks.py#L24](LIBERO_evaluation/libero_client_4tasks.py#L24)  
(Optional) Modify server port: [Evo1_server.py#L152](Evo_1/scripts/Evo1_server.py#L152)  
(Optional) Modify client port: [libero_client_4tasks.py#L23](LIBERO_evaluation/libero_client_4tasks.py#L23)  


#### 3️⃣ Run LIBERO Evaluation

```bash
# Terminal 1
conda activate Evo1

cd Evo_1

python scripts/Evo1_server.py
```

```bash
# Terminal 2
conda activate libero

cd LIBERO_evaluation

python libero_client_4tasks.py
```

## 🧠 Training on Your Own Dataset

We support **lerobot v2.1** format, please convert your data to this format.

We use MetaWorld Dataset here as an example.

### 📥 1. Download Dataset

```bash
mkdir Evo1_training_dataset/

cd Evo1_training_dataset/

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/MINT-SJTU/Evo1_MetaWorld_Dataset

cd Evo1_MetaWorld_Dataset/

git lfs pull
```

### ✏️ 2 Modify config

### ✏️ 2.1 Modify config.yaml

You need to modify the [config.yaml](Evo_1/dataset/config.yaml)

This is used to set the dataset path and the camera mapping.

### ✏️ 2.2 Set the cache path

You need to change the [cache_dir](Evo_1/dataset/lerobot_dataset_pretrain_mp.py#L174)

Set the cache path so the dataset can be loaded from .pkl files next time for faster loading.

### 🚀 3 Start Training

We use the two-stage training paradigm.

### 🚀 3.1 Setup deepspeed

```bash
accelerate config     
```
You can check this [setup guide](deepspeed_setup_example.txt)

<br>

### 🚀 3.2 Stage 1

We only train the integration module and action expert in stage 1.   

If you are training with multiple GPU, set --num_processes to the GPU number.  
You need to change the --run_name,--save_dir,--resume_path base on your own config.

```bash
conda activate Evo1

cd Evo_1/

accelerate launch --num_processes 1 --num_machines 1 --deepspeed_config_file ds_config.json scripts/train.py --run_name Evo1_metaworld_stage1 --action_head flowmatching --use_augmentation --lr 1e-5 --dropout 0.2 --weight_decay 1e-3 --batch_size 16 --image_size 448 --max_steps 5000 --log_interval 10 --ckpt_interval 2500 --warmup_steps 1000 --grad_clip_norm 1.0 --num_layers 8 --horizon 50 --finetune_action_head --disable_wandb --vlm_name OpenGVLab/InternVL3-1B --dataset_config_path dataset/config.yaml --per_action_dim 24 --state_dim 24 --save_dir /your/path/checkpoints/stage1
```
<br>

### 🚀 3.3 Stage 2
We perform Full-scale training in stage 2.   

```bash
conda activate Evo1

cd Evo_1/

accelerate launch --num_processes 1 --num_machines 1 --deepspeed_config_file ds_config.json scripts/train.py --run_name Evo1_metaworld_stage2 --action_head flowmatching --use_augmentation --lr 1e-5 --dropout 0.2 --weight_decay 1e-3 --batch_size 16 --image_size 448 --max_steps 80000 --log_interval 10 --ckpt_interval 2500 --warmup_steps 1000 --grad_clip_norm 1.0 --num_layers 8 --horizon 50 --finetune_vlm --finetune_action_head --disable_wandb --vlm_name OpenGVLab/InternVL3-1B --dataset_config_path dataset/config.yaml --per_action_dim 24 --state_dim 24 --save_dir /your/path/checkpoints/stage2 --resume --resume_pretrain --resume_path /your/path/checkpoints/stage1/step_5000
```

<br>

### 🚀 3.4 (Optional) Resume
If you want to resume the training process, you can use the following command (we use stage 2 as an example):

```bash
accelerate launch --num_processes 1 --num_machines 1 --deepspeed_config_file ds_config.json scripts/train.py --run_name Your_own_name --action_head flowmatching --use_augmentation --lr 1e-5 --dropout 0.2 --weight_decay 1e-3 --batch_size 16 --image_size 448 --max_steps 80000 --log_interval 10 --ckpt_interval 2500 --warmup_steps 1000 --grad_clip_norm 1.0 --num_layers 8 --horizon 50 --finetune_vlm --finetune_action_head --disable_wandb --vlm_name OpenGVLab/InternVL3-1B --dataset_config_path dataset/config.yaml --per_action_dim 24 --state_dim 24 --save_dir /your/path/to/save/the/checkpoints/ --resume  --resume_path /the/checkpoint/path/you/want/to/resume/from/step_20000
```


## 🦾 4. Inference in Your Own Embodiment
We provide an example of inference client script [Evo1_client_xarm6](Evo_1/scripts/Evo1_client_xarm6.py) for xArm6.

The key is to construct an observation dict and pass it to the server.
```bash

      obs = {
            # You need to change the image size to 448x448 before send in obs
            "image": [base_proc.tolist(), wrist_proc.tolist(), dummy_proc.tolist()],  
            # This shows which image is valid.
            "image_mask": [int(i) for i in [1, 1, 0]],
            # This is the state of the robot.
            "state": state.astype(float).tolist(),
            # This is the action mask that shows which action is valid.
            "action_mask": [[int(i) for i in action_mask[0]]],
            # This is the instruction of the task
            "prompt": task_instruction

      }

      try:
            # Send the observation to the server
            await ws.send(json.dumps(obs))
            result = await ws.recv()
            # Get the action chunk
            action_chunk = torch.tensor(json.loads(result))
            
            
      except Exception as e:
            print(f"❌ Inference Error: {e}")
            await asyncio.sleep(0.5)
            continue


```
## 🤖 5.Inference in Lerobot SO100/SO101

We add our policy in /so100_evo1/lerobot-main/src/lerobot/policies/evo1/

### 🔧 5.1 Environment Setup for Collecting LeRobot v2.1 Data 

The environment for data collection is different from the environment used for evaluation, because collecting demonstrations requires compatibility with the LeRobot v2.1 dataset format.
```bash

# Create and activate the conda environment for data collection
conda create -y -n lerobot python=3.10
conda activate lerobot

# Clone the LeRobot repository
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# Checkout the version compatible with v2.1 data format
git checkout v0.3.2

pip install -e .

pip install -e ".[feetech]"
```

### 🔧 5.2 Environment Setup for Evaluation
```bash
#Prepare the environment for Evo1_SO100
cd Evo_1/so100_evo1/

conda create -n Evo1_SO100 python=3.10

conda activate Evo1_SO100

#Install FlashAttention
wget https://ghproxy.net/https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

pip install flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

#Install LeRobot
conda install ffmpeg -c conda-forge

cd lerobot-main

pip install -e.

pip install -e ".[feetech]"

cd Evo_1/so100_evo1/

#Set your own LEROBOT_HOME which include the calibration file of so100
export HF_LEROBOT_HOME="Adress of your own LEROBOT_HOME"

pip install transformers accelerate

pip install timm
```
### ✏️ 5.3 Checkpoint modification

After you trained your model, you need to modify the checkpoint file to make it compatible with Lerobot SO100.

#### 5.3.1 Change the name of the config file
Rename the original file "config.json" to "model_config.json"

#### 5.3.2 Change camera name and image shape

Create a new config.json based on model_config.json.

We provide an example in [SO100_example_checkpoint](https://huggingface.co/MINT-SJTU/Evo1_SO100/tree/main)
```bash
hf download MINT-SJTU/Evo1_SO100 --local-dir /path/to/save/checkpoint/
```



The key is to change the camera name, image shape and rewrite the config.json to satisfy the Lerobot framework.

### 🚀 5.4 Run the Lerobot SO100/SO101

```bash
#Run the command
cd Evo-1/so100_evo1

lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACMXXXXXXX \
    --robot.id=your_so100_follower_arm_id \
    --robot.cameras="{ 
      front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30},
      wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}
    }" \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/eval_evo1 \
    --dataset.single_task="prompt of your task" \
    --policy.path= /path/of/your/checkpoint/

#Command example
lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=new_follower_arm \
    --robot.cameras="{ 
      front_env: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30},
      side_env: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}
    }" \
    --display_data=true \
    --dataset.repo_id=yinxinyuchen/eval_evo1 \
    --dataset.single_task="Grab the green cube and put the cube in the green box" \
    --policy.path=/home/dell/step_20000/
```
For reference, we also provide a recording that demonstrates how to evaluate Evo1 on SO100/SO101.
If you already have a trained checkpoint, please refer to the following links: \
[YouTube](https://www.youtube.com/watch?v=YzwkllipxXE) \
[bilibili](https://www.bilibili.com/video/BV1cg2QBhErT/?vd_source=17e6e0b7820cb5c4caae006748e7551e)

## 📚 Citation
```bash
@article{lin2025evo,
  title={Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment},
  author={Lin, Tao and Zhong, Yilei and Du, Yuxin and Zhang, Jingjing and Liu, Jiting and Chen, Yinxinyu and Gu, Encheng and Liu, Ziyan and Cai, Hongyi and Zou, Yanwen and others},
  journal={arXiv preprint arXiv:2511.04555},
  year={2025}
}
```


## 📬 Contact

If you encounter any issues or have suggestions,  
please open an issue or start a discussion on GitHub.  
We sincerely welcome your feedback and contributions.

You can also scan the QR code below to connect with me or join chatting group on WeChat:


<p align="center">
<img src="readme_pics/taolin.jpg" width="200" height="300">
<img src="readme_pics/wechat_group.jpg" width="200" height="300">
  
</p>
