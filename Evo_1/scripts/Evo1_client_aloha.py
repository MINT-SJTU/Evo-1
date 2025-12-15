import threading
import asyncio
import websockets
import numpy as np
import time
import torch
import json
from PIL import Image
import torchvision.transforms as T

from aloha.env import AlohaRealEnvironment

resize_size = 448
target_state_dim = 24
task_instruction = "Fold the towel."
video_lock = threading.Lock()
shared_frame = {"base": None, "wrist": None}
ENABLE_DISPLAY = True
num_steps = 300
SERVER_URI = "ws://localhost:9000" 

DEFAULT_RESET_POSITION = [0, -0.96, 1.16, 0, -0.3, 0]
PI0_RESET_POSITION     = [0, -1.5, 1.5, 0, 0, 0]


def aloha_env():
    return AlohaRealEnvironment(reset_position = DEFAULT_RESET_POSITION)


async def inference_thread():
    uri = SERVER_URI
    async with websockets.connect(uri, max_size=10_000_000) as ws:
    # if 1:
        print("connected to server")

        aloha = aloha_env()
        aloha.reset()

        for step in range(num_steps):
            print(f"=== Step {step} ===")

            obs_data = aloha.get_observation()
            raw_images = obs_data["images"]
            raw_state = obs_data["state"]

            print(f"obs: {raw_state}")

            cam_keys = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
            for key in cam_keys:
                if key not in raw_images:
                    raise ValueError(f"Camera {key} not found in observation images.")
                
            cam_high = raw_images["cam_high"]
            cam_left_wrist = raw_images["cam_left_wrist"]
            cam_right_wrist = raw_images["cam_right_wrist"]


            pad_dim = target_state_dim - len(raw_state)
            state = np.pad(raw_state, (0, pad_dim), constant_values=0)
            action_mask = [[1]*len(raw_state) + [0]*pad_dim]

            # build observation
            obs = {
                "image": [cam_high.tolist(), cam_left_wrist.tolist(), cam_right_wrist.tolist()],
                "image_mask": [int(i) for i in [1, 1, 1]],
                "state": state.astype(float).tolist(),
                "action_mask": [[int(i) for i in action_mask[0]]],
                "prompt": task_instruction
            }

            try:
                await ws.send(json.dumps(obs))
                result = await ws.recv()
                action_chunk = torch.tensor(json.loads(result))
                
                
            except Exception as e:
                print(f"❌ Inference Error: {e}")
                await asyncio.sleep(0.5)
                continue

            print(f"[Step {step}] gets the action: {action_chunk.shape}")

            for i, act in enumerate(action_chunk[:25]):
                
                action = act[:14]
                # joint = np.concatenate(act[:6], act[7:13]).tolist()
                joint = torch.cat([act[:6], act[7:13]]).tolist()
                grip = np.array([act[6].item(), act[13].item()])

                aloha.apply_action({"actions": action.tolist()})
                print(f"[Action {i}]: {action.tolist()}")
                time.sleep(1/100)


if __name__ == "__main__":

    try:
        asyncio.run(inference_thread())
    except KeyboardInterrupt:
        print("🛑 User interrupted, exiting program")
