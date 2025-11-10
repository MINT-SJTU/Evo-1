import threading
import asyncio
import websockets
import numpy as np
import cv2
import time
import torch
import json
from PIL import Image
from xarm.wrapper import XArmAPI
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


resize_size = 448
target_state_dim = 24
task_instruction = "Pick up the can and move it into the box."
video_lock = threading.Lock()
shared_frame = {"base": None, "wrist": None}
ENABLE_DISPLAY = True
num_steps = 300  


def convert_to_uint8(img):
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img

def resize_with_pad(image, method=Image.BILINEAR):
    img_pil = Image.fromarray(image)
    
    basic_transform = T.Compose([
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor()
    ])
    
    img_tensor = basic_transform(img_pil)
    img_numpy = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    return img_numpy


arm = XArmAPI('192.168.1.222')
arm.clean_error()
arm.motion_enable(True)
arm.set_gripper_enable(True)
arm.set_mode(6)
arm.set_state(0)
init_qpos_deg = [4.3, 15.5, -9.4, 182.6, 100.4, 0.3]
arm.set_servo_angle(angle=np.radians(init_qpos_deg), speed=3, is_radian=True)


def camera_thread():
    base_cam = cv2.VideoCapture("/dev/video10")
    wrist_cam = cv2.VideoCapture("/dev/video4")

    if ENABLE_DISPLAY:
        cv2.namedWindow("Base", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Wrist", cv2.WINDOW_NORMAL)

    while True:
        ret1, base = base_cam.read()
        ret2, wrist = wrist_cam.read()
        if not ret1 or not ret2:
            continue
        base = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
        wrist = cv2.cvtColor(wrist, cv2.COLOR_BGR2RGB)
        with video_lock:
            shared_frame["base"] = base
            shared_frame["wrist"] = wrist

        if ENABLE_DISPLAY:
            cv2.imshow("Base", cv2.cvtColor(base, cv2.COLOR_RGB2BGR))
            cv2.imshow("Wrist", cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        time.sleep(0.01)

    base_cam.release()
    wrist_cam.release()
    if ENABLE_DISPLAY:
        cv2.destroyAllWindows()


async def inference_thread():
    uri = "ws://localhost:8000"
    async with websockets.connect(uri, max_size=10_000_000) as ws:
        print("connected to server")

        for step in range(num_steps):
            with video_lock:
                base = shared_frame["base"]
                wrist = shared_frame["wrist"]

            if base is None or wrist is None:
                await asyncio.sleep(0.01)
                continue

            
            _, joint_rad = arm.get_servo_angle(is_radian=True)
            _, grip_pos = arm.get_gripper_position()
            raw_state = np.append(np.rad2deg(joint_rad[:-1]), grip_pos)
            pad_dim = target_state_dim - len(raw_state)
            state = np.pad(raw_state, (0, pad_dim), constant_values=0)
            action_mask = [[1]*len(raw_state) + [0]*pad_dim]

            
            base_proc = convert_to_uint8(resize_with_pad(base))
            wrist_proc = convert_to_uint8(resize_with_pad(wrist))
            dummy_proc = np.zeros((resize_size, resize_size, 3), dtype=np.uint8)

            # build observation
            obs = {
                "image": [base_proc.tolist(), wrist_proc.tolist(), dummy_proc.tolist()],
                "image_mask": [int(i) for i in [1, 1, 0]],
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
                joint = act[:6].tolist()
                grip = act[6].item()
                arm.set_servo_angle(angle=joint, speed=20, is_radian=False)
                arm.set_gripper_position(pos=grip, wait=False)
                print(f"[Joint {i}]: {joint}, {grip}")
                time.sleep(1/100)


if __name__ == "__main__":
    t_cam = threading.Thread(target=camera_thread, daemon=True)
    t_cam.start()

    try:
        asyncio.run(inference_thread())
    except KeyboardInterrupt:
        print("🛑 User interrupted, exiting program")
