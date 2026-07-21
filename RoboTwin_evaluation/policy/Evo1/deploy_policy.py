import asyncio
import json
import cv2
import numpy as np
import websockets


def encode_image_array(img_rgb):
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return bgr.astype(np.uint8).tolist()


def encode_obs(observation):
    head_img = observation["observation"]["head_camera"]["rgb"]
    left_img = observation["observation"]["left_camera"]["rgb"]
    right_img = observation["observation"]["right_camera"]["rgb"]
    state = observation["joint_action"]["vector"].tolist()
    return head_img, left_img, right_img, state


def _gaussian_smooth(actions, kernel_size=9):
    """Gaussian-weighted smoothing along the time axis (center weighted highest)."""
    T, D = actions.shape
    half = kernel_size // 2
    sigma = half / 2.0
    kernel = np.exp(-0.5 * (np.arange(kernel_size) - half) ** 2 / (sigma ** 2))
    kernel /= kernel.sum()

    smoothed = np.copy(actions)
    for t in range(T):
        start = max(0, t - half)
        end = min(T, t + half + 1)
        k_start = half - (t - start)
        k_end = half + (end - t)
        w = kernel[k_start:k_end]
        w = w / w.sum()  # renormalize at boundaries
        smoothed[t] = (actions[start:end] * w[:, None]).sum(axis=0)
    return smoothed


def smooth_actions(actions, kernel_size=9, smooth_type="gaussian"):
    """Smooth an action chunk (T, D) along time. Gaussian is the validated recipe."""
    T, D = actions.shape
    if T <= kernel_size:
        return actions
    if smooth_type == "gaussian":
        return _gaussian_smooth(actions, kernel_size)
    # uniform moving-average fallback
    smoothed = np.copy(actions)
    half = kernel_size // 2
    for t in range(T):
        start = max(0, t - half)
        end = min(T, t + half + 1)
        smoothed[t] = actions[start:end].mean(axis=0)
    return smoothed


class Evo1Proxy:
    def __init__(self, server_url, horizon, task_name):
        self.server_url = server_url
        self.horizon = horizon
        self.task_name = task_name
        self.dataset_key = f"robotwin_{task_name}"
        self.arm_key = "aloha_joint"
        self.ws = None
        self.loop = asyncio.new_event_loop()

    def connect(self):
        async def _connect():
            self.ws = await websockets.connect(
                self.server_url,
                ping_interval=None,
                ping_timeout=None,
                max_size=100_000_000,
            )
        self.loop.run_until_complete(_connect())

    def infer(self, head_img, left_img, right_img, state, prompt):
        payload = {
            "image": [
                encode_image_array(head_img),
                encode_image_array(left_img),
                encode_image_array(right_img),
            ],
            "state": state,
            "prompt": prompt,
            "image_mask": [1, 1, 1],
            "action_mask": [1] * 14 + [0] * 10,
            "dataset_key": self.dataset_key,
            "arm_key": self.arm_key,
        }

        async def _infer():
            await self.ws.send(json.dumps(payload))
            result = await self.ws.recv()
            return json.loads(result)

        return self.loop.run_until_complete(_infer())

    def close(self):
        if self.ws:
            self.loop.run_until_complete(self.ws.close())


def get_model(usr_args):
    server_url = usr_args.get("server_url", "ws://0.0.0.0:9000")
    horizon = int(usr_args.get("horizon", 37))
    task_name = usr_args["task_name"]

    proxy = Evo1Proxy(server_url, horizon, task_name)
    proxy.connect()
    return proxy


def eval(TASK_ENV, model, observation):
    head_img, left_img, right_img, state = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()

    actions_raw = model.infer(head_img, left_img, right_img, state, str(instruction))
    actions = np.array(actions_raw)  # (50, 24) denormalized

    # Smoothing hardcoded: gaussian kernel=9 (the validated RoboTwin recipe).
    # Executing raw chunks jitters the arm and roughly halves success rate.
    actions = smooth_actions(actions, kernel_size=9, smooth_type="gaussian")

    for i in range(model.horizon):
        action_14d = actions[i][:14]  # 7 left + 7 right joints
        TASK_ENV.take_action(action_14d)
        observation = TASK_ENV.get_obs()

    return observation


def reset_model(model):
    pass
