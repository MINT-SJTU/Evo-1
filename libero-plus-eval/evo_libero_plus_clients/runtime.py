import asyncio
import json
import logging
import math
import random
from pathlib import Path
from typing import List, Optional

import imageio
import numpy as np
import websockets
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from .config import ClientConfig


LIBERO_DUMMY_ACTION = [0.0] * 7


class EvaluationClient:
    def __init__(self, config: ClientConfig):
        self.config = config
        self.log = self._create_logger()

    def _create_logger(self) -> logging.Logger:
        self.config.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.config.video_log_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(f"libero_plus.{self.config.category}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.handlers.clear()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        file_handler = logging.FileHandler(
            self.config.log_file, mode="a", encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        return logger

    def filter_task_ids(self, task_suite_name: str, json_path: Optional[Path] = None):
        if json_path is None:
            json_path = (
                Path(get_libero_path("benchmark_root"))
                / "benchmark"
                / "task_classification.json"
            )

        with json_path.open("r", encoding="utf-8") as file:
            task_json = json.load(file)

        categories = self.config.category.split(",")
        return [
            task["id"] - 1
            for task in task_json[task_suite_name]
            if any(
                category in task["category"].lower()
                for category in categories
            )
        ]

    @staticmethod
    def encode_image_array(img_array: np.ndarray):
        return img_array.astype(np.uint8).tolist()

    @staticmethod
    def quat2axisangle(quat):
        quat = quat.copy()
        quat[3] = np.clip(quat[3], -1.0, 1.0)
        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(den, 0.0):
            return np.zeros(3)
        return (quat[:3] * 2.0 * math.acos(quat[3])) / den

    def obs_to_json_dict(self, obs, prompt, resize_size=448):
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(
            obs["robot0_eye_in_hand_image"][::-1, ::-1]
        )
        dummy_proc = np.zeros((resize_size, resize_size, 3), dtype=np.uint8)

        return {
            "image": [
                self.encode_image_array(img),
                self.encode_image_array(wrist_img),
                self.encode_image_array(dummy_proc),
            ],
            "state": np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    self.quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            ).tolist(),
            "prompt": prompt,
            "image_mask": [1, 1, 0],
            "action_mask": [1] * 7 + [0] * 17,
        }

    def get_libero_env(self, task, resolution=448):
        task_bddl_file = (
            Path(get_libero_path("bddl_files"))
            / task.problem_folder
            / task.bddl_file
        )
        env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=resolution,
            camera_widths=resolution,
        )
        env.seed(self.config.seed)
        return env, task.language

    def save_video(self, frames: List[np.ndarray], filename: str, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / filename
        if frames:
            imageio.mimsave(filepath, frames, fps=30)
            print(f"Saved video: {filepath} ({len(frames)} frames)")
        else:
            self.log.warning("No frames; video was not generated: %s", filepath)

    async def run_suite(self, task_suite_name: str, max_steps: int) -> None:
        task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
        filter_ids = self.filter_task_ids(task_suite_name)
        print(f"Numbers of tasks: {len(filter_ids)}")

        total_success = 0
        total_episodes = 0
        total_steps = 0

        async with websockets.connect(self.config.server_url) as ws:
            self.log.info("Start task suite %s", task_suite_name)

            for index, task_id in enumerate(filter_ids):
                print(f"task_id{task_id}")
                task = task_suite.get_task(task_id)
                initial_states = task_suite.get_task_init_states(task_id)
                env, task_description = self.get_libero_env(task)

                try:
                    task_episodes = min(
                        self.config.num_episodes, len(initial_states)
                    )
                    for episode in range(task_episodes):
                        env.reset()
                        obs = env.set_init_state(initial_states[episode])
                        for _ in range(10):
                            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

                        episode_done = False
                        executed_steps = 0
                        frames = []

                        for _ in range(max_steps // self.config.horizon):
                            executed_steps += self.config.horizon
                            await ws.send(
                                json.dumps(
                                    self.obs_to_json_dict(
                                        obs, str(task_description)
                                    )
                                )
                            )

                            result = await ws.recv()
                            try:
                                actions = np.array(json.loads(result))
                            except Exception as exc:
                                print(
                                    f"Failed to parse actions: {exc}; "
                                    f"content: {result}"
                                )
                                break

                            for action_index in range(self.config.horizon):
                                action = actions[action_index].tolist()
                                action[6] = -1 if action[6] > 0.5 else 1
                                try:
                                    obs, _, done, _ = env.step(action[:7])
                                except ValueError as exc:
                                    print(f"Environment action failed: {exc}")
                                    episode_done = False
                                    break

                                frames.append(
                                    np.hstack(
                                        [
                                            np.rot90(obs["agentview_image"], 2),
                                            np.rot90(
                                                obs["robot0_eye_in_hand_image"], 2
                                            ),
                                        ]
                                    )
                                )
                                if done:
                                    print("Task completed.")
                                    episode_done = True
                                    total_success += 1
                                    total_steps += executed_steps
                                    break
                            if episode_done:
                                break

                        self.save_video(
                            frames,
                            f"task{task_id}_episode{episode + 1}.mp4",
                            self.config.video_log_dir
                            / task_suite_name
                            / self.config.category,
                        )
                        status = "Success" if episode_done else "Fail"
                        self.log.info(
                            "Task %s | %s task | Episode %s: %s",
                            task_id,
                            index,
                            episode + 1,
                            status,
                        )
                    total_episodes += task_episodes
                finally:
                    env.close()

        self.log.info("All tasks summary")
        success_rate = (
            total_success / total_episodes * 100 if total_episodes else 0.0
        )
        self.log.info(
            "Successful episodes: %s/%s | Success rate: %.2f%%",
            total_success,
            total_episodes,
            success_rate,
        )
        if total_episodes:
            self.log.info(
                "Average steps: %.2f", total_steps / total_episodes
            )

    def run(self) -> None:
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
        self.log.info(
            "category=%s suites=%s server=%s output=%s",
            self.config.category,
            ",".join(self.config.task_suites),
            self.config.server_url,
            self.config.output_dir,
        )
        for suite in self.config.task_suites:
            asyncio.run(
                self.run_suite(suite, self.config.max_steps_for(suite))
            )


def run_evaluation(config: ClientConfig) -> None:
    EvaluationClient(config).run()
