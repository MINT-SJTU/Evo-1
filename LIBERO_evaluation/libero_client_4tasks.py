import asyncio
import websockets
import numpy as np
import json
import pathlib
import os
import logging
import math
import imageio
import random

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
os.environ["MUJOCO_GL"] = "osmes"

LIBERO_DUMMY_ACTION = [0.0] * 6 + [0.0]


######################################
class Args():
    horizon = 14
    max_steps = [25,25, 25, 95] 
    SERVER_URL = "ws://0.0.0.0:9000"
    ckpt_name = f"Your_Checkpoint_Name"  
    task_suites = ["libero_spatial","libero_goal", "libero_object", "libero_10"] 
    log_file = f"./log_file/{ckpt_name}.txt"
    num_episodes = 10
    SEED = 42
    
    

args = Args()

########################################
# ========= Logging =========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        
        logging.FileHandler(args.log_file, mode='a'),
        logging.StreamHandler()
    ]

)
log = logging.getLogger(__name__)

# ========= Photos to list[list[list[int]]] =========
def encode_image_array(img_array: np.ndarray):
    return img_array.astype(np.uint8).tolist()

# ========= Quaternion to Axis-Angle =========
def quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

# ========= Observation to JSON-compatible dict =========
def obs_to_json_dict(obs, prompt, resize_size=448):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    dummy_proc = np.zeros((resize_size, resize_size, 3), dtype=np.uint8)

    data = {
        "image": [
            encode_image_array(img),
            encode_image_array(wrist_img),
            encode_image_array(dummy_proc)
        ],
        "state": np.concatenate((
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )).tolist(),
        "prompt": prompt,
        "image_mask": [1, 1, 0],
        "action_mask": [1] * 7 + [0] * 17,
    }
    return data

# ========= Get the environment of LIBERO =========
def get_libero_env(task, resolution=448, seed=args.SEED):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description

# ========= Save the video log =========
def save_video(frames, filename="simulation.mp4", fps=20, save_dir="videos_2"):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    if len(frames) > 0:
        imageio.mimsave(filepath, frames, fps=fps)
        print(f"Video saved: {filepath} ({len(frames)} frames)")
    else:
        log.warning(f"⚠️ No frames to save. File not created: {filepath}")

# ========= Main Function =========
async def run(SERVER_URL: str, max_steps: int = None, num_episodes: int = None, horizon = None, task_suite_name = None):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    print(f"Numbers of tasks: {num_tasks_in_suite}")

    total_success = 0
    total_episodes = 0
    total_steps = 0

    async with websockets.connect(SERVER_URL) as ws:
        log.info(f"===========================Start task suite {task_suite_name}========================")

        for task_id in range(num_tasks_in_suite):

            print(f"task_id{task_id}")
            #if task_id+1 not in [1,5,7,9] :
             #   continue

            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, resolution=448, seed=args.SEED)

            log.info(f"\n========= Start task{task_id+1}: {task_description} =========")

            task_success = 0
            task_episodes = min(num_episodes, len(initial_states))

            for ep in range(task_episodes):
                print(f"\n===== Task {task_id} | Episode {ep+1} =====")

                env.reset()


                obs = env.set_init_state(initial_states[ep])
                t = 0
                while t < 10:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        

                prompt = str(task_description)
                print(prompt)
                episode_done = False
                max_step = 0
                frames = []

                for step in range(max_steps):
                    max_step += 1

                    send_data = obs_to_json_dict(obs, prompt)
                    await ws.send(json.dumps(send_data))
                    print(f"[Step {step}] Send observation")

                    result = await ws.recv()
                    try:
                        action_list = json.loads(result)
                        actions = np.array(action_list)
                        print(f"[Step {step}] recivied actions (shape={actions[0][6]})")
                    except Exception as e:
                        print(f"❌ Action parsing failed: {e}, content: {result}")
                        break

                    
                    for i in range(horizon):
                        action = actions[i].tolist()
                        print(action[:7])
                        if action[6]>0.5:
                            action[6] = -1
                        else:
                            action[6] = 1
                        
                        # action[6] = abs(1.0 - action[6])
                        
                        print(f"gripper action", action[6])
                        try:
                            obs, reward, done, info = env.step(action[:7])
                        except ValueError as ve:
                            print(f"❌ the action is not valid: {ve}")
                            episode_done = False
                            break

                        
                        frame = np.hstack([
                            np.rot90(obs["agentview_image"], 2),
                            np.rot90(obs["robot0_eye_in_hand_image"], 2)
                        ])
                        frames.append(frame)

                        print(f"[Step {step}] reward={reward:.2f}, done={done}")
                        if done:
                            print("Task completed")
                            episode_done = True
                            task_success += 1
                            total_success += 1
                            total_steps += max_step
                            break
                    if episode_done:
                        break

                
                save_video(frames, f"task{task_id+1}_episode{ep+1}.mp4", fps=30, save_dir=f"./video_log_file/{args.ckpt_name}/{task_suite_name}")

                if episode_done:
                    log.info(f"Task {task_id} | Episode {ep+1}: ✅ Success")
                else:
                    log.info(f"Task {task_id} | Episode {ep+1}: ❌ Fail")

                # exit(0)

            log.info(f"========= Task {task_id} Summary: {task_success}/{task_episodes} Successful =========")
            total_episodes += task_episodes

        # ======= Overall Summary =======
        log.info("\n========= Overall Task Summary =========")
        log.info(f"✅ Total Successful Episodes: {total_success}/{total_episodes}")
        if total_episodes > 0:
            log.info(f"📊 Average Steps: {total_steps / total_episodes:.2f}")




if __name__ == "__main__":
    np.random.seed(args.SEED)
    random.seed(args.SEED)
    
    for name, max_steps in zip(args.task_suites, args.max_steps):
        asyncio.run(run(SERVER_URL = args.SERVER_URL,
                        max_steps=max_steps, 
                        num_episodes=args.num_episodes,
                        horizon=args.horizon,
                        task_suite_name=name))
