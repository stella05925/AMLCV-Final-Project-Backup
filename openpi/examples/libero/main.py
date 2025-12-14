import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

import torch
import torch.nn.functional as F
from vggt.models.vggt import VGGT

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 1  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero+/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

def preprocess_image_for_vggt(image_array, target_size=224):
    """
    Preprocesses a single image (H, W, 3) for the VGGT model.
    """
    # Normalize to [0, 1]
    image = torch.from_numpy(image_array).float() / 255.0
    
    # Transpose to CHW: [H, W, 3] -> [3, H, W]
    image = image.permute(2, 0, 1)
    
    # Add batch dim for interpolation: [1, 3, H, W]
    image = image.unsqueeze(0)

    # Resize to target_size (224)
    if image.shape[2] != target_size or image.shape[3] != target_size:
        image = F.interpolate(
            image,
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False,
            antialias=True
        )
    # Remove batch dim: [3, H, W]
    return image.squeeze(0)

def pool_vggt_features(scene_tokens, pose_enc, depth_map, point_map):
    """
    Pools VGGT outputs into the single 2065-dim vector.
    Structure: [Tokens_Mean(1024), Tokens_Std(1024), Pose(9), Depth(2), Points(6)]
    Total: 2065 dimensions.
    """
    # 1. Scene Tokens: Mean AND Std (To get 2048 dims)
    tokens_mean = scene_tokens.mean(dim=[1, 2])  # [B, 1024]
    tokens_std = scene_tokens.std(dim=[1, 2])    # [B, 1024]
    tokens_pooled = torch.cat([tokens_mean, tokens_std], dim=-1) # [B, 2048]
    
    # 2. Camera: Mean [B, 9]
    pose_pooled = pose_enc.mean(dim=1)
    
    # 3. Depth: Mean + Std [B, 2]
    depth_mean = depth_map.mean(dim=[1, 2, 3])
    depth_std = depth_map.std(dim=[1, 2, 3])
    depth_features = torch.cat([depth_mean, depth_std], dim=-1)
    
    # 4. Points: Mean + Std [B, 6]
    point_mean = point_map.mean(dim=[1, 2, 3])
    point_std = point_map.std(dim=[1, 2, 3])
    point_features = torch.cat([point_mean, point_std], dim=-1)
    
    # Concatenate: 2048 + 9 + 2 + 6 = 2065
    all_features = torch.cat([
        tokens_pooled,
        pose_pooled,
        depth_features,
        point_features,
    ], dim=-1)
    
    return all_features.cpu().numpy()

def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # --- ADD THIS BLOCK: Initialize VGGT ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Initializing VGGT model on {device}...")
    
    # Load model
    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    vggt_model.eval()
    
    # Use bfloat16 for speed (matching your precompute script)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    logging.info("VGGT Model Loaded.")
    # ---------------------------------------

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img_raw = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_raw = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # --- ADD THIS BLOCK: Compute VGGT Features ---
                    # 1. Preprocess images
                    # Note: We use the raw 'img' and 'wrist_img' (numpy arrays), not the resized ones used for the policy visual encoder
                    # If your policy resize is 224, you can reuse, but VGGT specific preprocessing is safer
                    img_tensor = preprocess_image_for_vggt(img_raw).to(device)
                    wrist_tensor = preprocess_image_for_vggt(wrist_raw).to(device)

                    # 2. Stack input: [Batch=1, Seq=2, Channels=3, H=224, W=224]
                    # Sequence order: AgentView first, then Wrist (matching your precompute script)
                    vggt_input = torch.stack([img_tensor, wrist_tensor], dim=0).unsqueeze(0)
                    
                    logging.info("Computing VGGT features...")

                    # 3. Run Inference
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(dtype=dtype):
                            aggregated_tokens_list, ps_idx = vggt_model.aggregator(vggt_input)
                            
                            # Extract raw outputs
                            scene_tokens = aggregated_tokens_list[-1]
                            pose_enc = vggt_model.camera_head(aggregated_tokens_list)[-1]
                            depth_map, _ = vggt_model.depth_head(aggregated_tokens_list, vggt_input, ps_idx)
                            point_map, _ = vggt_model.point_head(aggregated_tokens_list, vggt_input, ps_idx)
                            
                            # Pool into single vector [1, 2065]
                            features_array = pool_vggt_features(scene_tokens, pose_enc, depth_map, point_map)
                    
                    current_vggt_features = features_array[0] # Shape (2065,)
                    logging.info("VGGT features computed, shape: " + str(current_vggt_features.shape))
                    
                    # ---------------------------------------------

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": obs["robot0_joint_pos"],
                            "observation/vggt_scene_features": current_vggt_features,
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    logging.info("Before executing action")

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
