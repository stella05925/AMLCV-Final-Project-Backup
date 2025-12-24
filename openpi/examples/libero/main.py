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

# --- FIX: Monkeypatch scaled_dot_product_attention for PyTorch 1.11 ---
if not hasattr(F, 'scaled_dot_product_attention'):
    def manual_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            if attn_mask is not None:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value

    torch.nn.functional.scaled_dot_product_attention = manual_sdpa
    print("PATCH APPLIED: Added manual scaled_dot_product_attention for PyTorch 1.x")
# ----------------------------------------------------------------------

from vggt.models.vggt import VGGT

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

@dataclasses.dataclass
class Args:
    host: str = "openpi_server"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 1
    video_out_path: str = "data/libero+/videos"
    seed: int = 7

# --- MATCHING TRAINING PREPROCESSING ---
def preprocess_image_for_vggt(image_array, target_size=224):
    """
    Exact match to training preprocessing.
    """
    # Normalize to [0, 1]
    image = torch.from_numpy(image_array).float() / 255.0
    
    # Transpose to CHW format [H, W, 3] -> [3, H, W]
    image = image.permute(2, 0, 1)
    
    # Resize to target_size using bilinear interpolation
    if image.shape[1] != target_size or image.shape[2] != target_size:
        image = F.interpolate(
            image.unsqueeze(0),
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False,
            antialias=True
        ).squeeze(0)
    
    return image

# --- MATCHING TRAINING POOLING (Output Dim: 1041) ---
def pool_vggt_features(scene_tokens, pose_enc, depth_map, point_map):
    """
    Exact match to training pooling logic.
    Returns vector of size 1041 (1024 + 9 + 2 + 6).
    """
    # 1. Pool scene tokens [B, 1024]
    tokens_pooled = scene_tokens.mean(dim=[1, 2])
    
    # 2. Pool camera parameters [B, 9]
    pose_pooled = pose_enc.mean(dim=1)
    
    # 3. Pool depth statistics [B, 2]
    depth_mean = depth_map.mean(dim=[1, 2, 3])
    depth_std = depth_map.std(dim=[1, 2, 3])
    depth_features = torch.cat([depth_mean, depth_std], dim=-1)
    
    # 4. Pool point map statistics [B, 6]
    point_mean = point_map.mean(dim=[1, 2, 3])
    point_std = point_map.std(dim=[1, 2, 3])
    point_features = torch.cat([point_mean, point_std], dim=-1)
    
    # Concatenate all features
    all_features = torch.cat([
        tokens_pooled,     # 1024
        pose_pooled,       # 9
        depth_features,    # 2
        point_features,    # 6
    ], dim=-1)             # Total: 1041
    
    return all_features.cpu().numpy()

def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial": max_steps = 50
    elif args.task_suite_name == "libero_object": max_steps = 280
    elif args.task_suite_name == "libero_goal": max_steps = 300
    elif args.task_suite_name == "libero_10": max_steps = 520
    elif args.task_suite_name == "libero_90": max_steps = 400
    else: raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # --- SAFE VGGT LOADING (PyTorch 1.11 Compatible) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # FORCE float16 because BFloat16 crashes on PyTorch 1.11 upsampling
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    logging.info(f"Loading VGGT to CPU first...")
    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B")
    
    logging.info(f"Casting to {dtype} and moving to {device}...")
    vggt_model = vggt_model.to(dtype=dtype, device=device)
    vggt_model.eval()
    logging.info("VGGT Model Loaded successfully.")
    # ---------------------------------------------------

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes, total_successes = 0, 0
    total_steps_from_success = 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        steps_from_success = 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img_raw = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_raw = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img_raw, args.resize_size, args.resize_size))
                    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_raw, args.resize_size, args.resize_size))
                    replay_images.append(img)

                    # --- INFERENCE ON REPLAN ONLY ---
                    if not action_plan:
                        # 1. Preprocess exactly like training
                        img_tensor = preprocess_image_for_vggt(img_raw).to(device)
                        wrist_tensor = preprocess_image_for_vggt(wrist_raw).to(device)
                        
                        vggt_input = torch.stack([img_tensor, wrist_tensor], dim=0).unsqueeze(0)
                        
                        # 2. Run Inference
                        with torch.no_grad():
                            with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
                                aggregated_tokens_list, ps_idx = vggt_model.aggregator(vggt_input)
                                
                                scene_tokens = aggregated_tokens_list[-1]
                                pose_enc = vggt_model.camera_head(aggregated_tokens_list)[-1]
                                depth_map, _ = vggt_model.depth_head(aggregated_tokens_list, vggt_input, ps_idx)
                                point_map, _ = vggt_model.point_head(aggregated_tokens_list, vggt_input, ps_idx)
                                
                                features_array = pool_vggt_features(scene_tokens, pose_enc, depth_map, point_map)
                        
                        current_vggt_features = features_array[0]

                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": obs["robot0_joint_pos"],
                            "observation/vggt_scene_features": current_vggt_features,
                            "prompt": str(task_description),
                        }
                        
                        action_chunk = client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps
                        action_plan.extend(action_chunk[: args.replan_steps])
                    # ---------------------------------
                    
                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    t += 1
                    if done:
                        task_successes += 1
                        steps_from_success +=t 
                        break

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    import traceback
                    traceback.print_exc()
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
            logging.info(f"# Episodes completed so far: {task_episodes}")
            if done:
                logging.info(f"Steps to success in this episode: {t}")
        # Log final results
        logging.info(f"# Successes for this task: {task_successes} ({task_successes / task_episodes * 100:6f}%)")
        if task_successes > 0:
                logging.info(f"Average steps to success (successful episodes only): {steps_from_success / task_successes}")
        total_steps_from_success += steps_from_success
        total_successes += task_successes

    logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:6f}%)")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Current total success rate: {float(task_successes) / float(total_episodes)}")
    if total_successes > 0:
            logging.info(f"Average steps to success (successful episodes only): {total_steps_from_success / total_successes}")


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)