# src/openpi/training/libero_hdf5_dataset.py
"""Direct HDF5 dataset loader for LIBERO with VGGT features."""

import h5py
import numpy as np
from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger(__name__)


class LiberoHDF5Dataset:
    """Load LIBERO dataset directly from HDF5 files with VGGT features."""
    
    # Use the same hardcoded mapping as LoadVGGTFeatures
    LIBERO_SPATIAL_TASKS = {
        0: "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5",
        1: "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5",
        2: "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5",
        3: "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo.hdf5",
        4: "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5",
        5: "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_demo.hdf5",
        6: "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo.hdf5",
        7: "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo.hdf5",
        8: "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_demo.hdf5",
        9: "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5",
    }

    def __init__(
        self,
        dataset_path: str,
        action_horizon: int = 10,
    ):
        """
        Args:
            dataset_path: Path to directory with HDF5 files
            action_horizon: Number of action steps to return
        """
        self.dataset_path = Path(dataset_path)
        self.action_horizon = action_horizon
        
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        # Load all episodes
        self.episodes = []
        self._load_all_episodes()
        
        logger.info(f"Loaded {len(self.episodes)} episodes from {dataset_path}")
    
    def _load_all_episodes(self):
        """Load metadata for all episodes."""
        hdf5_files = sorted(self.dataset_path.glob("*.hdf5"))
        
        if not hdf5_files:
            raise ValueError(f"No HDF5 files found in {self.dataset_path}")
        
        for task_id, hdf5_file in enumerate(hdf5_files):
            try:
                with h5py.File(hdf5_file, 'r') as f:
                    demo_keys = sorted([k for k in f['data'].keys() if k.startswith('demo_')])
                    
                    for demo_key in demo_keys:
                        demo = f['data'][demo_key]
                        
                        # Get task description from file attributes
                        task_desc = demo.attrs.get('language_instruction', 
                                                   hdf5_file.stem.replace('_demo', ''))
                        
                        num_frames = len(demo['actions'])
                        
                        # Store episode metadata
                        self.episodes.append({
                            'file': hdf5_file,
                            'demo_key': demo_key,
                            'task_id': task_id,
                            'task_desc': task_desc,
                            'num_frames': num_frames,
                        })
            except Exception as e:
                logger.warning(f"Error loading {hdf5_file}: {e}")
                continue
        
        logger.info(f"Found {len(hdf5_files)} task files, {len(self.episodes)} total episodes")
    
    def __len__(self):
        """Total number of valid frame indices."""
        return sum(max(1, ep['num_frames'] - self.action_horizon + 1) for ep in self.episodes)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample."""
        # Find which episode and frame this index corresponds to
        cumulative = 0
        for ep_idx, episode in enumerate(self.episodes):
            valid_frames = max(1, episode['num_frames'] - self.action_horizon + 1)
            if idx < cumulative + valid_frames:
                frame_idx = idx - cumulative
                return self._load_sample(episode, frame_idx, ep_idx)
            cumulative += valid_frames
        
        raise IndexError(f"Index {idx} out of range (max: {len(self)})")
    
    def _load_sample(self, episode: dict, frame_idx: int, ep_idx: int) -> dict[str, Any]:
        """Load a single sample from HDF5."""
        with h5py.File(episode['file'], 'r') as f:
            demo = f['data'][episode['demo_key']]
            
            # Load observation at frame_idx
            obs_image = demo['obs']['agentview_rgb'][frame_idx]
            obs_wrist = demo['obs']['eye_in_hand_rgb'][frame_idx]
            obs_state = demo['obs']['joint_states'][frame_idx]
            
            # Load action sequence
            end_idx = min(frame_idx + self.action_horizon, episode['num_frames'])
            actions = demo['actions'][frame_idx:end_idx]
            
            # Pad if needed
            if len(actions) < self.action_horizon:
                padding = np.zeros((self.action_horizon - len(actions), actions.shape[1]))
                actions = np.concatenate([actions, padding], axis=0)
            
            # Load VGGT features
            vggt_features = demo['vggt_scene_features'][frame_idx]
        
        # Return with keys that match what LiberoInputs expects
        return {
            'observation/image': obs_image,              # Changed
            'observation/wrist_image': obs_wrist,        # Changed
            'observation/state': obs_state,              # Changed
            'observation/vggt_scene_features': vggt_features,  # Changed
            'actions': actions,
            'prompt': episode['task_desc'],
            'episode_index': f"task_{episode['task_id']}/episode_{ep_idx}",
            'frame_index': frame_idx,
        }

class LiberoHDF5DatasetNoVGGT(LiberoHDF5Dataset):
    """LIBERO dataset loader WITHOUT VGGT features (for baseline)."""
    
    def _load_sample(self, episode: dict, frame_idx: int, ep_idx: int) -> dict[str, Any]:
        """Load a single sample from HDF5 (no VGGT features)."""
        with h5py.File(episode['file'], 'r') as f:
            demo = f['data'][episode['demo_key']]
            
            # Load observation at frame_idx
            obs_image = demo['obs']['agentview_rgb'][frame_idx]
            obs_wrist = demo['obs']['eye_in_hand_rgb'][frame_idx]
            obs_state = demo['obs']['joint_states'][frame_idx]
            
            # Load action sequence
            end_idx = min(frame_idx + self.action_horizon, episode['num_frames'])
            actions = demo['actions'][frame_idx:end_idx]
            
            # Pad if needed
            if len(actions) < self.action_horizon:
                padding = np.zeros((self.action_horizon - len(actions), actions.shape[1]))
                actions = np.concatenate([actions, padding], axis=0)
        
        # Return with simple keys (no 'observation/' prefix)
        return {
            'image': obs_image,              # Changed from 'observation/image'
            'wrist_image': obs_wrist,        # Changed from 'observation/wrist_image'
            'state': obs_state,              # Changed from 'observation/state'
            'actions': actions,
            'prompt': episode['task_desc'],
            'episode_index': f"task_{episode['task_id']}/episode_{ep_idx}",
            'frame_index': frame_idx,
        }