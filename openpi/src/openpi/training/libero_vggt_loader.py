# openpi/training/data_configs/libero_vggt_loader.py
"""Data loader for LIBERO dataset with precomputed VGGT features."""

import h5py
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Iterator

from openpi.models.model import Observation
from openpi.training.data_config import DataBatch


class LiberoVGGTDataLoader:
    """Loads LIBERO data with precomputed VGGT features."""
    
    def __init__(
        self,
        dataset_dir: str,
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Find all .hdf5 files
        self.hdf5_files = sorted(self.dataset_dir.glob("*.hdf5"))
        print(f"Found {len(self.hdf5_files)} dataset files in {dataset_dir}")
        
        # Load all episodes into memory (or use lazy loading)
        self.episodes = self._load_all_episodes()
        print(f"Loaded {len(self.episodes)} total episodes")
    
    def _load_all_episodes(self):
        """Load all episodes from HDF5 files."""
        episodes = []
        
        for hdf5_file in self.hdf5_files:
            with h5py.File(hdf5_file, 'r') as f:
                demo_keys = [k for k in f['data'].keys() if k.startswith('demo_')]
                
                for demo_key in demo_keys:
                    demo = f['data'][demo_key]
                    
                    # Load episode data
                    episode = {
                        'agentview_rgb': demo['obs']['agentview_rgb'][()],
                        'eye_in_hand_rgb': demo['obs']['eye_in_hand_rgb'][()],
                        'joint_states': demo['obs']['joint_states'][()],
                        'actions': demo['actions'][()],
                        'vggt_scene_features': demo['vggt_scene_features'][()],
                    }
                    
                    episodes.append(episode)
        
        return episodes
    
    def __iter__(self) -> Iterator[DataBatch]:
        """Iterate over batches."""
        indices = np.arange(len(self.episodes))
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, len(indices), self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            
            # Sample random timesteps from each episode
            batch_data = []
            for ep_idx in batch_indices:
                episode = self.episodes[ep_idx]
                
                # Sample random timestep
                t = np.random.randint(0, len(episode['actions']))
                
                batch_data.append({
                    'agentview_rgb': episode['agentview_rgb'][t],
                    'eye_in_hand_rgb': episode['eye_in_hand_rgb'][t],
                    'joint_states': episode['joint_states'][t],
                    'action': episode['actions'][t],
                    'vggt_scene_features': episode['vggt_scene_features'][t],
                })
            
            # Stack into batch
            yield self._collate_batch(batch_data)
    
    def _collate_batch(self, batch_data):
        """Collate list of dicts into batched arrays."""
        batch = {
            'images': {
                'agentview': jnp.array([x['agentview_rgb'] for x in batch_data]),
                'wrist': jnp.array([x['eye_in_hand_rgb'] for x in batch_data]),
            },
            'state': jnp.array([x['joint_states'] for x in batch_data]),
            'actions': jnp.array([x['action'] for x in batch_data]),
            'vggt_scene_features': jnp.array([x['vggt_scene_features'] for x in batch_data]),
        }
        
        # Create observation object
        observation = Observation(
            images=batch['images'],
            image_masks={k: jnp.ones(v.shape[0], dtype=bool) for k, v in batch['images'].items()},
            state=batch['state'],
            vggt_scene_features=batch['vggt_scene_features'],
        )
        
        return DataBatch(
            observation=observation,
            actions=batch['actions'],
        )