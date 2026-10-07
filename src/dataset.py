"""Dataset loader for four-channel far-field intensity/phase pairs."""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class MetaDataset(Dataset):
    """Load MATLAB v7.3 HDF5 data and expose tensors as ``[C, H, W]``."""

    def __init__(self, mat_path, intensity_key="intensity_dataset", phase_key="phase_dataset"):
        mat_path = Path(mat_path)
        if not mat_path.is_file():
            raise FileNotFoundError(f"Dataset file does not exist: {mat_path}")

        with h5py.File(mat_path, "r") as file:
            if intensity_key not in file or phase_key not in file:
                raise KeyError(f"Expected '{intensity_key}' and '{phase_key}', found: {list(file.keys())}")
            raw_intensity = np.asarray(file[intensity_key], dtype=np.float32)
            raw_phase = np.asarray(file[phase_key], dtype=np.float32)

        if raw_intensity.shape != raw_phase.shape or raw_intensity.ndim != 4:
            raise ValueError(
                "Intensity and phase must have identical four-dimensional shapes; "
                f"got {raw_intensity.shape} and {raw_phase.shape}."
            )

        channel_axes = [axis for axis, size in enumerate(raw_intensity.shape) if size == 4]
        if len(channel_axes) != 1:
            raise ValueError(f"Could not uniquely identify four-channel axis in {raw_intensity.shape}.")
        channel_axis = channel_axes[0]
        remaining_axes = [axis for axis in range(4) if axis != channel_axis]
        spatial_axes = None
        for first in range(len(remaining_axes)):
            for second in range(first + 1, len(remaining_axes)):
                candidate = [remaining_axes[first], remaining_axes[second]]
                if raw_intensity.shape[candidate[0]] == raw_intensity.shape[candidate[1]]:
                    spatial_axes = candidate
                    break
            if spatial_axes is not None:
                break
        if spatial_axes is None:
            raise ValueError(f"Could not identify square spatial axes in {raw_intensity.shape}.")
        sample_axis = next(axis for axis in remaining_axes if axis not in spatial_axes)

        order = [sample_axis, channel_axis, *spatial_axes]
        intensity = raw_intensity.transpose(order).copy()
        phase = raw_phase.transpose(order).copy()
        maxima = intensity.max(axis=(1, 2, 3), keepdims=True)
        intensity /= np.maximum(maxima, 1e-8)

        self.intensity = intensity
        self.phase = np.angle(np.exp(1j * phase)).astype(np.float32)
        self.path = mat_path
        print(f"Loaded {len(self)} samples from {mat_path.name}: {self.intensity.shape}")

    def __len__(self):
        return self.intensity.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.intensity[index]), torch.from_numpy(self.phase[index])
