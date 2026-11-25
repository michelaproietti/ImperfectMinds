"""Load the THINGS-EEG2 test images (Gifford et al., 2022; OSF 3jk45).

Uses the preprocessed data released with the dataset: one file per participant,
`sub-XX/preprocessed_eeg_test.npy`, with the responses to the 200 test images, 80 repeats each,
already epoched (-200 to 800 ms), downsampled to 100 Hz, baseline-corrected and noise-normalised
(multivariate noise normalisation). Download it with `data/download_things_eeg2.py`.
"""
from pathlib import Path

import mne
import numpy as np


def load_subject(sub_dir, images_dir, n_repeats=None):
    """Test-image responses of one participant.

    sub_dir    : folder `.../sub-XX` containing preprocessed_eeg_test.npy
    images_dir : folder `.../THINGS-EEG2` containing test_images/ and image_metadata.npy
    n_repeats  : average only the first n repeats (default: all 80)

    Returns eeg (200, n_channels, n_times) averaged over repeats, in noise-normalised units (not volts),
    info (mne.Info with montage), times (s), stim_paths (200 image paths, same order as eeg).
    """
    data = np.load(Path(sub_dir) / "preprocessed_eeg_test.npy", allow_pickle=True).item()
    trials = data["preprocessed_eeg_data"]                 # (200 images, 80 repeats, channels, times)
    keep = [i for i, ch in enumerate(data["ch_names"]) if ch != "stim"]   # the file includes the trigger channel
    eeg = trials[:, :n_repeats, keep].mean(1)

    ch_names = [data["ch_names"][i] for i in keep]
    info = mne.create_info(ch_names, sfreq=1 / np.diff(data["times"]).mean(), ch_types="eeg")
    montage = "colin27_1005" if "colin27_1005" in mne.channels.get_builtin_montages() else "standard_1005"
    info.set_montage(montage, match_case=False, on_missing="warn")
    return eeg, info, np.asarray(data["times"]), test_image_paths(images_dir)


def test_image_paths(images_dir):
    """Paths of the 200 test images, in the order of the first axis of the EEG data."""
    images_dir = Path(images_dir)
    meta = np.load(images_dir / "image_metadata.npy", allow_pickle=True).item()
    return [images_dir / "test_images" / c / f for c, f in zip(meta["test_img_concepts"], meta["test_img_files"])]


def check_subject(sub_dir, images_dir):
    """Print what is needed to confirm the assumptions above. Run once on one participant."""
    data = np.load(Path(sub_dir) / "preprocessed_eeg_test.npy", allow_pickle=True).item()
    x, times = data["preprocessed_eeg_data"], np.asarray(data["times"])
    print("keys:", list(data))
    print("shape (images, repeats, channels, times):", x.shape)
    print(f"times: {times[0]:.3f} to {times[-1]:.3f} s, {len(times)} points")
    print("channels:", list(data["ch_names"]), "(the 'stim' trigger channel is dropped when loading)")
    paths = test_image_paths(images_dir)
    missing = [p for p in paths if not p.exists()]
    print("first images:", [f"{p.parent.name}/{p.name}" for p in paths[:3]], "| missing:", len(missing))
