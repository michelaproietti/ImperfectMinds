"""Compare a pretrained vision model with EEG using Representational Similarity Analysis (RSA).

Pipeline
    images -> model features (last layer) -> model RDM
    EEG epochs -> one RDM per time point -> RSA with the model RDM
    RSA per time point  -> timecourse
    RSA per channel neighbourhood and time point -> animated topomap

Data conventions
    eeg   : (n_images, n_channels, n_times) array, already averaged over repetitions
    info  : mne.Info with channel positions (needed for topomaps and neighbourhoods)
    times : (n_times,) array in seconds
"""
import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from scipy.stats import rankdata


# ----------------------------------------------------------------------------
# 1. Model features
# ----------------------------------------------------------------------------
def load_images(paths, preprocess):
    """Open image files and apply the model's own preprocessing -> (N, 3, H, W) tensor."""
    import torch
    from PIL import Image

    return torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths])


def extract_features(images, model, layer="avgpool", batch_size=32, device="cpu"):
    """Activations of `layer` for every image -> (N, D) array.

    For torchvision ResNets "avgpool" is the last representation before the classifier
    (2048-d for ResNet-50). Use layer="fc" for the 1000 class scores instead.
    Run `[n for n, _ in model.named_modules()]` to see the available layers.
    """
    import torch

    out = []
    layer_module = dict(model.named_modules())[layer]
    handle = layer_module.register_forward_hook(lambda m, i, o: out.append(o.flatten(1).cpu()))
    model.eval().to(device)
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            model(images[i : i + batch_size].to(device))
    handle.remove()
    return torch.cat(out).numpy()


# ----------------------------------------------------------------------------
# 2. RDMs and RSA
# ----------------------------------------------------------------------------
def rdm(patterns):
    """Representational dissimilarity matrix: 1 - Pearson r between all pairs of rows."""
    return 1 - np.corrcoef(patterns)


def upper(matrix):
    """Upper triangle (without the diagonal) as a vector."""
    return matrix[np.triu_indices(len(matrix), k=1)]


def eeg_rdms(eeg):
    """One RDM per time point -> (n_times, n_pairs), each row an upper triangle."""
    return np.stack([upper(rdm(eeg[:, :, t])) for t in range(eeg.shape[2])])


def spearman(vector, rows):
    """Spearman correlation between `vector` (P,) and every row of `rows` (T, P)."""
    a = rankdata(vector)
    b = rankdata(rows, axis=1)
    a = (a - a.mean()) / a.std()
    b = (b - b.mean(1, keepdims=True)) / b.std(1, keepdims=True)
    return b @ a / len(a)


def rsa_timecourse(eeg, model_rdm):
    """RSA at every time point, using all channels -> (n_times,)."""
    return spearman(upper(model_rdm), eeg_rdms(eeg))


def rsa_searchlight(eeg, info, model_rdm, n_neighbors=8):
    """RSA at every time point for every channel and its nearest neighbours -> (n_channels, n_times)."""
    pos = np.array([ch["loc"][:3] for ch in info["chs"]])
    dist = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    neighbours = np.argsort(dist, axis=1)[:, :n_neighbors]  # includes the channel itself
    return np.stack([rsa_timecourse(eeg[:, nb, :], model_rdm) for nb in neighbours])


# ----------------------------------------------------------------------------
# 3. Figures
# ----------------------------------------------------------------------------
def plot_erp(eeg, info, times, channel, ax=None, scale=1e6, unit="µV"):
    """Average EEG response (over images) of one channel, with a +-1 SEM band.

    By default eeg is in volts and is shown in microvolts. For data that is not in volts
    (e.g. noise-normalised THINGS-EEG2), use scale=1, unit="a.u.".
    """
    ax = ax or plt.subplots(figsize=(6, 3))[1]
    i = info["ch_names"].index(channel)
    x = eeg[:, i, :] * scale
    mean, sem = x.mean(0), x.std(0) / np.sqrt(len(x))
    ax.plot(times * 1000, mean, color="#272741")
    ax.fill_between(times * 1000, mean - sem, mean + sem, color="#14B8A6", alpha=0.3)
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set(xlabel="Time from image onset (ms)", ylabel=f"Amplitude ({unit})", title=f"Channel {channel}")
    return ax


def plot_rdms(model_rdm, brain_rdm, brain_title="EEG RDM", axes=None):
    """Show a model RDM and a brain RDM side by side."""
    axes = axes if axes is not None else plt.subplots(1, 2, figsize=(8, 4))[1]
    for ax, m, title in zip(axes, (model_rdm, brain_rdm), ("Model RDM", brain_title)):
        im = ax.imshow(m, cmap="viridis")
        ax.set(title=title, xlabel="Image", ylabel="Image")
        plt.colorbar(im, ax=ax, label="1 - correlation", fraction=0.046)
    return axes


def plot_rsa_timecourse(times, rsa, ax=None):
    ax = ax or plt.subplots(figsize=(6, 3))[1]
    ax.plot(times * 1000, rsa, color="#DB2777")
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set(xlabel="Time from image onset (ms)", ylabel="RSA (Spearman rho)", title="Model-brain alignment")
    return ax


def animate_topomap(rsa_map, info, times, path, vmax=None, fps=10, step=1, figsize=(4, 4.4)):
    """Animate the searchlight RSA map over time. Saves .mp4 or .gif depending on `path`."""
    vmax = vmax or np.abs(rsa_map).max()
    frames = range(0, len(times), step)
    fig, ax = plt.subplots(figsize=figsize)
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(-vmax, vmax), cmap="RdBu_r")
    fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.08, label="RSA (Spearman rho)")

    def draw(t):
        ax.clear()
        mne.viz.plot_topomap(rsa_map[:, t], info, axes=ax, cmap="RdBu_r", vlim=(-vmax, vmax),
                             contours=0, sensors=True, show=False)
        ax.set_title(f"{times[t] * 1000:.0f} ms")

    writer = PillowWriter(fps=fps) if str(path).endswith(".gif") else FFMpegWriter(fps=fps, extra_args=["-pix_fmt", "yuv420p"])
    FuncAnimation(fig, draw, frames=frames).save(path, writer=writer, dpi=150)
    plt.close(fig)