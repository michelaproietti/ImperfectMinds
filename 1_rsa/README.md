# 1_rsa: comparing a vision model with EEG using RSA

Code for part 1 of *Comparing minds* in the Imperfect Minds newsletter: [link to the post].

It asks a simple question: which images does a brain treat as alike, and does a pretrained vision model treat the same images as alike? We build a table of pairwise differences between images for the model and for the EEG (a representational dissimilarity matrix, RDM), and compare the tables with representational similarity analysis (RSA, [Kriegeskorte et al., 2008](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/neuro.06.004.2008/pdf)).

## The six steps

Both notebooks follow the same outline:

1. Generate / load the EEG data
2. Create the EEG RDMs
3. Load the pretrained model
4. Create the model RDMs
5. RSA: compare EEG and model RDMs
6. Topomaps: where on the head?

## Set up the environment

The analysis runs fine on a CPU (it only passes 200 images through the model), so a GPU is not needed.

### Option A: pyenv (lightest)

Python 3.11 works with current PyTorch and MNE. Follow these steps:
1. Install brew.
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
On MacOS:
```
echo >> ~/.zprofile
echo 'eval "$(/opt/homebrew/bin/brew shellenv zsh)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv zsh)"
```
On Linux:
```
echo >> ~/.bashrc
echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> ~/.bashrc
eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
```

Then, in this folder:

```bash
brew install pyenv
pyenv install --list | grep -E '^\s*3\.11\.[0-9]+$' | tail -3
```
The last command gives you the latest available python version. Pick the latest one (when this repo was created, I used 3.11.18).
```
pyenv install 3.11.x
brew install pyenv-virtualenv
pyenv virtualenv 3.11.x imperfect-minds-rsa
pyenv local imperfect-minds-rsa

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# macOS: the standard build is already CPU/Apple-silicon ready
# pip install torch torchvision

pip install -r requirements.txt
jupyter lab
```

Without the pyenv-virtualenv plugin, use a plain virtual environment instead of the second and third lines:

```bash
pyenv install 3.11 && pyenv local 3.11
python -m venv .venv && source .venv/bin/activate
```

(`pyenv local` creates a `.python-version` file, and `python -m venv` creates `.venv/`; neither belongs in the repository.)

### Option B: Docker (same environment on every machine)

The image starts from the official PyTorch image (so PyTorch and torchvision match each other) and installs the other requirements on top.

```bash
docker build -t imperfect-minds-rsa .
docker run --rm -it \
  -p 127.0.0.1:8888:8888 \
  -v "$PWD":/workspace \
  -v "$HOME/data":/data \
  -v imperfect-minds-torch-cache:/root/.cache/torch \
  imperfect-minds-rsa
```

Open the link that starts with `http://127.0.0.1:8888/lab?token=...` printed in the terminal.

- `-v "$PWD":/workspace` makes this folder (and the notebooks you save) available inside the container.
- `-v "$HOME/data":/data` is where you put the THINGS-EEG2 data for the real-data notebook, so set `DATA_DIR = Path("/data/THINGS-EEG2")` there. Change the left side if your data is elsewhere.
- The last `-v` keeps the downloaded ResNet-50 weights, so they are not downloaded again every time.
- The image is large (about 6 GB, it includes CUDA, which is not needed here), so the first build takes a while. It is built for Linux on Intel/AMD processors; on Apple silicon Macs it runs under emulation and is slow, so use option A there.


## Start here: the synthetic demo

`demo_synthetic.ipynb` generates images and EEG data of its own, with a signal planted on purpose (100-400 ms after the image, electrodes at the back of the head), and then runs the six steps with a real pretrained ResNet-50, exactly as in the real-data notebook. It checks at the end that RSA finds the signal where it was planted. It takes a few minutes, and the first run downloads the model weights (about 100 MB). The figures show synthetic data, not a real result.

## Real data: THINGS-EEG2

`demo_things_eeg.ipynb` runs the same six steps on the 200 test images of THINGS-EEG2 ([Gifford et al., 2022](https://doi.org/10.1016/j.neuroimage.2022.119754); [OSF 3jk45](https://osf.io/3jk45/)) against a pretrained ResNet-50. Ten participants each saw every test image 80 times, which gives much cleaner image-by-image responses than a single viewing.

### Download the data

The notebook uses the preprocessed EEG released with the dataset, plus the 200 test images. From the `data` folder next to this one:

```bash
cd ../data
pip install requests          # if not installed yet
python download_things_eeg2.py --subjects 1 2 3
```

Leave out `--subjects` to get all 10 participants. The script reads only the test partition (`preprocessed_eeg_test.npy`) out of each participant's zip on OSF, so it downloads about 820 MB per participant instead of the full 4 GB zip. It skips files that are already there, so you can run it again if it stops halfway. It creates:

```
data/THINGS-EEG2/
    sub-01/preprocessed_eeg_test.npy     # 200 images x 80 repeats x 63 electrodes x 100 time points
    sub-02/...
    test_images/00001_aircraft_carrier/aircraft_carrier_06s.jpg ...
    image_metadata.npy                   # image order
```

By default it downloads the 63-electrode version, which is needed for the topomaps. `--channels 17` downloads the 17 occipital and parietal electrodes used in the dataset paper instead (about 220 MB per participant). RSA works with both, but the topomap then only covers the back of the head.

To download by hand instead, get `sub-XX__63_channels.zip` from the [preprocessed EEG data](https://osf.io/anp5v/) component and `test_images.zip` and `image_metadata.npy` from the [image set](https://osf.io/y63gw/) component, then unzip them into the layout above. Only `preprocessed_eeg_test.npy` is needed from each zip.

### Run the notebook

1. Set `DATA_DIR` and `SUBJECTS` in the first code cell.
2. The first time, run the `check_subject` cell: it prints the shape of the data, the time axis, the electrode names and the first test images, and checks that all 200 images are there.
3. Run the rest of the notebook.

What `dataset.load_subject` does: load the preprocessed test data (already epoched from -200 to 800 ms, downsampled to 100 Hz, baseline-corrected and noise-normalised by the dataset authors) and average the 80 repeats of each image. The noise normalisation rescales each electrode, so amplitudes are in arbitrary units, not microvolts. RSA is computed per participant, then averaged.

## Files

| File | What it is |
|---|---|
| `demo_synthetic.ipynb` | The six steps on synthetic images and EEG |
| `demo_things_eeg.ipynb` | The six steps on THINGS-EEG2 (outputs cleared) |
| `rsa.py` | RDMs, RSA, searchlight over electrodes, plotting, topomap animation. Not the `rsa` package from PyPI: keep the notebooks in this folder so they import this one |
| `dataset.py` | Loader for the THINGS-EEG2 test data |
| `../data/download_things_eeg2.py` | Downloads the THINGS-EEG2 test data and images |
| `requirements.txt` | Python packages (PyTorch comes with the Docker image) |
| `Dockerfile` | PyTorch image plus the requirements |

## Status and limits

- Not included: noise ceiling, statistical tests (e.g. permutations), cross-validation. A correlation here is not a measure of how similar the brain and the model are overall; see the post for what RSA does and does not tell us.
- The topomap is a rough picture of where on the scalp the signal is strongest. EEG blurs the signal, so it does not identify brain areas.
- The `.mp4` option of `animate_topomap` needs ffmpeg (included in the Docker image); `.gif` does not.

## Data and licences

The code is yours to reuse (see the repository licence). THINGS-EEG2 and its images have their own licences (see the OSF pages): check them before sharing anything derived from the images. Clear the outputs of `demo_things_eeg.ipynb` before committing it, so no image or data is stored in the notebook.
