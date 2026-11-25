#!/usr/bin/env python3
"""Download the THINGS-EEG2 test partition (Gifford et al., 2022; OSF 3jk45).

Only what the RSA notebook needs: for each participant, preprocessed_eeg_test.npy
(200 test images x 80 repeats x channels x 100 time points), plus the 200 test images.
The training partition is skipped: the file is read straight out of the zip on OSF
with HTTP range requests, so the full 4 GB zip is never downloaded.

    python download_things_eeg2.py                     # participants 1-10, 63 channels
    python download_things_eeg2.py --subjects 1 2 3    # some participants
    python download_things_eeg2.py --channels 17       # the 17 occipito-parietal channels only (smaller)

Result:
    THINGS-EEG2/
        sub-01/preprocessed_eeg_test.npy     (about 820 MB with 63 channels, 220 MB with 17)
        ...
        test_images/00001_aircraft_carrier/aircraft_carrier_06s.jpg ...
        image_metadata.npy
"""
import argparse
import io
import shutil
import time
import zipfile
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API = "https://api.osf.io/v2/nodes/{}/files/osfstorage/"
EEG_NODE, IMAGES_NODE = "anp5v", "y63gw"   # "Preprocessed EEG data" and "Image set" components

session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=Retry(
    total=10, backoff_factor=3, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=None)))


def list_files(node):
    """{file name: download url} for an OSF component (the OSF API pages its results)."""
    files, url = {}, API.format(node)
    while url:
        page = session.get(url, timeout=60)
        page.raise_for_status()
        page = page.json()
        files.update({f["attributes"]["name"]: f["links"]["download"] for f in page["data"]})
        url = page["links"].get("next")
    return files


class RemoteFile(io.RawIOBase):
    """A file on a web server that zipfile can seek in, read with HTTP range requests."""

    def __init__(self, url):
        r = session.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.url, self.size, self.pos = r.url, int(r.headers["Content-Length"]), 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buffer):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buffer), self.size) - 1
        r = session.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=120)
        r.raise_for_status()
        n = len(r.content)
        buffer[:n] = r.content
        self.pos += n
        return n


def extract_remote(url, member_suffix, dest):
    """Copy the one member of a remote zip whose name ends with `member_suffix` to `dest`."""
    zf = zipfile.ZipFile(io.BufferedReader(RemoteFile(url), buffer_size=16 << 20))
    (member,) = [m for m in zf.infolist() if m.filename.endswith(member_suffix)]
    if dest.exists() and dest.stat().st_size == member.file_size:
        print(f"skip   {dest}")
        return
    print(f"get    {dest} ({member.file_size / 1e6:.0f} MB)", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with zf.open(member) as src, open(tmp, "wb") as out:
        shutil.copyfileobj(src, out, length=16 << 20)
    tmp.rename(dest)


def download(url, dest):
    if dest.exists():
        print(f"skip   {dest}")
        return
    print(f"get    {dest}", flush=True)
    with session.get(url, stream=True, timeout=120) as r, open(dest, "wb") as out:
        r.raise_for_status()
        shutil.copyfileobj(r.raw, out, length=16 << 20)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 11)))
    p.add_argument("--channels", type=int, choices=[17, 63], default=63)
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "THINGS-EEG2")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    images = list_files(IMAGES_NODE)
    download(images["image_metadata.npy"], args.out / "image_metadata.npy")
    if not (args.out / "test_images").exists():
        download(images["test_images.zip"], args.out / "test_images.zip")
        with zipfile.ZipFile(args.out / "test_images.zip") as zf:
            zf.extractall(args.out)
        (args.out / "test_images.zip").unlink()

    eeg = list_files(EEG_NODE)
    failed = []
    for s in args.subjects:
        name = f"sub-{s:02d}" + ("__63_channels" if args.channels == 63 else "") + ".zip"
        for attempt in range(5):
            try:
                extract_remote(eeg[name], "preprocessed_eeg_test.npy",
                               args.out / f"sub-{s:02d}" / "preprocessed_eeg_test.npy")
                break
            except (requests.RequestException, OSError) as e:
                print(f"       retry {attempt + 1}: {e}")
                time.sleep(30)
        else:
            failed.append(s)
    print("\nFailed:", failed or "none")


if __name__ == "__main__":
    main()
