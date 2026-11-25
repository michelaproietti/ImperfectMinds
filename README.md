# ImperfectMinds

Code for **Imperfect Minds**, a newsletter on Substack about how brains and AI models see, think and make mistakes: [check it out](https://imperfectminds.substack.com/)!

Each issue that includes an analysis has its own folder here, with notebooks you can run to reproduce the figures in the post and try the analysis yourself. Each folder has a README with setup instructions, the data it needs and how to download it.

## Issues

| Folder | Issue | What it is about |
|---|---|---|
| [`1_rsa/`](1_rsa/) | *Comparing minds*, part 1 | Which images does a brain treat as alike, and does a vision model treat the same images as alike? We compare a pretrained ResNet-50 with human EEG responses to 200 object images (THINGS-EEG2) using representational similarity analysis (RSA). We look at when the two agree (time course) and where on the scalp (topomaps). A synthetic demo that needs no downloads shows the method first. |

## Repository layout

- `1_rsa/`, … : one folder per issue, each with its own README, notebooks and code.
- `data/` : datasets and the download scripts for them. The datasets themselves are not part of the repository; each issue's README explains how to get the ones it needs.

## Licence

The code is released under the [MIT licence](LICENSE). The datasets used in the issues have their own licences, linked from each issue's README.
