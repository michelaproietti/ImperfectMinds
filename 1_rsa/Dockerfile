# PyTorch and torchvision come with the base image; everything else comes from requirements.txt.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# ffmpeg is only needed to save the topomap animation as .mp4 (the .gif works without it)
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
