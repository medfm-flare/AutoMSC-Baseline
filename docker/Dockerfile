# FLARE-AutoMSC submission image: joint segmentation + classification (nnUNet MTL).
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# nnUNet's paths.py only warns when these are unset; define dummies to keep logs clean.
ENV nnUNet_raw=/workspace/nnUNet_raw \
    nnUNet_preprocessed=/workspace/nnUNet_preprocessed \
    nnUNet_results=/workspace/nnUNet_results \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# Install the package and its dependencies first (better layer caching).
COPY pyproject.toml setup.py README.md LICENSE ./
COPY nnunetv2 ./nnunetv2
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir opencv-python-headless torchmetrics

# Inference code + baked-in model weights + entry script.
COPY baseline_infer.py flare_predict.py predict.sh ./
COPY model_weights ./model_weights

CMD ["/bin/bash", "-c", "sh predict.sh"]
