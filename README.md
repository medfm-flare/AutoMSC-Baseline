# AutoMSC — Docker Submission

AutoMSC extends the nnUNet framework with joint segmentation and classification, producing
both a pixel-level segmentation mask and an image-level classification prediction from a
single model.

This branch packages a trained baseline as a **FLARE-style Docker image** that runs joint
segmentation + classification on a folder of test cases.

## Docker Submission (FLARE)

The image follows the standard test protocol:

```bash
docker load -i teamname.tar.gz
docker container run --gpus "device=1" -m 28G --name teamname --rm \
  -v $PWD/FLARE_Test/:/workspace/inputs/ \
  -v $PWD/teamname_outputs/:/workspace/outputs/ \
  teamname:latest /bin/bash -c "sh predict.sh"
```

The container reads a **flat folder of one dataset's cases** from `/workspace/inputs/`
(nnUNet naming: `{case}_0000.nii.gz`, `{case}_0001.nii.gz`, … one file per modality) and
writes to `/workspace/outputs/`:

- `{case}.nii.gz` — segmentation mask per case
- `results.csv` — classification predictions (`identifier,probs`; binary → single
  probability, multi-class → probability list)

Inference uses **fold-0** `checkpoint_best.pth` (seg + cls, no overlay video).

### Files

| File | Role |
|------|------|
| `flare_predict.py` | Batch entry point; groups modality files per case, runs `SingleFoldPredictor`, writes masks + `results.csv`. |
| `predict.sh` | Container entry — runs `flare_predict.py` on `/workspace/inputs` → `/workspace/outputs`. |
| `docker/Dockerfile` | `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime` base; installs the package + `opencv-python-headless` + `torchmetrics`; bakes in `model_weights/`. |
| `build_and_save.sh` | Builds the image and runs `docker save … \| gzip` → `<team>.tar.gz`. |
| `model_weights/` | Drop your trained model here **before building** (git-ignored). |

### 1. Add your model weights

Place your trained fold-0 model under `model_weights/` so the tree is:

```text
model_weights/
└── Dataset00X_Name/                     # one of the 7 AutoMSC baselines
    └── <plans_folder>/                  # e.g. nnUNetCLSTrainerMTL__nnUNetPlans__3d_fullres
        ├── dataset.json
        ├── plans.json
        └── fold_0/checkpoint_best.pth
```

Exactly **one** `Dataset*` folder should be present — `flare_predict.py` auto-detects it
(override with `--dataset`). See `model_weights/README.md` for the `plans_folder` per dataset.

### 2. Build and package

```bash
sh build_and_save.sh teamname            # → teamname.tar.gz
```

### 3. Run the test protocol

Use the `docker load` / `docker run` commands shown above. Outputs appear in
`teamname_outputs/`.

## License

This project is licensed under the [Apache License 2.0](https://github.com/ChingYuanYu/nnunetcls/blob/main/LICENSE).
