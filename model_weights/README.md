# model_weights/

Place your **trained fold-0 model here before building the Docker image**. It is baked
into the image at build time (`COPY model_weights ./model_weights`).

Exactly **one** `Dataset*` folder should be present — it defines the task this image runs.
`flare_predict.py` auto-detects it (override with `--dataset`).

## Required layout

```text
model_weights/
└── Dataset00X_Name/                     # one of the 7 AutoMSC baselines
    └── <plans_folder>/                  # see the table below
        ├── dataset.json
        ├── plans.json
        └── fold_0/
            └── checkpoint_best.pth
```

## plans_folder per dataset

| Dataset                 | plans_folder                                        |
|-------------------------|-----------------------------------------------------|
| Dataset001_BrainMets    | `nnUNetCLSTrainerMTL__nnUNetPlans__3d_fullres`      |
| Dataset002_MU_Glioma    | `nnUNetCLSTrainerMTL__nnUNetPlans__3d_fullres`      |
| Dataset003_UCSD_PTGB    | `nnUNetCLSTrainerMTL__nnUNetResEncUNetMPlans__3d_fullres` |
| Dataset004_PICAI        | `nnUNetCLSTrainerMTL__nnUNetPlans__3d_fullres`      |
| Dataset005_LUNA25       | `nnUNetCLSTrainerMTL__nnUNetPlans__3d_fullres`      |
| Dataset061_PETWB_Lung   | `nnUNetCLSTrainerMTL__nnUNetPlans__3d_fullres`      |
| Dataset062_PETWB_Liver  | `nnUNetCLSTrainerMTL__nnUNetPlans__3d_fullres`      |

## Example

```text
model_weights/
└── Dataset005_LUNA25/
    └── nnUNetCLSTrainerMTL__nnUNetPlans__3d_fullres/
        ├── dataset.json
        ├── plans.json
        └── fold_0/checkpoint_best.pth
```

Model files are git-ignored — only this README and `.gitkeep` are tracked.
