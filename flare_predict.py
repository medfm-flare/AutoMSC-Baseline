"""
FLARE-AutoMSC Docker submission entry point.

Runs single fold-0 joint segmentation + classification inference over a flat folder
of one dataset's cases (nnUNet naming: ``{case}_0000.nii.gz``, ``{case}_0001.nii.gz`` …)
and writes, to the output folder:

    - ``{case}.nii.gz``   : segmentation mask per case
    - ``results.csv``     : classification predictions (columns: identifier, probs)

The target dataset (one of the 7 AutoMSC baselines) is auto-detected from the single
``Dataset*`` subfolder present under ``--model_dir``; override with ``--dataset``.
Reuses ``SingleFoldPredictor`` / ``MODEL_REGISTRY`` from ``baseline_infer.py``.
"""

import os
import re
import argparse
from collections import defaultdict

import numpy as np
import torch
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import join

from baseline_infer import SingleFoldPredictor, MODEL_REGISTRY
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO


def group_images_by_idx(input_folder):
    """Group modality files by case id. Matches ``{case}_0000.nii.gz`` naming."""
    pattern = re.compile(r"^(.*)_\d{4}\.nii\.gz$")
    grouped = defaultdict(list)
    for entry in os.scandir(input_folder):
        if entry.name.endswith(".nii.gz"):
            match = pattern.match(entry.name)
            if match:
                grouped[match.group(1)].append(entry.path)
    for paths in grouped.values():
        paths.sort()  # ensures channel order _0000, _0001, ...
    return dict(grouped)


def detect_dataset(model_dir, override=None):
    """Return the dataset_name to use, from an explicit override or auto-detection."""
    if override is not None:
        if override not in MODEL_REGISTRY:
            raise ValueError(
                f"--dataset {override!r} is not a known baseline. "
                f"Choose from: {list(MODEL_REGISTRY.keys())}")
        return override

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"model_dir does not exist: {model_dir}")

    candidates = [d for d in os.listdir(model_dir)
                  if d in MODEL_REGISTRY and os.path.isdir(join(model_dir, d))]
    if len(candidates) == 0:
        raise RuntimeError(
            f"No known Dataset* model folder found under {model_dir}. "
            f"Expected one of {list(MODEL_REGISTRY.keys())}.")
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple model folders found under {model_dir}: {candidates}. "
            f"This image handles a single dataset per run — keep only one, "
            f"or pass --dataset explicitly.")
    return candidates[0]


def build_predictor(model_dir, dataset_name, device, fold, checkpoint):
    plans_folder = MODEL_REGISTRY[dataset_name]["plans_folder"]
    model_training_output_dir = join(model_dir, dataset_name, plans_folder)

    dev = torch.device(device, 0) if device != "cpu" else torch.device("cpu")
    perform_on_device = (dev.type != "cpu")

    predictor = SingleFoldPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=perform_on_device,
        device=dev,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir, fold=fold, checkpoint_name=checkpoint)
    predictor.network.to(dev)
    return predictor


def main():
    parser = argparse.ArgumentParser(
        description="FLARE-AutoMSC single-fold seg+cls inference over a flat input folder.")
    parser.add_argument("--input_dir", default="/workspace/inputs",
                        help="Flat folder of {case}_000X.nii.gz images.")
    parser.add_argument("--output_dir", default="/workspace/outputs",
                        help="Where to write {case}.nii.gz masks and results.csv.")
    parser.add_argument("--model_dir", default="/workspace/model_weights",
                        help="Folder containing a single Dataset*/<plans_folder>/ model.")
    parser.add_argument("--dataset", default=None,
                        help="Override dataset auto-detection (e.g. Dataset005_LUNA25).")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--checkpoint", default="checkpoint_best.pth")
    parser.add_argument("--use_softmax", action="store_true",
                        help="Apply softmax to the segmentation output (default: argmax logits).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset_name = detect_dataset(args.model_dir, args.dataset)
    print(f"[flare_predict] dataset: {dataset_name}")
    predictor = build_predictor(
        args.model_dir, dataset_name, args.device, args.fold, args.checkpoint)

    expected = len(predictor.dataset_json["channel_names"])
    cases = group_images_by_idx(args.input_dir)
    if not cases:
        raise RuntimeError(f"No '{{case}}_000X.nii.gz' images found in {args.input_dir}")
    print(f"[flare_predict] {len(cases)} case(s); expecting {expected} modality file(s) each.")

    identifiers, all_probs = [], []
    for case in tqdm(sorted(cases.keys()), desc="cases"):
        image_paths = cases[case]
        if len(image_paths) != expected:
            raise ValueError(
                f"Case {case}: expected {expected} modality file(s) "
                f"(channels={list(predictor.dataset_json['channel_names'].values())}), "
                f"but found {len(image_paths)}: {[os.path.basename(p) for p in image_paths]}")

        image_npy, props = SimpleITKIO().read_images(image_paths)
        segmentation, cls_probs = predictor.inference(
            image_npy, props, use_softmax=args.use_softmax)

        seg_path = join(args.output_dir, f"{case}.nii.gz")
        sitk_img = sitk.GetImageFromArray(segmentation.numpy().astype(np.uint8))
        sitk_img.SetSpacing(props["sitk_stuff"]["spacing"])
        sitk_img.SetOrigin(props["sitk_stuff"]["origin"])
        sitk_img.SetDirection(props["sitk_stuff"]["direction"])
        sitk.WriteImage(sitk_img, seg_path)

        identifiers.append(case)
        all_probs.append(cls_probs.flatten().tolist())

    results = pd.DataFrame({"identifier": identifiers, "probs": all_probs})
    results_path = join(args.output_dir, "results.csv")
    results.to_csv(results_path, index=False)
    print(f"[flare_predict] wrote {len(identifiers)} mask(s) and {results_path}")


if __name__ == "__main__":
    main()
