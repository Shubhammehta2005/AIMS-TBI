#!/usr/bin/env python3
"""
AIMS-TBI: Adaptive brain-mask normalization + nnU-Net ResEnc L.

Input files:
  dataset/T1_for_HDBET/scan_XXXX_T1.nii.gz
  dataset/SkullStripped/scan_XXXX_T1_bet.nii.gz
  dataset/ChallengeFiles/scan_XXXX_Lesion.nii.gz

The original T1 image is normalized using statistics computed inside
the HD-BET brain mask. The normalized original T1 is the model input.

IMPORTANT:
- The split is scan-level, not patient-level.
- The paper does not publish every DA5 augmentation parameter.
- This is a paper-inspired implementation, not a bit-exact reproduction.
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
HOLDOUT_FRACTION = 0.20

PLANNER = "nnUNetPlannerResEncL"
PLANS_ID = "nnUNetResEncUNetLPlans"


# ============================================================
# 2. COMMAND EXECUTION
# ============================================================

def run(cmd, env):
    """Execute a command and stop if it fails."""
    cmd = [str(x) for x in cmd]

    print("\n>>>", " ".join(cmd), flush=True)

    subprocess.run(
        cmd,
        check=True,
        env=env
    )


# ============================================================
# 3. FILE IDENTIFICATION
# ============================================================

def case_id_from_mask(mask_path):
    suffix = "_Lesion.nii.gz"

    if not mask_path.name.endswith(suffix):
        raise ValueError(
            f"Unexpected mask filename: {mask_path.name}"
        )

    return mask_path.name[:-len(suffix)]


# ============================================================
# 4. PAPER-SPECIFIC NORMALIZATION
# ============================================================

def normalize_case(
    t1_path,
    bet_path,
    mask_path,
    image_out,
    label_out
):
    """
    Calculate mean/std inside the HD-BET brain mask.

    Normalize the original T1 image using those statistics.
    Do not use the skull-stripped intensities as model input.
    """

    t1_nii = nib.load(str(t1_path))
    bet_nii = nib.load(str(bet_path))
    mask_nii = nib.load(str(mask_path))

    # --------------------------------------------------------
    # Validate shapes
    # --------------------------------------------------------

    if (
        t1_nii.shape != bet_nii.shape
        or t1_nii.shape != mask_nii.shape
    ):
        raise ValueError(
            f"Shape mismatch: "
            f"T1={t1_nii.shape}, "
            f"HD-BET={bet_nii.shape}, "
            f"label={mask_nii.shape}"
        )

    # --------------------------------------------------------
    # Validate spatial geometry
    # --------------------------------------------------------

    if not (
        np.allclose(
            t1_nii.affine,
            bet_nii.affine,
            atol=1e-3
        )
        and
        np.allclose(
            t1_nii.affine,
            mask_nii.affine,
            atol=1e-3
        )
    ):
        raise ValueError(
            "Affine mismatch between T1, HD-BET and lesion mask"
        )

    # --------------------------------------------------------
    # Load voxel data
    # --------------------------------------------------------

    x = np.asarray(
        t1_nii.dataobj,
        dtype=np.float32
    )

    bet = np.asarray(
        bet_nii.dataobj,
        dtype=np.float32
    )

    y = np.asarray(mask_nii.dataobj)

    if not (
        np.isfinite(x).all()
        and np.isfinite(bet).all()
        and np.isfinite(y).all()
    ):
        raise ValueError(
            "Non-finite voxel values detected"
        )

    # --------------------------------------------------------
    # Generate brain mask from HD-BET output
    # --------------------------------------------------------

    brain = np.abs(bet) > 1e-8

    n_brain = int(brain.sum())

    if n_brain < 1000:
        raise ValueError(
            f"Brain mask is implausibly small: {n_brain}"
        )

    # --------------------------------------------------------
    # Adaptive normalization
    # --------------------------------------------------------

    brain_values = x[brain].astype(np.float64)

    mu = float(brain_values.mean())

    sigma = float(brain_values.std(ddof=0))

    if not np.isfinite(sigma) or sigma <= 1e-8:
        raise ValueError(
            f"Invalid brain-mask standard deviation: {sigma}"
        )

    # Normalize the entire original T1 volume.
    # Statistics are calculated only inside the brain mask.
    #
    # No percentile clipping.
    # No forced zeroing outside the brain mask.

    xn = (
        (x.astype(np.float64) - mu) / sigma
    ).astype(np.float32)

    # Binary segmentation labels
    yn = (y > 0).astype(np.uint8)

    # --------------------------------------------------------
    # Save normalized image
    # --------------------------------------------------------

    image_out.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    label_out.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    image_header = t1_nii.header.copy()

    image_header.set_data_dtype(np.float32)

    nib.save(
        nib.Nifti1Image(
            xn,
            t1_nii.affine,
            image_header
        ),
        str(image_out)
    )

    # --------------------------------------------------------
    # Save label
    # --------------------------------------------------------

    label_header = mask_nii.header.copy()

    label_header.set_data_dtype(np.uint8)

    nib.save(
        nib.Nifti1Image(
            yn,
            mask_nii.affine,
            label_header
        ),
        str(label_out)
    )

    return {
        "brain_voxels": n_brain,
        "brain_mean": mu,
        "brain_std": sigma,
        "lesion_voxels": int(yn.sum()),
        "shape": list(x.shape),
        "spacing": [
            float(v)
            for v in t1_nii.header.get_zooms()[:3]
        ]
    }


# ============================================================
# 5. DISCOVER AND VALIDATE CASES
# ============================================================

def build_cases(
    t1_dir,
    bet_dir,
    label_dir,
    skip_invalid
):

    masks = sorted(
        label_dir.glob("*_Lesion.nii.gz")
    )

    if not masks:
        raise RuntimeError(
            f"No lesion masks found in {label_dir}"
        )

    cases = {}
    failures = {}

    for seg in masks:

        sid = case_id_from_mask(seg)

        t1 = t1_dir / f"{sid}_T1.nii.gz"

        bet = bet_dir / f"{sid}_T1_bet.nii.gz"

        try:

            if not t1.is_file():
                raise FileNotFoundError(
                    f"Missing original T1: {t1}"
                )

            if not bet.is_file():
                raise FileNotFoundError(
                    f"Missing HD-BET image: {bet}"
                )

            # Read the label now to catch corrupt files.
            label = nib.load(str(seg))

            y = np.asarray(label.dataobj)

            if not np.isfinite(y).all():
                raise ValueError(
                    "Lesion mask contains non-finite values"
                )

            cases[sid] = {
                "t1": t1,
                "bet": bet,
                "mask": seg,
                "has_lesion": bool(
                    np.any(y > 0)
                )
            }

        except Exception as exc:

            failures[sid] = str(exc)

            print(
                f"[SKIP] {sid}: {exc}",
                flush=True
            )

            if not skip_invalid:
                raise

    if len(cases) < 10:
        raise RuntimeError(
            f"Only {len(cases)} valid cases. "
            "Refusing to create a small split."
        )

    print(
        f"\nValid cases: {len(cases)}"
    )

    print(
        f"Invalid cases: {len(failures)}"
    )

    return cases, failures


# ============================================================
# 6. PREPARE DATASET AND HOLDOUT SPLIT
# ============================================================

def prepare_dataset(
    t1_dir,
    bet_dir,
    label_dir,
    raw_dataset,
    work_dir,
    skip_invalid
):

    cases, failures = build_cases(
        t1_dir,
        bet_dir,
        label_dir,
        skip_invalid
    )

    ids = sorted(cases)

    strata = np.array([
        int(cases[sid]["has_lesion"])
        for sid in ids
    ])

    # Stratify by lesion presence where possible.
    counts = np.bincount(
        strata,
        minlength=2
    )

    can_stratify = (
        len(counts) == 2
        and np.all(counts >= 2)
    )

    if can_stratify:

        train_ids, holdout_ids = train_test_split(
            ids,
            test_size=HOLDOUT_FRACTION,
            random_state=SEED,
            stratify=strata
        )

    else:

        print(
            "[WARNING] Cannot stratify by lesion presence."
        )

        train_ids, holdout_ids = train_test_split(
            ids,
            test_size=HOLDOUT_FRACTION,
            random_state=SEED
        )

    train_ids = sorted(train_ids)

    holdout_ids = sorted(holdout_ids)

    # Ensure no case is present in both splits.
    assert set(train_ids).isdisjoint(
        set(holdout_ids)
    )

    # --------------------------------------------------------
    # Clean stale dataset files
    # --------------------------------------------------------

    if raw_dataset.exists():
        shutil.rmtree(raw_dataset)

    images_tr = raw_dataset / "imagesTr"

    labels_tr = raw_dataset / "labelsTr"

    images_ts = raw_dataset / "imagesTs"

    labels_ts = raw_dataset / "labelsTs"

    for directory in (
        images_tr,
        labels_tr,
        images_ts,
        labels_ts
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    report = {}

    # --------------------------------------------------------
    # Write training and holdout cases
    # --------------------------------------------------------

    splits = [
        (
            "train",
            train_ids,
            images_tr,
            labels_tr
        ),
        (
            "holdout",
            holdout_ids,
            images_ts,
            labels_ts
        )
    ]

    for (
        split_name,
        split_ids,
        image_dir,
        label_dir
    ) in splits:

        for sid in split_ids:

            case = cases[sid]

            print(
                f"[PREP:{split_name}] {sid}",
                flush=True
            )

            info = normalize_case(
                case["t1"],
                case["bet"],
                case["mask"],
                image_dir / f"{sid}_0000.nii.gz",
                label_dir / f"{sid}.nii.gz"
            )

            report[sid] = {
                **info,
                "split": split_name
            }

    # --------------------------------------------------------
    # nnU-Net dataset metadata
    # --------------------------------------------------------

    dataset_json = {
        "channel_names": {
            "0": "T1"
        },
        "labels": {
            "background": 0,
            "lesion": 1
        },
        "numTraining": len(train_ids),
        "file_ending": ".nii.gz"
    }

    (
        raw_dataset / "dataset.json"
    ).write_text(
        json.dumps(
            dataset_json,
            indent=2
        )
    )

    # --------------------------------------------------------
    # Save split and preprocessing report
    # --------------------------------------------------------

    work_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    split_data = {
        "seed": SEED,
        "holdout_fraction": HOLDOUT_FRACTION,
        "train_ids": train_ids,
        "holdout_ids": holdout_ids,
        "note": (
            "Scan-level split. Patient linkage and "
            "longitudinal grouping are unavailable."
        )
    }

    (
        work_dir / "split.json"
    ).write_text(
        json.dumps(
            split_data,
            indent=2
        )
    )

    (
        work_dir / "preparation_report.json"
    ).write_text(
        json.dumps(
            {
                "prepared_count": len(report),
                "train_count": len(train_ids),
                "holdout_count": len(holdout_ids),
                "failures_before_split": failures,
                "cases": report
            },
            indent=2
        )
    )

    print("\nDataset preparation completed.")

    print(
        f"Training cases: {len(train_ids)}"
    )

    print(
        f"Holdout cases: {len(holdout_ids)}"
    )

    print(
        f"Reports saved to: {work_dir}"
    )


# ============================================================
# 7. nnU-NET ENVIRONMENT
# ============================================================

def setup_nnunet(root):

    root = Path(root)

    raw = root / "nnUNet_raw"

    pre = root / "nnUNet_preprocessed"

    results = root / "nnUNet_results"

    for directory in (raw, pre, results):

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    env = os.environ.copy()

    env.update({
        "nnUNet_raw": str(raw),
        "nnUNet_preprocessed": str(pre),
        "nnUNet_results": str(results)
    })

    return env, raw, pre, results


# ============================================================
# 8. FIND AND VALIDATE PLANS
# ============================================================

def find_plans(preprocessed_dataset):

    expected = (
        preprocessed_dataset /
        f"{PLANS_ID}.json"
    )

    if expected.is_file():
        return expected

    candidates = list(
        preprocessed_dataset.glob(
            "*ResEnc*L*Plans*.json"
        )
    )

    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(
        f"Could not uniquely identify plans file.\n"
        f"Expected: {expected}\n"
        f"Candidates: {[p.name for p in candidates]}"
    )


def set_no_normalization(plans_path):

    data = json.loads(
        plans_path.read_text()
    )

    configs = data.get("configurations")

    if not isinstance(configs, dict) or not configs:
        raise RuntimeError(
            f"Unexpected plans schema: {plans_path}"
        )

    found = []

    for name, cfg in configs.items():

        schemes = cfg.get(
            "normalization_schemes"
        )

        if (
            isinstance(schemes, dict)
            and "0" in schemes
        ):

            schemes["0"] = "NoNormalization"

            found.append(name)

    if not found:
        raise RuntimeError(
            "Could not find normalization_schemes['0'] "
            f"in {plans_path}"
        )

    plans_path.write_text(
        json.dumps(
            data,
            indent=2
        )
    )

    # Re-read the file and verify the changes.
    check = json.loads(
        plans_path.read_text()
    )

    for name in found:

        if (
            check["configurations"][name]
            ["normalization_schemes"]["0"]
            != "NoNormalization"
        ):
            raise RuntimeError(
                f"Normalization verification failed: {name}"
            )

    print(
        "[OK] NoNormalization set for:",
        ", ".join(found)
    )


# ============================================================
# 9. PLAN AND PREPROCESS
# ============================================================

def plan_and_preprocess(
    dataset_id,
    env,
    pre_root
):

    # NOTE:
    # Verify these CLI entry points against your installed
    # nnU-Net version before running the full pipeline.

    run(
        [
            "nnUNetv2_extract_fingerprint",
            "-d",
            dataset_id
        ],
        env
    )

    run(
        [
            "nnUNetv2_plan_experiment",
            "-d",
            dataset_id,
            "-pl",
            PLANNER
        ],
        env
    )

    pre_ds = (
        pre_root /
        f"Dataset{dataset_id:03d}_AIMSTBI"
    )

    plans_path = find_plans(pre_ds)

    set_no_normalization(plans_path)

    run(
        [
            "nnUNetv2_preprocess",
            "-d",
            dataset_id,
            "-plans_name",
            plans_path.stem
        ],
        env
    )

    return plans_path.stem


# ============================================================
# 10. TRAIN
# ============================================================

def train(
    dataset_id,
    plans_id,
    env,
    folds
):

    for fold in folds:

        run(
            [
                "nnUNetv2_train",
                dataset_id,
                "3d_fullres",
                fold,
                "-p",
                plans_id
            ],
            env
        )


# ============================================================
# 11. PREDICT ON HELD-OUT DATA
# ============================================================

def predict(
    dataset_id,
    plans_id,
    env,
    raw_dataset,
    output_dir,
    folds
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    run(
        [
            "nnUNetv2_predict",
            "-i",
            raw_dataset / "imagesTs",
            "-o",
            output_dir,
            "-d",
            dataset_id,
            "-c",
            "3d_fullres",
            "-p",
            plans_id,
            "-f",
            *folds
        ],
        env
    )


# ============================================================
# 12. HOLDOUT EVALUATION
# ============================================================

def evaluate(
    pred_dir,
    labels_dir,
    report_path
):

    rows = []

    for pred in sorted(
        pred_dir.glob("*.nii.gz")
    ):

        sid = pred.name.removesuffix(
            ".nii.gz"
        )

        gt_path = labels_dir / f"{sid}.nii.gz"

        if not gt_path.exists():

            print(
                f"[EVAL SKIP] Missing label for {sid}"
            )

            continue

        pred_nii = nib.load(str(pred))

        gt_nii = nib.load(str(gt_path))

        if (
            pred_nii.shape != gt_nii.shape
            or
            not np.allclose(
                pred_nii.affine,
                gt_nii.affine,
                atol=1e-3
            )
        ):
            raise ValueError(
                f"Prediction/label geometry mismatch: {sid}"
            )

        p = (
            np.asarray(pred_nii.dataobj) > 0
        )

        y = (
            np.asarray(gt_nii.dataobj) > 0
        )

        # ----------------------------------------------------
        # Lesion Dice
        # ----------------------------------------------------

        intersection = int(
            np.logical_and(p, y).sum()
        )

        pred_voxels = int(p.sum())

        gt_voxels = int(y.sum())

        denom = pred_voxels + gt_voxels

        if denom == 0:
            lesion_dice = 1.0
        else:
            lesion_dice = (
                2.0 * intersection / denom
            )

        # ----------------------------------------------------
        # Non-lesion Dice
        # ----------------------------------------------------

        non_intersection = int(
            np.logical_and(~p, ~y).sum()
        )

        non_denom = int(
            (~p).sum() + (~y).sum()
        )

        if non_denom == 0:
            nonlesion_dice = 1.0
        else:
            nonlesion_dice = (
                2.0 * non_intersection / non_denom
            )

        overall_dice = (
            lesion_dice + nonlesion_dice
        ) / 2.0

        rows.append({
            "case": sid,
            "lesion_dice": lesion_dice,
            "nonlesion_dice": nonlesion_dice,
            "overall_dice": overall_dice,
            "predicted_lesion_voxels": pred_voxels,
            "ground_truth_lesion_voxels": gt_voxels
        })

    if not rows:
        raise RuntimeError(
            "No predictions could be evaluated."
        )

    report = {
        "evaluated_cases": len(rows),
        "mean_lesion_dice": float(
            np.mean([
                r["lesion_dice"]
                for r in rows
            ])
        ),
        "mean_nonlesion_dice": float(
            np.mean([
                r["nonlesion_dice"]
                for r in rows
            ])
        ),
        "mean_overall_dice": float(
            np.mean([
                r["overall_dice"]
                for r in rows
            ])
        ),
        "cases": rows,
        "note": (
            "Local holdout metrics. This does not reproduce "
            "the paper's official mean-position metric."
        )
    }

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2
        )
    )

    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k != "cases"
            },
            indent=2
        )
    )

    print(
        f"Evaluation report: {report_path}"
    )


# ============================================================
# 13. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--project",
        type=Path,
        default=Path(
            "/home/student_login/Shubham"
        )
    )

    parser.add_argument(
        "--t1-dir",
        type=Path,
        default=Path(
            "/home/student_login/Shubham/dataset/T1_for_HDBET"
        )
    )

    parser.add_argument(
        "--bet-dir",
        type=Path,
        default=Path(
            "/home/student_login/Shubham/dataset/SkullStripped"
        )
    )

    parser.add_argument(
        "--label-dir",
        type=Path,
        default=Path(
            "/home/student_login/Shubham/dataset/ChallengeFiles"
        )
    )

    parser.add_argument(
        "--dataset-id",
        type=int,
        default=501
    )

    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip missing or corrupt input pairs"
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare data without training"
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Run nnU-Net planning, training and evaluation"
    )

    parser.add_argument(
        "--folds",
        nargs="+",
        default=["0", "1", "2", "3", "4"]
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Check nnU-Net installation
    # --------------------------------------------------------

    if not args.prepare_only:

        try:
            import nnunetv2  # noqa: F401

        except ImportError:

            raise SystemExit(
                "nnU-Net v2 is missing. "
                "Activate your server environment first."
            )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    work_dir = (
        args.project /
        "outputs_paper_resencl"
    )

    env, nnraw, nnpre, nnresults = setup_nnunet(
        args.project
    )

    raw_dataset = (
        nnraw /
        f"Dataset{args.dataset_id:03d}_AIMSTBI"
    )

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    prepare_dataset(
        args.t1_dir,
        args.bet_dir,
        args.label_dir,
        raw_dataset,
        work_dir,
        args.skip_invalid
    )

    if args.prepare_only:

        print(
            "\nPreparation-only run completed."
        )

        print(
            "Inspect split.json and preparation_report.json "
            "before training."
        )

        return

    if not args.train:

        print(
            "Data prepared only. Use --train after verification."
        )

        return

    # --------------------------------------------------------
    # Plan, preprocess, train and predict
    # --------------------------------------------------------

    plans_id = plan_and_preprocess(
        args.dataset_id,
        env,
        nnpre
    )

    train(
        args.dataset_id,
        plans_id,
        env,
        args.folds
    )

    pred_dir = (
        work_dir /
        "holdout_predictions"
    )

    predict(
        args.dataset_id,
        plans_id,
        env,
        raw_dataset,
        pred_dir,
        args.folds
    )

    evaluate(
        pred_dir,
        raw_dataset / "labelsTs",
        work_dir / "holdout_metrics.json"
    )


if __name__ == "__main__":
    main()