# ============================================================
# AIMS-TBI 2026
# 3D Patch-Based U-Net
# Subject-level 80/20 Train-Test Split
# 5-Fold Stratified CV on Training Set
# Weighted BCE + Tversky + Boundary Loss
#
# IMPORTANT:
#   - Model INPUT = HD-BET skull-stripped T1 images
#   - Ground truth = original Lesion masks from ChallengeFiles
#   - Entire MRI remains 3D; patches are used only for memory
#   - Hausdorff is calculated only during final TEST evaluation
# ============================================================

import os
import re
import glob
import json
import random
import math
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm


# EXPERIMENT V4: BCE(0.25) + Tversky(1.0) + Boundary(0.1)\n# Validation: lesion-focused + random/background patches\n\n# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = "/home/student_login/Shubham"

# ------------------------------------------------------------
# INPUT DIRECTORIES
# ------------------------------------------------------------
# HD-BET outputs are used as the MODEL INPUT.
#
# Your SkullStripped directory contains both:
#   scan_XXXX_T1.nii.gz
#   scan_XXXX_T1_bet.nii.gz
#
# We explicitly use *_T1_bet.nii.gz.

HDBET_DIR = os.path.join(
    PROJECT_DIR,
    "dataset",
    "SkullStripped"
)

# Original dataset is used ONLY for lesion ground-truth masks.
LESION_DIR = os.path.join(
    PROJECT_DIR,
    "dataset",
    "ChallengeFiles"
)

# ------------------------------------------------------------
# OUTPUTS
# ------------------------------------------------------------

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "outputs_3d_v4"
)

MODEL_DIR = os.path.join(
    OUTPUT_DIR,
    "models"
)

PREDICTION_DIR = os.path.join(
    OUTPUT_DIR,
    "predictions"
)

LOG_DIR = os.path.join(
    OUTPUT_DIR,
    "logs"
)

RESULT_DIR = os.path.join(
    OUTPUT_DIR,
    "results"
)

for directory in [
    OUTPUT_DIR,
    MODEL_DIR,
    PREDICTION_DIR,
    LOG_DIR,
    RESULT_DIR
]:
    os.makedirs(
        directory,
        exist_ok=True
    )


# ------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else
    "mps"
    if torch.backends.mps.is_available()
    else
    "cpu"
)

print(
    f"[INFO] Device: {DEVICE}"
)


# ------------------------------------------------------------
# RANDOM SEED
# ------------------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ------------------------------------------------------------
# DATA SPLIT
# ------------------------------------------------------------

TEST_SIZE = 0.20
NUM_FOLDS = 5


# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------

NUM_EPOCHS = 30
BATCH_SIZE = 2

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

EARLY_STOP_PATIENCE = 5
LR_PATIENCE = 3
LR_FACTOR = 0.5


# ------------------------------------------------------------
# 3D PATCH
# ------------------------------------------------------------

PATCH_SIZE = (
    96,
    96,
    32
)

# Training patch sampling:
#
# Most random patches in this dataset can contain only normal
# brain. We deliberately sample lesion-centered patches when
# possible so that the model sees enough positive examples.

LESION_PATCH_PROB = 0.80

# Number of training patches sampled from each subject per epoch.
PATCHES_PER_SUBJECT = 1

# Validation uses two patches per subject:
#   - one lesion-focused patch for lesion-positive subjects
#   - one random/background patch
# This prevents validation from looking only at lesion-centered patches.
VAL_PATCHES_PER_SUBJECT = 2

# Approximate fraction of training samples drawn from lesion-positive
# subjects after the subject-level split.
TRAIN_LESION_SUBJECT_PROB = 0.80

# Fraction of patch size used as maximum random offset around
# a lesion voxel.
LESION_CENTER_JITTER = 0.25


# ------------------------------------------------------------
# PATCH-BASED INFERENCE
# ------------------------------------------------------------

INFERENCE_OVERLAP = 0.50


# ------------------------------------------------------------
# LOSS
# ------------------------------------------------------------

TVERSKY_ALPHA = 0.30
TVERSKY_BETA = 0.70

BCE_WEIGHT = 0.25
TVERSKY_WEIGHT = 1.0
BOUNDARY_WEIGHT = 0.10

SMOOTH = 1e-6


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

BASE_CHANNELS = 16


# ------------------------------------------------------------
# TRAINING OUTPUT
# ------------------------------------------------------------

# Set to 1 if you want every training iteration printed.
# This can generate a LOT of terminal output.
PRINT_EVERY_ITER = 0

NUM_WORKERS = 0


# ------------------------------------------------------------
# MIXED PRECISION
# ------------------------------------------------------------

USE_AMP = (
    DEVICE.type == "cuda"
)


print(
    f"[INFO] HD-BET input directory: {HDBET_DIR}"
)

print(
    f"[INFO] Lesion-mask directory: {LESION_DIR}"
)

print(
    f"[INFO] Output directory: {OUTPUT_DIR}"
)

print(
    f"[INFO] Patch size: {PATCH_SIZE}"
)

print(
    f"[INFO] AMP enabled: {USE_AMP}"
)


# ============================================================
# 2. SUBJECT DISCOVERY
# ============================================================

def get_subject_pairs():
    """
    Pair:
        HD-BET skull-stripped T1
    with:
        original lesion mask

    IMPORTANT:
        The original T1 files are NOT used.
    """

    print(
        "\n[STEP] Searching for HD-BET T1 images..."
    )

    hdbet_files = glob.glob(
        os.path.join(
            HDBET_DIR,
            "*_T1_bet.nii.gz"
        )
    )

    lesion_files = glob.glob(
        os.path.join(
            LESION_DIR,
            "*_Lesion.nii.gz"
        )
    )

    print(
        f"[INFO] HD-BET T1 files found: "
        f"{len(hdbet_files)}"
    )

    print(
        f"[INFO] Lesion masks found: "
        f"{len(lesion_files)}"
    )

    lesion_by_id = {}

    lesion_pattern = re.compile(
        r"scan_(\d+)_Lesion\.nii\.gz$"
    )

    for path in lesion_files:

        match = lesion_pattern.match(
            os.path.basename(path)
        )

        if match:
            sid = match.group(1)
            lesion_by_id[sid] = path

    pairs = []

    t1_pattern = re.compile(
        r"scan_(\d+)_T1_bet\.nii\.gz$"
    )

    for t1_path in sorted(hdbet_files):

        match = t1_pattern.match(
            os.path.basename(t1_path)
        )

        if not match:
            continue

        sid = match.group(1)

        if sid not in lesion_by_id:
            print(
                f"[WARN] No lesion mask for scan_{sid}"
            )
            continue

        pairs.append(
            (
                sid,
                t1_path,
                lesion_by_id[sid]
            )
        )

    print(
        f"[INFO] Matched subjects: "
        f"{len(pairs)}"
    )

    if len(pairs) == 0:
        raise RuntimeError(
            "No matched HD-BET T1 / lesion pairs found."
        )

    return pairs


# ============================================================
# 3. SUBJECT-LEVEL LESION LABEL
# ============================================================

def get_subject_labels(
    subject_pairs
):
    """
    Determine whether each subject has any lesion.

    This label is used ONLY for stratified splitting.
    It is not used as the segmentation target.
    """

    print(
        "\n[STEP] Determining subject-level lesion presence..."
    )

    valid_pairs = []
    labels = []

    for (
        sid,
        t1_path,
        mask_path
    ) in tqdm(
        subject_pairs,
        desc="Reading lesion labels"
    ):

        try:

            mask = nib.load(
                mask_path
            ).get_fdata()

            has_lesion = int(
                np.any(mask > 0)
            )

            valid_pairs.append(
                (
                    sid,
                    t1_path,
                    mask_path
                )
            )

            labels.append(
                has_lesion
            )

        except Exception as exc:

            print(
                f"[WARN] Skipping {sid}: {exc}"
            )

    lesion_count = int(
        sum(labels)
    )

    no_lesion_count = (
        len(labels)
        -
        lesion_count
    )

    print(
        f"[INFO] Usable subjects: "
        f"{len(valid_pairs)}"
    )

    print(
        f"[INFO] Subjects with lesion: "
        f"{lesion_count}"
    )

    print(
        f"[INFO] Subjects without lesion: "
        f"{no_lesion_count}"
    )

    return (
        valid_pairs,
        np.asarray(
            labels,
            dtype=np.int64
        )
    )


# ============================================================
# 4. VOLUME LOADING / NORMALIZATION
# ============================================================

def load_subject_volume(
    t1_path,
    mask_path
):
    """
    Load one complete 3D subject.

    T1:
        HD-BET skull-stripped image

    Mask:
        original lesion ground truth
    """

    t1_nii = nib.load(
        t1_path
    )

    mask_nii = nib.load(
        mask_path
    )

    image = (
        t1_nii
        .get_fdata()
        .astype(np.float32)
    )

    mask = (
        mask_nii
        .get_fdata()
        .astype(np.float32)
    )

    if image.ndim != 3:
        raise ValueError(
            f"T1 is not 3D: {image.shape}"
        )

    if mask.ndim != 3:
        raise ValueError(
            f"Lesion mask is not 3D: {mask.shape}"
        )

    if image.shape != mask.shape:
        raise ValueError(
            "Shape mismatch: "
            f"T1={image.shape}, "
            f"mask={mask.shape}"
        )

    # --------------------------------------------------------
    # Verify spatial metadata.
    # --------------------------------------------------------

    image_zooms = t1_nii.header.get_zooms()[:3]
    mask_zooms = mask_nii.header.get_zooms()[:3]

    if not np.allclose(
        image_zooms,
        mask_zooms,
        atol=1e-3
    ):
        print(
            "[WARN] T1/mask voxel spacing differs:"
        )
        print(
            f"       T1   : {image_zooms}"
        )
        print(
            f"       Mask : {mask_zooms}"
        )

    # --------------------------------------------------------
    # HD-BET brain mask.
    #
    # Because the T1 has already been skull stripped, nonzero
    # voxels provide the effective brain region for intensity
    # normalization.
    # --------------------------------------------------------

    brain_mask = (
        np.abs(image) > 1e-8
    )

    if not np.any(brain_mask):
        raise ValueError(
            "HD-BET image contains no nonzero brain voxels."
        )

    # --------------------------------------------------------
    # Robust MRI normalization:
    #
    # 1. percentile clipping
    # 2. z-score using brain voxels
    # --------------------------------------------------------

    brain_values = image[
        brain_mask
    ]

    low, high = np.percentile(
        brain_values,
        [1, 99]
    )

    image = np.clip(
        image,
        low,
        high
    )

    mean = image[
        brain_mask
    ].mean()

    std = image[
        brain_mask
    ].std()

    image = (
        image - mean
    ) / (
        std + 1e-8
    )

    # Keep background at zero after normalization.
    image[
        ~brain_mask
    ] = 0.0

    mask = (
        mask > 0
    ).astype(np.float32)

    return (
        image,
        mask,
        tuple(
            float(x)
            for x in image_zooms
        )
    )


# ============================================================
# 5. PATCH UTILITIES
# ============================================================

def pad_volume_to_patch_size(
    image,
    mask,
    patch_size
):
    """
    Pad a complete 3D volume if one or more dimensions are
    smaller than the requested patch size.
    """

    pad_width = []

    for current, target in zip(
        image.shape,
        patch_size
    ):

        total_pad = max(
            0,
            target - current
        )

        before = (
            total_pad // 2
        )

        after = (
            total_pad - before
        )

        pad_width.append(
            (
                before,
                after
            )
        )

    if any(
        before > 0 or after > 0
        for before, after in pad_width
    ):

        image = np.pad(
            image,
            pad_width,
            mode="constant",
            constant_values=0
        )

        mask = np.pad(
            mask,
            pad_width,
            mode="constant",
            constant_values=0
        )

    return (
        image,
        mask,
        pad_width
    )


def crop_patch(
    volume,
    start,
    patch_size
):
    """
    Extract a 3D patch.
    """

    x, y, z = start

    px, py, pz = patch_size

    return volume[
        x:x + px,
        y:y + py,
        z:z + pz
    ]


def random_valid_start(
    shape,
    patch_size,
    rng
):
    """
    Randomly select a valid patch start coordinate.
    """

    starts = []

    for size, patch in zip(
        shape,
        patch_size
    ):

        maximum = (
            size - patch
        )

        if maximum <= 0:
            starts.append(0)
        else:
            starts.append(
                int(
                    rng.integers(
                        0,
                        maximum + 1
                    )
                )
            )

    return tuple(starts)


def lesion_centered_start(
    lesion_mask,
    patch_size,
    rng
):
    """
    Pick a lesion voxel and construct a patch around it.
    """

    lesion_coords = np.argwhere(
        lesion_mask > 0
    )

    if len(lesion_coords) == 0:
        return None

    center = lesion_coords[
        rng.integers(
            0,
            len(lesion_coords)
        )
    ]

    starts = []

    for dim in range(3):

        patch = patch_size[dim]
        coord = int(
            center[dim]
        )

        jitter = int(
            patch
            *
            LESION_CENTER_JITTER
        )

        offset = int(
            rng.integers(
                -jitter,
                jitter + 1
            )
        )

        start = (
            coord
            -
            patch // 2
            +
            offset
        )

        max_start = (
            lesion_mask.shape[dim]
            -
            patch
        )

        start = max(
            0,
            min(
                start,
                max_start
            )
        )

        starts.append(
            int(start)
        )

    return tuple(starts)


# ============================================================
# 6. 3D PATCH DATASET
# ============================================================

class PatchDataset(
    Dataset
):
    """
    3D patch dataset with lesion-subject oversampling.

    Training:
      - ~80% of sampled subjects are lesion-positive.
      - Positive subjects receive lesion-centered patches.
      - Negative subjects receive random patches.
      - Sampling is applied only after the train/validation/test split.

    Validation:
      - Every validation subject contributes two patches per epoch.
      - Positive subjects: one lesion-centered + one random patch.
      - Negative subjects: two random patches.
      - Positive-patch Dice and background false-positive rate are
        tracked separately.
    """

    def __init__(
        self,
        subject_pairs,
        patches_per_subject=PATCHES_PER_SUBJECT,
        training=True
    ):
        self.subject_pairs = list(subject_pairs)
        self.patches_per_subject = patches_per_subject
        self.training = training
        self.epoch = 0

        self.lesion_subjects = []
        self.background_subjects = []

        for pair in self.subject_pairs:
            _sid, _t1_path, mask_path = pair

            mask_nii = nib.load(mask_path)
            mask = np.asarray(
                mask_nii.dataobj,
                dtype=np.float32
            )

            if np.any(mask > 0):
                self.lesion_subjects.append(pair)
            else:
                self.background_subjects.append(pair)

        self.length = (
            len(self.subject_pairs)
            *
            self.patches_per_subject
        )

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        # Epoch-dependent seed means a new patch/augmentation is sampled
        # instead of repeating exactly the same patch every epoch.
        rng = np.random.default_rng(
            SEED
            + self.epoch * 100000
            + int(index)
        )

        if self.training:

            choose_lesion = (
                len(self.lesion_subjects) > 0
                and rng.random() < TRAIN_LESION_SUBJECT_PROB
            )

            if choose_lesion:
                pool = self.lesion_subjects
            else:
                pool = (
                    self.background_subjects
                    if len(self.background_subjects) > 0
                    else self.lesion_subjects
                )

            pair = pool[
                int(rng.integers(0, len(pool)))
            ]

        else:
            subject_index = (
                index // self.patches_per_subject
            )

            pair = self.subject_pairs[
                subject_index
            ]

        sid, t1_path, mask_path = pair

        image, mask, _spacing = load_subject_volume(
            t1_path,
            mask_path
        )

        image, mask, _pad_width = pad_volume_to_patch_size(
            image,
            mask,
            PATCH_SIZE
        )

        has_lesion = np.any(mask > 0)

        # Training always uses lesion-centered patches for positive
        # subjects. Validation deliberately alternates between a
        # lesion-focused patch and a random/background patch.
        use_lesion_center = (
            has_lesion
            and (
                self.training
                or (
                    self.patches_per_subject > 1
                    and (index % self.patches_per_subject == 0)
                )
            )
        )

        if use_lesion_center:
            start = lesion_centered_start(
                mask,
                PATCH_SIZE,
                rng
            )

            if start is None:
                start = random_valid_start(
                    image.shape,
                    PATCH_SIZE,
                    rng
                )
        else:
            start = random_valid_start(
                image.shape,
                PATCH_SIZE,
                rng
            )

        image_patch = crop_patch(
            image,
            start,
            PATCH_SIZE
        )

        mask_patch = crop_patch(
            mask,
            start,
            PATCH_SIZE
        )

        if self.training:

            if rng.random() < 0.5:
                image_patch = np.flip(
                    image_patch,
                    axis=0
                ).copy()
                mask_patch = np.flip(
                    mask_patch,
                    axis=0
                ).copy()

            if rng.random() < 0.5:
                image_patch = np.flip(
                    image_patch,
                    axis=1
                ).copy()
                mask_patch = np.flip(
                    mask_patch,
                    axis=1
                ).copy()

            if rng.random() < 0.5:
                image_patch = np.flip(
                    image_patch,
                    axis=2
                ).copy()
                mask_patch = np.flip(
                    mask_patch,
                    axis=2
                ).copy()

        image_tensor = torch.from_numpy(
            image_patch
        ).unsqueeze(0).float()

        mask_tensor = torch.from_numpy(
            mask_patch
        ).unsqueeze(0).float()

        return (
            image_tensor,
            mask_tensor,
            sid
        )


# ============================================================
# 7. 3D U-NET
# ============================================================

class ConvBlock3D(
    nn.Module
):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm3d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm3d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(
        self,
        x
    ):
        return self.block(x)


class UNet3D(
    nn.Module
):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=BASE_CHANNELS
    ):

        super().__init__()

        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2
        c5 = c4 * 2

        self.enc1 = ConvBlock3D(
            in_channels,
            c1
        )

        self.enc2 = ConvBlock3D(
            c1,
            c2
        )

        self.enc3 = ConvBlock3D(
            c2,
            c3
        )

        self.enc4 = ConvBlock3D(
            c3,
            c4
        )

        self.pool = nn.MaxPool3d(
            kernel_size=2,
            stride=2
        )

        self.bottleneck = ConvBlock3D(
            c4,
            c5
        )

        self.up4 = nn.ConvTranspose3d(
            c5,
            c4,
            kernel_size=2,
            stride=2
        )

        self.dec4 = ConvBlock3D(
            c4 + c4,
            c4
        )

        self.up3 = nn.ConvTranspose3d(
            c4,
            c3,
            kernel_size=2,
            stride=2
        )

        self.dec3 = ConvBlock3D(
            c3 + c3,
            c3
        )

        self.up2 = nn.ConvTranspose3d(
            c3,
            c2,
            kernel_size=2,
            stride=2
        )

        self.dec2 = ConvBlock3D(
            c2 + c2,
            c2
        )

        self.up1 = nn.ConvTranspose3d(
            c2,
            c1,
            kernel_size=2,
            stride=2
        )

        self.dec1 = ConvBlock3D(
            c1 + c1,
            c1
        )

        self.final = nn.Conv3d(
            c1,
            out_channels,
            kernel_size=1
        )

    @staticmethod
    def _match_size(
        x,
        reference
    ):

        if x.shape[2:] == reference.shape[2:]:
            return x

        return F.interpolate(
            x,
            size=reference.shape[2:],
            mode="trilinear",
            align_corners=False
        )

    def forward(
        self,
        x
    ):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        b = self.bottleneck(
            self.pool(e4)
        )

        d4 = self.up4(b)
        d4 = self._match_size(
            d4,
            e4
        )
        d4 = torch.cat(
            [d4, e4],
            dim=1
        )
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self._match_size(
            d3,
            e3
        )
        d3 = torch.cat(
            [d3, e3],
            dim=1
        )
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._match_size(
            d2,
            e2
        )
        d2 = torch.cat(
            [d2, e2],
            dim=1
        )
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._match_size(
            d1,
            e1
        )
        d1 = torch.cat(
            [d1, e1],
            dim=1
        )
        d1 = self.dec1(d1)

        return self.final(d1)


# ============================================================
# 8. TVERSKY LOSS
# ============================================================

class TverskyLoss(
    nn.Module
):

    def __init__(
        self,
        alpha=TVERSKY_ALPHA,
        beta=TVERSKY_BETA,
        smooth=SMOOTH
    ):

        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(
        self,
        logits,
        targets
    ):

        probabilities = torch.sigmoid(
            logits
        )

        probabilities = probabilities.reshape(
            probabilities.shape[0],
            -1
        )

        targets = targets.reshape(
            targets.shape[0],
            -1
        )

        true_positive = (
            probabilities
            *
            targets
        ).sum(dim=1)

        false_positive = (
            probabilities
            *
            (1.0 - targets)
        ).sum(dim=1)

        false_negative = (
            (1.0 - probabilities)
            *
            targets
        ).sum(dim=1)

        tversky = (
            true_positive
            +
            self.smooth
        ) / (
            true_positive
            +
            self.alpha * false_positive
            +
            self.beta * false_negative
            +
            self.smooth
        )

        return (
            1.0
            -
            tversky.mean()
        )


# ============================================================
# 9. BOUNDARY LOSS
# ============================================================

class BoundaryLoss(
    nn.Module
):
    """
    Distance-transform-based boundary loss.

    The ground-truth boundary is represented using a signed
    distance map.

    Predictions close to the correct boundary receive lower
    penalty than predictions far from it.

    The distance map is constructed from the target mask, so
    this loss is used only during training.
    """

    def __init__(
        self
    ):

        super().__init__()

    @staticmethod
    def signed_distance(
        mask
    ):
        """
        Create signed distance map.

        Positive inside lesion.
        Negative outside lesion.
        """

        mask_np = (
            mask
            .detach()
            .cpu()
            .numpy()
        )

        distance_maps = []

        for sample in mask_np:

            sample = (
                sample[0] > 0.5
            )

            if not np.any(sample):

                distance_maps.append(
                    np.zeros_like(
                        sample,
                        dtype=np.float32
                    )
                )

                continue

            inside = distance_transform_edt(
                sample
            )

            outside = distance_transform_edt(
                ~sample
            )

            signed = (
                outside
                -
                inside
            ).astype(
                np.float32
            )

            distance_maps.append(
                signed
            )

        result = np.stack(
            distance_maps,
            axis=0
        )

        return torch.from_numpy(
            result
        ).unsqueeze(1)

    def forward(
        self,
        logits,
        targets
    ):

        probabilities = torch.sigmoid(
            logits
        )

        signed_distance = self.signed_distance(
            targets
        ).to(
            logits.device,
            dtype=logits.dtype
        )

        # Normalize the distance map so the boundary term does
        # not become excessively large for large volumes.
        max_distance = (
            signed_distance.abs()
            .flatten(1)
            .amax(dim=1)
            .view(-1, 1, 1, 1, 1)
        )

        signed_distance = (
            signed_distance
            /
            (max_distance + 1e-6)
        )

        # The boundary term compares prediction error with the
        # signed distance to the true boundary.
        boundary_error = (
            torch.abs(
                probabilities
                -
                targets
            )
            *
            signed_distance.abs()
        )

        return boundary_error.mean()


# ============================================================
# 10. COMBINED LOSS
# ============================================================

class CombinedLoss(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

        self.tversky = TverskyLoss(
            alpha=TVERSKY_ALPHA,
            beta=TVERSKY_BETA
        )

        self.boundary = BoundaryLoss()

    def forward(
        self,
        logits,
        targets
    ):

        bce_loss = self.bce(
            logits,
            targets
        )

        tversky_loss = self.tversky(
            logits,
            targets
        )

        boundary_loss = self.boundary(
            logits,
            targets
        )

        total = (
            BCE_WEIGHT * bce_loss
            +
            TVERSKY_WEIGHT * tversky_loss
            +
            BOUNDARY_WEIGHT * boundary_loss
        )

        components = {
            "bce": float(
                bce_loss.detach().cpu()
            ),
            "tversky": float(
                tversky_loss.detach().cpu()
            ),
            "boundary": float(
                boundary_loss.detach().cpu()
            ),
            "total": float(
                total.detach().cpu()
            )
        }

        return (
            total,
            components
        )

# ============================================================
# 11. DICE
# ============================================================

def dice_score_from_logits(
    logits,
    targets
):
    """
    Batch Dice using thresholded predictions.
    """

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities > 0.5
    ).float()

    predictions = predictions.reshape(
        predictions.shape[0],
        -1
    )

    targets = targets.reshape(
        targets.shape[0],
        -1
    )

    intersection = (
        predictions
        *
        targets
    ).sum(dim=1)

    denominator = (
        predictions.sum(dim=1)
        +
        targets.sum(dim=1)
    )

    dice = (
        2.0 * intersection
        +
        SMOOTH
    ) / (
        denominator
        +
        SMOOTH
    )

    # Both empty = perfect overlap.
    both_empty = (
        denominator == 0
    )

    dice = torch.where(
        both_empty,
        torch.ones_like(dice),
        dice
    )

    return float(
        dice.mean()
        .detach()
        .cpu()
    )


# ============================================================
# 12. PATCH GRID FOR FULL-VOLUME INFERENCE
# ============================================================

def compute_patch_starts(
    length,
    patch,
    overlap
):
    """
    Generate overlapping patch starts that cover the complete
    dimension, including the end of the volume.
    """

    if length <= patch:
        return [0]

    stride = max(
        1,
        int(
            patch
            *
            (1.0 - overlap)
        )
    )

    starts = list(
        range(
            0,
            length - patch + 1,
            stride
        )
    )

    final_start = (
        length - patch
    )

    if starts[-1] != final_start:
        starts.append(
            final_start
        )

    return starts


def get_patch_grid(
    shape,
    patch_size,
    overlap
):
    """
    Return every patch start required to cover a full 3D
    volume.
    """

    xs = compute_patch_starts(
        shape[0],
        patch_size[0],
        overlap
    )

    ys = compute_patch_starts(
        shape[1],
        patch_size[1],
        overlap
    )

    zs = compute_patch_starts(
        shape[2],
        patch_size[2],
        overlap
    )

    return [
        (x, y, z)
        for x in xs
        for y in ys
        for z in zs
    ]


# ============================================================
# 13. FULL-VOLUME INFERENCE
# ============================================================

@torch.no_grad()
def predict_full_volume(
    model,
    image,
    patch_size=PATCH_SIZE,
    overlap=INFERENCE_OVERLAP
):
    """
    Run sliding-window inference over the complete 3D volume.

    Overlapping predictions are averaged.
    """

    model.eval()

    original_shape = image.shape

    pad_width = []

    for current, target in zip(
        image.shape,
        patch_size
    ):

        total_pad = max(
            0,
            target - current
        )

        before = (
            total_pad // 2
        )

        after = (
            total_pad - before
        )

        pad_width.append(
            (
                before,
                after
            )
        )

    padded = np.pad(
        image,
        pad_width,
        mode="constant",
        constant_values=0
    )

    prediction_sum = np.zeros(
        padded.shape,
        dtype=np.float32
    )

    prediction_count = np.zeros(
        padded.shape,
        dtype=np.float32
    )

    starts = get_patch_grid(
        padded.shape,
        patch_size,
        overlap
    )

    for start in starts:

        patch = crop_patch(
            padded,
            start,
            patch_size
        )

        patch_tensor = (
            torch.from_numpy(
                patch
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(DEVICE)
        )

        logits = model(
            patch_tensor
        )

        probabilities = torch.sigmoid(
            logits
        )

        probabilities = (
            probabilities
            .squeeze()
            .cpu()
            .numpy()
        )

        x, y, z = start

        px, py, pz = patch_size

        prediction_sum[
            x:x + px,
            y:y + py,
            z:z + pz
        ] += probabilities

        prediction_count[
            x:x + px,
            y:y + py,
            z:z + pz
        ] += 1.0

    probability_volume = (
        prediction_sum
        /
        np.maximum(
            prediction_count,
            1e-8
        )
    )

    # Remove padding.
    slices = []

    for dim in range(3):

        before = pad_width[dim][0]
        length = original_shape[dim]

        slices.append(
            slice(
                before,
                before + length
            )
        )

    probability_volume = (
        probability_volume[
            tuple(slices)
        ]
    )

    prediction = (
        probability_volume > 0.5
    ).astype(
        np.uint8
    )

    return (
        probability_volume,
        prediction
    )


# ============================================================
# 14. HAUSDORFF DISTANCE
# ============================================================

def hausdorff_distance_mm(
    prediction,
    target,
    spacing
):
    """
    Symmetric Hausdorff distance in millimetres.

    This is intentionally used only during final evaluation,
    not every training epoch.
    """

    pred_points = np.argwhere(
        prediction > 0
    )

    target_points = np.argwhere(
        target > 0
    )

    # Both empty.
    if (
        len(pred_points) == 0
        and
        len(target_points) == 0
    ):
        return 0.0

    # One empty and one non-empty.
    if (
        len(pred_points) == 0
        or
        len(target_points) == 0
    ):

        image_shape = np.asarray(
            prediction.shape
        )

        spacing_array = np.asarray(
            spacing
        )

        diagonal = np.sqrt(
            np.sum(
                (
                    image_shape
                    *
                    spacing_array
                ) ** 2
            )
        )

        return float(
            diagonal
        )

    spacing_array = np.asarray(
        spacing
    )

    pred_points = (
        pred_points
        *
        spacing_array
    )

    target_points = (
        target_points
        *
        spacing_array
    )

    forward = directed_hausdorff(
        pred_points,
        target_points
    )[0]

    backward = directed_hausdorff(
        target_points,
        pred_points
    )[0]

    return float(
        max(
            forward,
            backward
        )
    )


# ============================================================
# 15. VALIDATION LOSS / DICE
# ============================================================

def validate_patches(
    model,
    loader,
    criterion
):
    """
    Validation using both lesion-focused and random/background patches.

    Returns:
        validation_loss,
        positive_soft_dice,
        positive_hard_dice,
        background_fp_rate,
        validation_score

    Positive-patch Dice measures lesion overlap.
    Background FP rate measures how much foreground the model predicts
    on patches whose ground truth contains no lesion.

    The balanced validation score is:

        positive_soft_dice * (1 - background_fp_rate)

    This rewards lesion overlap while directly penalizing
    massive foreground predictions on background patches.
    """

    model.eval()

    total_loss = 0.0
    positive_soft_dice_sum = 0.0
    positive_hard_dice_sum = 0.0

    positive_patch_count = 0

    background_false_positive_voxels = 0
    background_total_voxels = 0

    batch_count = 0

    with torch.no_grad():

        for images, masks, _subject_ids in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            masks = masks.to(
                DEVICE,
                non_blocking=True
            )

            logits = model(images)

            loss, _components = criterion(
                logits,
                masks
            )

            total_loss += float(
                loss.detach().cpu()
            )

            probabilities = torch.sigmoid(logits)

            probs_flat = probabilities.reshape(
                probabilities.shape[0],
                -1
            )

            targets_flat = masks.reshape(
                masks.shape[0],
                -1
            )

            positive_targets = (
                targets_flat.sum(dim=1) > 0
            )

            negative_targets = ~positive_targets

            # ------------------------------------------------
            # LESION-POSITIVE PATCHES
            # ------------------------------------------------

            if torch.any(positive_targets):

                p = probs_flat[
                    positive_targets
                ]

                t = targets_flat[
                    positive_targets
                ]

                soft_intersection = (
                    p * t
                ).sum(dim=1)

                soft_denominator = (
                    p.sum(dim=1)
                    +
                    t.sum(dim=1)
                )

                soft_dice = (
                    2.0 * soft_intersection
                    +
                    SMOOTH
                ) / (
                    soft_denominator
                    +
                    SMOOTH
                )

                hard_predictions = (
                    p > 0.5
                ).float()

                hard_intersection = (
                    hard_predictions * t
                ).sum(dim=1)

                hard_denominator = (
                    hard_predictions.sum(dim=1)
                    +
                    t.sum(dim=1)
                )

                hard_dice = (
                    2.0 * hard_intersection
                    +
                    SMOOTH
                ) / (
                    hard_denominator
                    +
                    SMOOTH
                )

                positive_soft_dice_sum += float(
                    soft_dice.sum().detach().cpu()
                )

                positive_hard_dice_sum += float(
                    hard_dice.sum().detach().cpu()
                )

                positive_patch_count += int(
                    positive_targets.sum().item()
                )

            # ------------------------------------------------
            # BACKGROUND PATCHES
            # ------------------------------------------------

            if torch.any(negative_targets):

                p_bg = probs_flat[
                    negative_targets
                ]

                # Fraction of voxels predicted as foreground.
                bg_predicted_foreground = (
                    p_bg > 0.5
                ).float().sum()

                background_false_positive_voxels += int(
                    bg_predicted_foreground
                    .detach()
                    .cpu()
                )

                background_total_voxels += int(
                    p_bg.numel()
                )

            batch_count += 1

    if batch_count == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan")
        )

    validation_loss = (
        total_loss / batch_count
    )

    if positive_patch_count > 0:
        positive_soft_dice = (
            positive_soft_dice_sum
            /
            positive_patch_count
        )

        positive_hard_dice = (
            positive_hard_dice_sum
            /
            positive_patch_count
        )
    else:
        positive_soft_dice = float("nan")
        positive_hard_dice = float("nan")

    if background_total_voxels > 0:
        background_fp_rate = (
            background_false_positive_voxels
            /
            background_total_voxels
        )
    else:
        background_fp_rate = 0.0

    if np.isfinite(positive_soft_dice):
        validation_score = (
            positive_soft_dice
            *
            max(
                0.0,
                1.0 - background_fp_rate
            )
        )
    else:
        validation_score = float("nan")

    return (
        validation_loss,
        positive_soft_dice,
        positive_hard_dice,
        background_fp_rate,
        validation_score
    )

# ============================================================
# 16. FULL-VOLUME VALIDATION FOR FINAL FOLD SELECTION
# ============================================================

@torch.no_grad()
def evaluate_subjects_full_volume(
    model,
    subject_pairs,
    save_predictions=False,
    prediction_prefix="validation"
):
    """
    Full 3D evaluation.

    Dice is calculated on reconstructed full volumes.

    Hausdorff is also calculated here, but this function should
    NOT be called every epoch.
    """

    model.eval()

    dice_scores = []
    hd_scores = []

    subject_results = []

    for (
        sid,
        t1_path,
        mask_path
    ) in tqdm(
        subject_pairs,
        desc="Full-volume evaluation"
    ):

        image, target, spacing = (
            load_subject_volume(
                t1_path,
                mask_path
            )
        )

        probability, prediction = (
            predict_full_volume(
                model,
                image
            )
        )

        intersection = np.logical_and(
            prediction > 0,
            target > 0
        ).sum()

        denominator = (
            (prediction > 0).sum()
            +
            (target > 0).sum()
        )

        if denominator == 0:

            dice = 1.0

        else:

            dice = (
                2.0 * intersection
                /
                denominator
            )

        hd = hausdorff_distance_mm(
            prediction,
            target,
            spacing
        )

        dice_scores.append(
            float(dice)
        )

        hd_scores.append(
            float(hd)
        )

        subject_results.append(
            {
                "subject_id": sid,
                "dice": float(dice),
                "hausdorff_mm": float(hd)
            }
        )

        if save_predictions:

            output_path = os.path.join(
                PREDICTION_DIR,
                f"{prediction_prefix}_"
                f"scan_{sid}_Lesion_pred.nii.gz"
            )

            original_mask_nii = nib.load(
                mask_path
            )

            prediction_nii = nib.Nifti1Image(
                prediction.astype(
                    np.uint8
                ),
                affine=original_mask_nii.affine,
                header=original_mask_nii.header
            )

            nib.save(
                prediction_nii,
                output_path
            )

    return {
        "mean_dice": float(
            np.mean(dice_scores)
        )
        if dice_scores
        else float("nan"),

        "std_dice": float(
            np.std(dice_scores)
        )
        if dice_scores
        else float("nan"),

        "mean_hausdorff_mm": float(
            np.mean(hd_scores)
        )
        if hd_scores
        else float("nan"),

        "std_hausdorff_mm": float(
            np.std(hd_scores)
        )
        if hd_scores
        else float("nan"),

        "subjects": subject_results
    }


# ============================================================
# 17. TRAIN ONE FOLD
# ============================================================

def train_one_fold(
    fold_number,
    train_subjects,
    val_subjects
):
    """
    Train one CV fold.

    Returns:
        best model path
        best validation Dice
        epoch history
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        f"[FOLD {fold_number}/{NUM_FOLDS}]"
    )

    print(
        "=" * 75
    )

    print(
        f"[INFO] Train subjects: "
        f"{len(train_subjects)}"
    )

    print(
        f"[INFO] Validation subjects: "
        f"{len(val_subjects)}"
    )

    train_dataset = PatchDataset(
        train_subjects,
        patches_per_subject=PATCHES_PER_SUBJECT,
        training=True
    )

    val_dataset = PatchDataset(
        val_subjects,
        patches_per_subject=VAL_PATCHES_PER_SUBJECT,
        training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )

    model = UNet3D().to(
        DEVICE
    )

    criterion = CombinedLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=LR_PATIENCE,
            factor=LR_FACTOR
        )
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=USE_AMP
    )

    best_val_score = -float("inf")

    best_model_path = os.path.join(
        MODEL_DIR,
        f"fold_{fold_number}_best.pth"
    )

    history = []

    epochs_without_improvement = 0

    # --------------------------------------------------------
    # EPOCH LOOP
    # --------------------------------------------------------

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        train_dataset.set_epoch(epoch)
        val_dataset.set_epoch(epoch)

        model.train()

        running_loss = 0.0
        running_bce = 0.0
        running_tversky = 0.0
        running_boundary = 0.0

        iteration_count = 0

        print(
            f"\n[FOLD {fold_number}] "
            f"Epoch {epoch}/{NUM_EPOCHS}"
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        for iteration, (
            images,
            masks,
            _subject_ids
        ) in enumerate(
            train_loader,
            start=1
        ):

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            masks = masks.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type=DEVICE.type,
                enabled=USE_AMP
            ):

                logits = model(
                    images
                )

                loss, components = (
                    criterion(
                        logits,
                        masks
                    )
                )

            scaler.scale(
                loss
            ).backward()

            # ------------------------------------------------
            # Gradient diagnostics.
            # ------------------------------------------------

            scaler.unscale_(
                optimizer
            )

            gradient_norm = (
                torch.nn.utils
                .clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            running_loss += (
                components["total"]
            )

            running_bce += (
                components["bce"]
            )

            running_tversky += (
                components["tversky"]
            )

            running_boundary += (
                components["boundary"]
            )

            iteration_count += 1

            if (
                PRINT_EVERY_ITER > 0
                and
                iteration
                %
                PRINT_EVERY_ITER
                == 0
            ):

                print(
                    f"  Iteration "
                    f"{iteration}/"
                    f"{len(train_loader)} | "
                    f"Loss={components['total']:.5f} | "
                    f"Tversky={components['tversky']:.5f} | "
                    f"Boundary={components['boundary']:.5f} | "
                    f"GradNorm={float(gradient_norm):.4f}"
                )

        train_loss = (
            running_loss
            /
            max(
                iteration_count,
                1
            )
        )


        train_bce = (
            running_bce
            /
            max(
                iteration_count,
                1
            )
        )

        train_tversky = (
            running_tversky
            /
            max(
                iteration_count,
                1
            )
        )

        train_boundary = (
            running_boundary
            /
            max(
                iteration_count,
                1
            )
        )

        # ----------------------------------------------------
        # VALIDATION
        #
        # Patch-level validation Dice/loss is intentionally
        # used here because full-volume inference every epoch
        # would be expensive.
        # ----------------------------------------------------

        (
            val_loss,
            val_soft_dice,
            val_dice,
            val_background_fp_rate,
            val_score
        ) = validate_patches(
            model,
            val_loader,
            criterion
        )

        # Balanced validation score rewards lesion overlap while
        # penalizing foreground predictions on true-background patches.
        scheduler.step(
            val_score
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        print(
            f"\n[FOLD {fold_number}] "
            f"Epoch {epoch} SUMMARY"
        )

        print(
            f"  Train Loss       : "
            f"{train_loss:.6f}"
        )

        print(
            f"  Train BCE        : "
            f"{train_bce:.6f}"
        )

        print(
            f"  Train Tversky    : "
            f"{train_tversky:.6f}"
        )

        print(
            f"  Train Boundary   : "
            f"{train_boundary:.6f}"
        )

        print(
            f"  Validation Loss  : "
            f"{val_loss:.6f}"
        )

        print(
            f"  Soft Dice Accuracy   : "
            f"{val_soft_dice:.6f}"
        )

        print(
            f"  Hard Dice @ 0.5      : "
            f"{val_dice:.6f}"
        )

        print(
            f"  Background FP Rate   : "
            f"{val_background_fp_rate:.6f}"
        )

        print(
            f"  Balanced Val Score   : "
            f"{val_score:.6f}"
        )

        print(
            f"  Learning Rate    : "
            f"{current_lr:.2e}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_bce": train_bce,
                "train_tversky": train_tversky,
                "train_boundary": train_boundary,
                "val_loss": val_loss,
                "val_soft_dice": val_soft_dice,
                "val_dice": val_dice,
                "val_background_fp_rate": val_background_fp_rate,
                "val_score": val_score,
                "learning_rate": current_lr
            }
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if val_score > best_val_score:

            best_val_score = val_score

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "fold":
                        fold_number,

                    "epoch":
                        epoch,

                    "val_soft_dice":
                        val_soft_dice,

                    "val_dice":
                        val_dice,

                    "val_background_fp_rate":
                        val_background_fp_rate,

                    "val_score":
                        val_score,

                    "config":
                        {
                            "patch_size":
                                PATCH_SIZE,

                            "base_channels":
                                BASE_CHANNELS,

                            "tversky_alpha":
                                TVERSKY_ALPHA,

                            "tversky_beta":
                                TVERSKY_BETA,

                            "bce_weight":
                                BCE_WEIGHT,

                            "boundary_weight":
                                BOUNDARY_WEIGHT,

                            "validation_patches_per_subject":
                                VAL_PATCHES_PER_SUBJECT
                        }
                },
                best_model_path
            )

            print(
                f"  [BEST MODEL SAVED] "
                f"Score={val_score:.6f} "
                f"SoftDice={val_soft_dice:.6f} "
                f"HardDice={val_dice:.6f} "
                f"BG-FP={val_background_fp_rate:.6f}"
            )

        else:

            epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= EARLY_STOP_PATIENCE
            ):

                print(
                    f"\n[FOLD {fold_number}] "
                    f"Early stopping after "
                    f"{EARLY_STOP_PATIENCE} "
                    f"epochs without improvement."
                )

                break

    # --------------------------------------------------------
    # Save history.
    # --------------------------------------------------------

    history_path = os.path.join(
        LOG_DIR,
        f"fold_{fold_number}_history.csv"
    )

    pd.DataFrame(
        history
    ).to_csv(
        history_path,
        index=False
    )

    print(
        f"[INFO] Saved training history: "
        f"{history_path}"
    )

    print(
        f"[INFO] Best balanced validation score: "
        f"{best_val_score:.6f}"
    )

    return (
        best_model_path,
        best_val_score
    )


# ============================================================
# 18. MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 75
    )

    print(
        "AIMS-TBI 2026 - 3D PATCH-BASED SEGMENTATION"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # FIND SUBJECTS
    # --------------------------------------------------------

    subject_pairs = get_subject_pairs()

    subject_pairs, labels = (
        get_subject_labels(
            subject_pairs
        )
    )

    # --------------------------------------------------------
    # 80 / 20 SUBJECT-LEVEL SPLIT
    #
    # The 20% test set is NEVER passed to CV.
    # --------------------------------------------------------

    indices = np.arange(
        len(subject_pairs)
    )

    train_indices, test_indices = (
        train_test_split(
            indices,
            test_size=TEST_SIZE,
            random_state=SEED,
            shuffle=True,
            stratify=labels
        )
    )

    train_subjects = [
        subject_pairs[i]
        for i in train_indices
    ]

    test_subjects = [
        subject_pairs[i]
        for i in test_indices
    ]

    train_labels = labels[
        train_indices
    ]

    test_labels = labels[
        test_indices
    ]

    print(
        "\n[DATA SPLIT]"
    )

    print(
        f"Training subjects: "
        f"{len(train_subjects)}"
    )

    print(
        f"Testing subjects : "
        f"{len(test_subjects)}"
    )

    print(
        f"Train lesion subjects: "
        f"{int(train_labels.sum())}"
    )

    print(
        f"Train no-lesion subjects: "
        f"{len(train_labels) - int(train_labels.sum())}"
    )

    print(
        f"Test lesion subjects: "
        f"{int(test_labels.sum())}"
    )

    print(
        f"Test no-lesion subjects: "
        f"{len(test_labels) - int(test_labels.sum())}"
    )

    # Save split IDs for reproducibility.
    split_rows = []

    for sid, _, _ in train_subjects:

        split_rows.append(
            {
                "subject_id": sid,
                "split": "train_cv"
            }
        )

    for sid, _, _ in test_subjects:

        split_rows.append(
            {
                "subject_id": sid,
                "split": "test"
            }
        )

    pd.DataFrame(
        split_rows
    ).to_csv(
        os.path.join(
            RESULT_DIR,
            "subject_split.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # 5-FOLD STRATIFIED CV ON TRAINING SET ONLY
    # --------------------------------------------------------

    skf = StratifiedKFold(
        n_splits=NUM_FOLDS,
        shuffle=True,
        random_state=SEED
    )

    fold_results = []

    best_fold_model = None
    best_fold_dice = -float("inf")
    best_fold_number = None

    print(
        "\n"
        + "=" * 75
    )

    print(
        "STARTING 5-FOLD CROSS VALIDATION"
    )

    print(
        "=" * 75
    )

    for fold_number, (
        fold_train_idx,
        fold_val_idx
    ) in enumerate(
        skf.split(
            train_subjects,
            train_labels
        ),
        start=1
    ):

        fold_train_subjects = [
            train_subjects[i]
            for i in fold_train_idx
        ]

        fold_val_subjects = [
            train_subjects[i]
            for i in fold_val_idx
        ]

        model_path, val_dice = (
            train_one_fold(
                fold_number,
                fold_train_subjects,
                fold_val_subjects
            )
        )

        fold_results.append(
            {
                "fold":
                    fold_number,

                "validation_dice":
                    float(val_dice),

                "model_path":
                    model_path
            }
        )

        if val_dice > best_fold_dice:

            best_fold_dice = val_dice

            best_fold_model = model_path

            best_fold_number = fold_number

    # --------------------------------------------------------
    # SAVE CV SUMMARY
    # --------------------------------------------------------

    cv_dice_values = [
        item["validation_dice"]
        for item in fold_results
    ]

    cv_summary = {
        "folds":
            fold_results,

        "mean_validation_dice":
            float(
                np.mean(
                    cv_dice_values
                )
            ),

        "std_validation_dice":
            float(
                np.std(
                    cv_dice_values
                )
            ),

        "best_fold":
            best_fold_number,

        "best_fold_validation_dice":
            float(best_fold_dice)
    }

    with open(
        os.path.join(
            RESULT_DIR,
            "cv_summary.json"
        ),
        "w"
    ) as file:

        json.dump(
            cv_summary,
            file,
            indent=4
        )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "CROSS-VALIDATION SUMMARY"
    )

    print(
        "=" * 75
    )

    for result in fold_results:

        print(
            f"Fold {result['fold']}: "
            f"Dice={result['validation_dice']:.6f}"
        )

    print(
        f"Mean CV Dice: "
        f"{cv_summary['mean_validation_dice']:.6f}"
    )

    print(
        f"Std CV Dice: "
        f"{cv_summary['std_validation_dice']:.6f}"
    )

    print(
        f"Selected fold: "
        f"{best_fold_number}"
    )

    # --------------------------------------------------------
    # LOAD BEST CV MODEL
    #
    # The test set has still never been used.
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 75
    )

    print(
        "FINAL TEST EVALUATION"
    )

    print(
        "=" * 75
    )

    print(
        f"[INFO] Loading best CV model "
        f"from fold {best_fold_number}"
    )

    checkpoint = torch.load(
        best_fold_model,
        map_location=DEVICE
    )

    final_model = UNet3D().to(
        DEVICE
    )

    final_model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # --------------------------------------------------------
    # FINAL TEST
    #
    # Full-volume 3D inference.
    # Dice + Hausdorff.
    # Predictions saved as NIfTI.
    # --------------------------------------------------------

    test_results = (
        evaluate_subjects_full_volume(
            final_model,
            test_subjects,
            save_predictions=True,
            prediction_prefix="test"
        )
    )

    test_summary = {
        "selected_fold":
            best_fold_number,

        "cv_mean_dice":
            cv_summary[
                "mean_validation_dice"
            ],

        "cv_std_dice":
            cv_summary[
                "std_validation_dice"
            ],

        "test_mean_dice":
            test_results[
                "mean_dice"
            ],

        "test_std_dice":
            test_results[
                "std_dice"
            ],

        "test_mean_hausdorff_mm":
            test_results[
                "mean_hausdorff_mm"
            ],

        "test_std_hausdorff_mm":
            test_results[
                "std_hausdorff_mm"
            ],

        "test_subjects":
            test_results[
                "subjects"
            ]
    }

    with open(
        os.path.join(
            RESULT_DIR,
            "test_results.json"
        ),
        "w"
    ) as file:

        json.dump(
            test_summary,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # SAVE FINAL MODEL
    # --------------------------------------------------------

    final_model_path = os.path.join(
        MODEL_DIR,
        "final_model.pth"
    )

    torch.save(
        {
            "model_state_dict":
                final_model.state_dict(),

            "selected_fold":
                best_fold_number,

            "cv_validation_dice":
                best_fold_dice,

            "test_results":
                test_summary,

            "config":
                {
                    "patch_size":
                        PATCH_SIZE,

                    "base_channels":
                        BASE_CHANNELS,

                    "batch_size":
                        BATCH_SIZE,

                    "learning_rate":
                        LEARNING_RATE,

                    "tversky_alpha":
                        TVERSKY_ALPHA,

                    "tversky_beta":
                        TVERSKY_BETA,

                    "boundary_weight":
                        BOUNDARY_WEIGHT
                }
        },
        final_model_path
    )

    # --------------------------------------------------------
    # FINAL PRINT
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 75
    )

    print(
        "FINAL TEST RESULTS"
    )

    print(
        "=" * 75
    )

    print(
        f"Test Dice: "
        f"{test_results['mean_dice']:.6f} "
        f"+/- "
        f"{test_results['std_dice']:.6f}"
    )

    print(
        f"Test Hausdorff: "
        f"{test_results['mean_hausdorff_mm']:.4f} mm "
        f"+/- "
        f"{test_results['std_hausdorff_mm']:.4f} mm"
    )

    print(
        f"\n[INFO] Final model: "
        f"{final_model_path}"
    )

    print(
        f"[INFO] Predictions: "
        f"{PREDICTION_DIR}"
    )

    print(
        f"[INFO] Results: "
        f"{RESULT_DIR}"
    )

    print(
        "\n[DONE]"
    )


if __name__ == "__main__":
    main()
