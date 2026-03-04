from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image


DEFAULT_DATA_ROOT = Path("data/AATTCT-IDS")
DEFAULT_OUTPUT_DIR = Path("data/aattct_visceral_subset")


@dataclass(frozen=True)
class PairRecord:
    patient_id: str
    image_source: Path
    label_source: Path
    image_dest: Path
    label_dest: Path
    image_height: int
    image_width: int
    visceral_pixels: int
    visceral_ratio: float


def iter_patient_dirs(path: Path) -> Iterable[Path]:
    for child in sorted(path.iterdir()):
        if child.is_dir() and child.name.startswith("patient_"):
            yield child


def collect_valid_pairs(patient_image_dir: Path, patient_label_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []

    for image_path in sorted(patient_image_dir.glob("*.jpg")):
        label_path = patient_label_dir / f"{image_path.stem}.png"
        if label_path.exists():
            pairs.append((image_path, label_path))

    return pairs


def compute_visceral_metrics(label_path: Path) -> tuple[int, int, int, float]:
    label_arr = np.array(Image.open(label_path))

    if label_arr.ndim == 3:
        mask = label_arr[..., 0] > 0
        height, width = label_arr.shape[:2]
    else:
        mask = label_arr > 0
        height, width = label_arr.shape

    visceral_pixels = int(mask.sum())
    total_pixels = int(height * width)
    visceral_ratio = float(visceral_pixels / total_pixels) if total_pixels else 0.0

    return height, width, visceral_pixels, visceral_ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a paired subset from AATTCT-IDS Extracted images and Visceral labels. "
            "Samples up to N image-label pairs per patient and computes normalized visceral area."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root path containing AATTCT-IDS/Image and AATTCT-IDS/Label.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for subset files and manifest.csv.",
    )
    parser.add_argument(
        "--samples-per-patient",
        type=int,
        default=4,
        help="Maximum number of matched image-label pairs to sample per patient.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible patient-level sampling.",
    )
    parser.add_argument(
        "--copy-files",
        action="store_true",
        default=True,
        help="Copy sampled image and label files into output-dir (default: enabled).",
    )
    parser.add_argument(
        "--no-copy-files",
        action="store_false",
        dest="copy_files",
        help="Only write manifest.csv using source paths; do not copy files.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    images_root = data_root / "Image" / "Extracted"
    labels_root = data_root / "Label" / "Visceral"

    if not images_root.exists() or not labels_root.exists():
        raise FileNotFoundError(
            f"Could not find required folders:\n  {images_root}\n  {labels_root}"
        )

    rng = random.Random(args.seed)

    sampled_records: List[PairRecord] = []
    skipped_patients: list[tuple[str, int]] = []

    if args.copy_files:
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels").mkdir(parents=True, exist_ok=True)

    for patient_image_dir in iter_patient_dirs(images_root):
        patient_id = patient_image_dir.name
        patient_label_dir = labels_root / patient_id

        if not patient_label_dir.exists():
            skipped_patients.append((patient_id, 0))
            continue

        valid_pairs = collect_valid_pairs(patient_image_dir, patient_label_dir)
        if not valid_pairs:
            skipped_patients.append((patient_id, 0))
            continue

        sample_size = min(args.samples_per_patient, len(valid_pairs))
        selected_pairs = rng.sample(valid_pairs, sample_size)

        if len(valid_pairs) < args.samples_per_patient:
            skipped_patients.append((patient_id, len(valid_pairs)))

        for image_src, label_src in selected_pairs:
            if args.copy_files:
                image_dest_dir = output_dir / "images" / patient_id
                label_dest_dir = output_dir / "labels" / patient_id
                image_dest_dir.mkdir(parents=True, exist_ok=True)
                label_dest_dir.mkdir(parents=True, exist_ok=True)

                image_dest = image_dest_dir / image_src.name
                label_dest = label_dest_dir / label_src.name

                if not image_dest.exists():
                    shutil.copy2(image_src, image_dest)
                if not label_dest.exists():
                    shutil.copy2(label_src, label_dest)
            else:
                image_dest = image_src
                label_dest = label_src

            height, width, visceral_pixels, visceral_ratio = compute_visceral_metrics(label_src)
            sampled_records.append(
                PairRecord(
                    patient_id=patient_id,
                    image_source=image_src,
                    label_source=label_src,
                    image_dest=image_dest,
                    label_dest=label_dest,
                    image_height=height,
                    image_width=width,
                    visceral_pixels=visceral_pixels,
                    visceral_ratio=visceral_ratio,
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "patient_id",
                "image_path",
                "label_path",
                "image_height",
                "image_width",
                "visceral_pixels",
                "visceral_ratio",
                "image_source",
                "label_source",
            ],
        )
        writer.writeheader()

        for row in sampled_records:
            writer.writerow(
                {
                    "patient_id": row.patient_id,
                    "image_path": str(row.image_dest.relative_to(output_dir))
                    if row.image_dest.is_relative_to(output_dir)
                    else str(row.image_dest),
                    "label_path": str(row.label_dest.relative_to(output_dir))
                    if row.label_dest.is_relative_to(output_dir)
                    else str(row.label_dest),
                    "image_height": row.image_height,
                    "image_width": row.image_width,
                    "visceral_pixels": row.visceral_pixels,
                    "visceral_ratio": f"{row.visceral_ratio:.8f}",
                    "image_source": str(row.image_source),
                    "label_source": str(row.label_source),
                }
            )

    patient_count = len({record.patient_id for record in sampled_records})
    print(f"Saved {len(sampled_records)} matched pairs across {patient_count} patients")
    print(f"Manifest: {manifest_path}")

    if skipped_patients:
        print("Patients with fewer than requested matched pairs (or missing labels):")
        for patient_id, available in skipped_patients:
            print(f"  {patient_id}: available_pairs={available}")


if __name__ == "__main__":
    main()
