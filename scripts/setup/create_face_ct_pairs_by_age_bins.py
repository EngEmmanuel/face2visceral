from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


DEFAULT_CT_ROOT = Path("data/aattct_visceral_subset")
DEFAULT_UTK_ROOT = Path("data/utkdataset")
DEFAULT_OUTPUT_PATH = Path("data/paired_face_ct/paired_manifest.csv")
DEFAULT_AGE_BIN_EDGES = [0, 20, 30, 40, 50, 60, 200]


@dataclass(frozen=True)
class UtkRecord:
    image_path: Path
    age: int
    sex: int
    sex_label: str


@dataclass(frozen=True)
class CtRecord:
    patient_id: str
    image_path: Path
    label_path: Path
    visceral_pixels: int
    visceral_ratio: float


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    age_bin: str
    age_bin_index: int
    utk_image_path: Path
    utk_age: int
    utk_sex: int
    utk_sex_label: str
    ct_patient_id: str
    ct_image_path: Path
    ct_label_path: Path
    visceral_pixels: int
    visceral_ratio: float


def parse_age_bin_edges(raw: str | None) -> List[int]:
    if raw is None:
        return DEFAULT_AGE_BIN_EDGES.copy()

    parts = [part.strip() for part in raw.split(",") if part.strip()]
    edges = [int(part) for part in parts]

    if len(edges) < 2:
        raise ValueError("Age bin edges must contain at least two integers")
    if any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)):
        raise ValueError("Age bin edges must be strictly increasing")

    return edges


def age_to_bin_index(age: int, edges: List[int]) -> int | None:
    for index in range(len(edges) - 1):
        left = edges[index]
        right = edges[index + 1]
        if left <= age < right:
            return index

    if age == edges[-1]:
        return len(edges) - 2

    return None


def bin_label(bin_index: int, edges: List[int]) -> str:
    left = edges[bin_index]
    right = edges[bin_index + 1] - 1
    return f"{left}-{right}"


def resolve_path(root: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    return path if path.is_absolute() else (root / path)


def load_utk_records(utk_root: Path, edges: List[int]) -> Dict[int, List[UtkRecord]]:
    metadata_path = utk_root / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"UTK metadata not found: {metadata_path}")

    bins: Dict[int, List[UtkRecord]] = {index: [] for index in range(len(edges) - 1)}

    with metadata_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "age", "sex"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"UTK metadata missing required columns. "
                f"Found={reader.fieldnames}, required={sorted(required)}"
            )

        for row in reader:
            age = int(row["age"])
            sex = int(row["sex"])
            index = age_to_bin_index(age, edges)
            if index is None:
                continue

            image_path = resolve_path(utk_root, row["image_path"])
            if not image_path.exists():
                continue

            bins[index].append(
                UtkRecord(
                    image_path=image_path,
                    age=age,
                    sex=sex,
                    sex_label=row.get("sex_label", "female" if sex == 1 else "male"),
                )
            )

    return bins


def load_ct_records(ct_root: Path) -> List[CtRecord]:
    manifest_path = ct_root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"CT manifest not found: {manifest_path}")

    records: List[CtRecord] = []

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"patient_id", "image_path", "label_path", "visceral_pixels", "visceral_ratio"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"CT manifest missing required columns. "
                f"Found={reader.fieldnames}, required={sorted(required)}"
            )

        for row in reader:
            image_path = resolve_path(ct_root, row["image_path"])
            label_path = resolve_path(ct_root, row["label_path"])
            if not image_path.exists() or not label_path.exists():
                continue

            records.append(
                CtRecord(
                    patient_id=row["patient_id"],
                    image_path=image_path,
                    label_path=label_path,
                    visceral_pixels=int(row["visceral_pixels"]),
                    visceral_ratio=float(row["visceral_ratio"]),
                )
            )

    return records


def split_ct_into_rank_bins(ct_records: List[CtRecord], num_bins: int) -> Dict[int, List[CtRecord]]:
    sorted_records = sorted(ct_records, key=lambda x: x.visceral_ratio)
    bins: Dict[int, List[CtRecord]] = {index: [] for index in range(num_bins)}

    total = len(sorted_records)
    base = total // num_bins
    remainder = total % num_bins

    start = 0
    for index in range(num_bins):
        size = base + (1 if index < remainder else 0)
        end = start + size
        bins[index] = sorted_records[start:end]
        start = end

    return bins


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create paired face↔CT examples by age bins using UTK ages and CT visceral-ratio ranking. "
            "Higher visceral ratio is mapped to older age bins by default."
        )
    )
    parser.add_argument(
        "--ct-root",
        type=Path,
        default=DEFAULT_CT_ROOT,
        help="Root of CT subset containing manifest.csv, images/, labels/",
    )
    parser.add_argument(
        "--utk-root",
        type=Path,
        default=DEFAULT_UTK_ROOT,
        help="Root of UTK subset containing metadata.csv and images/",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to write the paired manifest CSV",
    )
    parser.add_argument(
        "--age-bin-edges",
        type=str,
        default=None,
        help='Comma-separated integer edges, e.g. "0,20,30,40,50,60,200"',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for pairing inside each bin",
    )
    parser.add_argument(
        "--younger-higher-visceral",
        action="store_true",
        help="If set, invert mapping so higher visceral ratio maps to younger bins",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ct_root = args.ct_root.resolve()
    utk_root = args.utk_root.resolve()
    output_path = args.output_path.resolve()
    edges = parse_age_bin_edges(args.age_bin_edges)

    rng = random.Random(args.seed)

    utk_bins = load_utk_records(utk_root=utk_root, edges=edges)
    ct_records = load_ct_records(ct_root=ct_root)

    num_bins = len(edges) - 1
    ct_rank_bins = split_ct_into_rank_bins(ct_records=ct_records, num_bins=num_bins)

    mapping: Dict[int, int] = {}
    for age_bin_index in range(num_bins):
        if args.younger_higher_visceral:
            ct_bin_index = num_bins - 1 - age_bin_index
        else:
            ct_bin_index = age_bin_index
        mapping[age_bin_index] = ct_bin_index

    pairs: List[PairRecord] = []
    unpaired_ct = 0
    unpaired_utk = 0

    for age_bin_index in range(num_bins):
        utk_records = list(utk_bins.get(age_bin_index, []))
        ct_records_for_bin = list(ct_rank_bins.get(mapping[age_bin_index], []))

        rng.shuffle(utk_records)
        rng.shuffle(ct_records_for_bin)

        pair_count = min(len(utk_records), len(ct_records_for_bin))
        unpaired_utk += len(utk_records) - pair_count
        unpaired_ct += len(ct_records_for_bin) - pair_count

        label = bin_label(age_bin_index, edges)

        for idx in range(pair_count):
            utk_row = utk_records[idx]
            ct_row = ct_records_for_bin[idx]
            pair_id = f"{label.replace('-', '_')}_{idx:05d}"

            pairs.append(
                PairRecord(
                    pair_id=pair_id,
                    age_bin=label,
                    age_bin_index=age_bin_index,
                    utk_image_path=utk_row.image_path,
                    utk_age=utk_row.age,
                    utk_sex=utk_row.sex,
                    utk_sex_label=utk_row.sex_label,
                    ct_patient_id=ct_row.patient_id,
                    ct_image_path=ct_row.image_path,
                    ct_label_path=ct_row.label_path,
                    visceral_pixels=ct_row.visceral_pixels,
                    visceral_ratio=ct_row.visceral_ratio,
                )
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pair_id",
                "age_bin",
                "age_bin_index",
                "utk_image_path",
                "utk_age",
                "utk_sex",
                "utk_sex_label",
                "ct_patient_id",
                "ct_image_path",
                "ct_label_path",
                "visceral_pixels",
                "visceral_ratio",
            ],
        )
        writer.writeheader()
        for row in pairs:
            writer.writerow(
                {
                    "pair_id": row.pair_id,
                    "age_bin": row.age_bin,
                    "age_bin_index": row.age_bin_index,
                    "utk_image_path": str(row.utk_image_path),
                    "utk_age": row.utk_age,
                    "utk_sex": row.utk_sex,
                    "utk_sex_label": row.utk_sex_label,
                    "ct_patient_id": row.ct_patient_id,
                    "ct_image_path": str(row.ct_image_path),
                    "ct_label_path": str(row.ct_label_path),
                    "visceral_pixels": row.visceral_pixels,
                    "visceral_ratio": f"{row.visceral_ratio:.8f}",
                }
            )

    print(f"Wrote {len(pairs)} paired rows to: {output_path}")
    print(f"Unpaired UTK rows (strict bin pairing): {unpaired_utk}")
    print(f"Unpaired CT rows (strict bin pairing): {unpaired_ct}")

    print("Per-bin summary:")
    for age_bin_index in range(num_bins):
        label = bin_label(age_bin_index, edges)
        utk_count = len(utk_bins.get(age_bin_index, []))
        ct_count = len(ct_rank_bins.get(mapping[age_bin_index], []))
        paired_count = min(utk_count, ct_count)
        print(f"  {label}: utk={utk_count}, ct={ct_count}, paired={paired_count}")


if __name__ == "__main__":
    main()
