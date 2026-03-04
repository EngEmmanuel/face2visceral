from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


MALE = 0
FEMALE = 1
DEFAULT_SOURCE_DIR = Path(
    "data/raw/UTKFace"
)
DEFAULT_OUTPUT_DIR = Path("data/utkdataset")


@dataclass(frozen=True)
class UTKRecord:
    age: int
    sex: int
    file_path: Path


def parse_utk_metadata(file_name: str) -> tuple[int, int] | None:
    parts = file_name.split("_")
    if len(parts) < 3:
        return None

    try:
        age = int(parts[0])
        sex = int(parts[1])
    except ValueError:
        return None

    if sex not in (MALE, FEMALE):
        return None

    return age, sex


def iter_image_files(source_dir: Path) -> Iterable[Path]:
    for path in source_dir.iterdir():
        if path.is_file() and ".jpg" in path.name.lower():
            yield path


def build_index(source_dir: Path) -> Dict[int, Dict[int, List[UTKRecord]]]:
    index: Dict[int, Dict[int, List[UTKRecord]]] = defaultdict(lambda: defaultdict(list))

    for image_path in iter_image_files(source_dir):
        parsed = parse_utk_metadata(image_path.name)
        if parsed is None:
            continue

        age, sex = parsed
        index[age][sex].append(UTKRecord(age=age, sex=sex, file_path=image_path))

    return index


def sample_balanced_records(
    index: Dict[int, Dict[int, List[UTKRecord]]],
    per_sex_count: int,
    rng: random.Random,
) -> tuple[list[UTKRecord], dict[int, dict[str, int]]]:
    sampled_records: list[UTKRecord] = []
    skipped_ages: dict[int, dict[str, int]] = {}

    for age in sorted(index.keys()):
        male_records = index[age].get(MALE, [])
        female_records = index[age].get(FEMALE, [])

        if len(male_records) < per_sex_count or len(female_records) < per_sex_count:
            skipped_ages[age] = {"male": len(male_records), "female": len(female_records)}
            continue

        sampled_records.extend(rng.sample(male_records, per_sex_count))
        sampled_records.extend(rng.sample(female_records, per_sex_count))

    return sampled_records, skipped_ages


def write_output(records: list[UTKRecord], output_dir: Path, source_root: Path) -> Path:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["image_path", "file_name", "age", "sex", "sex_label", "source_path"],
        )
        writer.writeheader()

        for record in records:
            destination = images_dir / record.file_path.name
            if not destination.exists():
                shutil.copy2(record.file_path, destination)

            writer.writerow(
                {
                    "image_path": str(destination.relative_to(output_dir)),
                    "file_name": destination.name,
                    "age": record.age,
                    "sex": record.sex,
                    "sex_label": "male" if record.sex == MALE else "female",
                    "source_path": str(record.file_path.relative_to(source_root)),
                }
            )

    return metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a balanced UTKFace subset with 50 male and 50 female samples per age. "
            "Ages without enough samples are skipped."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Path to the raw UTKFace image directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where sampled data and metadata.csv are written.",
    )
    parser.add_argument(
        "--per-sex-count",
        type=int,
        default=50,
        help="How many male/female images to sample for each age.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    index = build_index(source_dir)
    rng = random.Random(args.seed)
    sampled_records, skipped_ages = sample_balanced_records(
        index=index,
        per_sex_count=args.per_sex_count,
        rng=rng,
    )

    metadata_path = write_output(
        records=sampled_records,
        output_dir=output_dir,
        source_root=source_dir,
    )

    kept_ages = len({record.age for record in sampled_records})
    total_images = len(sampled_records)

    print(f"Saved {total_images} images across {kept_ages} ages to: {output_dir}")
    print(f"Metadata file: {metadata_path}")

    if skipped_ages:
        print("Skipped ages due to insufficient male/female samples:")
        for age, counts in skipped_ages.items():
            print(f"  age={age}: male={counts['male']}, female={counts['female']}")


if __name__ == "__main__":
    main()
