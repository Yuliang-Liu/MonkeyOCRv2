#!/usr/bin/env python3

import argparse
import json
import os
import random
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import IO, Iterable, List, Sequence, Tuple

from PIL import Image


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Prepare HME100K and merged UniMER1M+HME100K train set following UniMERNet issue #14."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--archive", type=Path, default=repo_root / "data" / "archive.zip")
    parser.add_argument("--raw-root", type=Path, default=repo_root / "data" / "HME100K_raw")
    parser.add_argument(
        "--hme-output-root",
        type=Path,
        default=repo_root / "data" / "HME100K_unimernet_train",
    )
    parser.add_argument(
        "--merged-root",
        type=Path,
        default=repo_root / "data" / "UniMER1M_HME100K_merged",
    )
    parser.add_argument(
        "--unimer-root",
        "--unimerm-root",
        dest="unimerm_root",
        type=Path,
        default=repo_root / "data" / "UniMER1M",
        help="Existing UniMER1M root with images/ and train.txt.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--convert-workers", type=int, default=8)
    parser.add_argument("--materialize-workers", type=int, default=32)
    parser.add_argument("--materialize-chunk-size", type=int, default=8192)
    parser.add_argument(
        "--link-mode",
        choices=["hardlink", "symlink"],
        default="hardlink",
        help="How to materialize merged images.",
    )
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--keep-merge-manifest", action="store_true")
    parser.add_argument("--reuse-existing-manifest", action="store_true")
    parser.add_argument("--reset-merged-output", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def count_pngs(path: Path) -> int:
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".png")


def reset_merged_output(merged_root: Path) -> None:
    images_dir = merged_root / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    for file_name in ["train.txt", "summary.json"]:
        target = merged_root / file_name
        if target.exists():
            target.unlink()


def extract_archive(archive_path: Path, raw_root: Path) -> None:
    sentinel = raw_root / "HME100k" / "train" / "train_labels.txt"
    if sentinel.exists():
        print(f"[extract] skip, found {sentinel}")
        return
    ensure_dir(raw_root)
    print(f"[extract] unzip {archive_path} -> {raw_root}")
    subprocess.run(
        ["unzip", "-n", str(archive_path), "-d", str(raw_root)],
        check=True,
    )


def load_hme_train_samples(train_labels_path: Path, train_images_dir: Path, seed: int) -> List[Tuple[Path, str]]:
    samples: List[Tuple[Path, str]] = []
    with train_labels_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            try:
                image_name, caption = line.split("\t", 1)
            except ValueError as exc:
                raise ValueError(f"Invalid HME100K label format at line {line_no}: {line!r}") from exc
            image_path = train_images_dir / image_name
            if not image_path.exists():
                raise FileNotFoundError(f"Missing HME100K image: {image_path}")
            samples.append((image_path, caption))

    rng = random.Random(seed)
    rng.shuffle(samples)
    return samples


def convert_one_image(item: Tuple[int, Path, Path]) -> None:
    _, src_path, dst_path = item
    if dst_path.exists():
        return
    with Image.open(src_path) as img:
        img.save(dst_path)


def convert_hme_train(raw_root: Path, output_root: Path, seed: int, workers: int) -> dict:
    train_labels_path = raw_root / "HME100k" / "train" / "train_labels.txt"
    train_images_dir = raw_root / "HME100k" / "train" / "train_images"
    if not train_labels_path.exists():
        raise FileNotFoundError(f"Missing {train_labels_path}")

    samples = load_hme_train_samples(train_labels_path, train_images_dir, seed)
    images_dir = output_root / "images"
    ensure_dir(images_dir)

    print(f"[convert] converting {len(samples)} HME100K train images into {images_dir}")
    tasks: List[Tuple[int, Path, Path]] = []
    for idx, (src_path, _) in enumerate(samples):
        dst_path = images_dir / f"{idx:07d}.png"
        tasks.append((idx, src_path, dst_path))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(convert_one_image, task) for task in tasks]
        for future in futures:
            future.result()

    train_txt = output_root / "train.txt"
    mapping_tsv = output_root / "source_mapping.tsv"
    summary_json = output_root / "summary.json"
    with train_txt.open("w", encoding="utf-8") as train_f, mapping_tsv.open("w", encoding="utf-8") as map_f:
        map_f.write("new_index\tnew_image\toriginal_image\tcaption\n")
        for idx, (src_path, caption) in enumerate(samples):
            train_f.write(caption + "\n")
            map_f.write(f"{idx}\t{idx:07d}.png\t{src_path.name}\t{caption}\n")

    image_count = count_pngs(images_dir)
    line_count = count_lines(train_txt)
    summary = {
        "seed": seed,
        "source_labels": str(train_labels_path),
        "source_images": str(train_images_dir),
        "output_images": str(images_dir),
        "output_annotation": str(train_txt),
        "image_count": image_count,
        "line_count": line_count,
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if image_count != len(samples) or line_count != len(samples):
        raise RuntimeError(
            f"HME100K conversion mismatch: samples={len(samples)}, images={image_count}, lines={line_count}"
        )
    print(f"[convert] done: {image_count} images, {line_count} annotations")
    return summary


def iter_valid_pairs(images_dir: Path, annotation_path: Path) -> Iterable[Tuple[str, str]]:
    with annotation_path.open("r", encoding="utf-8") as f:
        for idx, raw_line in enumerate(f):
            equation = raw_line.rstrip("\n")
            if equation == "":
                continue
            image_path = images_dir / f"{idx:07d}.png"
            if image_path.exists():
                yield (str(image_path), equation)


def build_merge_manifest(
    datasets: Sequence[Tuple[str, Path, Path]], manifest_path: Path, seed: int
) -> Tuple[int, dict]:
    ensure_dir(manifest_path.parent)
    rng = random.Random(seed)
    counts = {}
    total = 0
    with manifest_path.open("w", encoding="utf-8") as out_f:
        for name, images_dir, annotation_path in datasets:
            count = 0
            for image_path, equation in iter_valid_pairs(images_dir, annotation_path):
                random_key = rng.getrandbits(64)
                out_f.write(f"{random_key:020d}\t{image_path}\t{equation}\n")
                count += 1
                total += 1
            counts[name] = count
            print(f"[merge] collected {count} valid pairs from {name}")
    print(f"[merge] manifest has {total} pairs")
    return total, counts


def flush_materialize_chunk(
    chunk: List[Tuple[str, str]],
    train_f: IO[str],
    images_dir: Path,
    start_index: int,
    link_mode: str,
) -> int:
    if not chunk:
        return 0
    written = 0
    for image_path, equation in chunk:
        if not Path(image_path).exists():
            continue
        dst_path = images_dir / f"{start_index + written:07d}.png"
        if not dst_path.exists():
            if link_mode == "hardlink":
                os.link(image_path, dst_path)
            else:
                os.symlink(image_path, dst_path)
        train_f.write(equation + "\n")
        written += 1
    return written


def materialize_merged_dataset(
    sorted_manifest_path: Path,
    merged_root: Path,
    total_pairs: int,
    keep_manifest: bool,
    workers: int,
    chunk_size: int,
    reset_output: bool,
    link_mode: str,
) -> dict:
    ensure_dir(merged_root)
    if reset_output:
        reset_merged_output(merged_root)
    images_dir = merged_root / "images"
    ensure_dir(images_dir)
    train_txt = merged_root / "train.txt"

    print(f"[merge] materializing merged dataset into {merged_root}")
    written_pairs = 0
    with sorted_manifest_path.open("r", encoding="utf-8") as manifest_f, train_txt.open(
        "w", encoding="utf-8"
    ) as train_f:
        chunk: List[Tuple[str, str]] = []
        for raw_line in manifest_f:
            _, image_path, equation = raw_line.rstrip("\n").split("\t", 2)
            chunk.append((image_path, equation))
            if len(chunk) >= chunk_size:
                written_pairs += flush_materialize_chunk(
                    chunk=chunk,
                    train_f=train_f,
                    images_dir=images_dir,
                    start_index=written_pairs,
                    link_mode=link_mode,
                )
                chunk = []
        written_pairs += flush_materialize_chunk(
            chunk=chunk,
            train_f=train_f,
            images_dir=images_dir,
            start_index=written_pairs,
            link_mode=link_mode,
        )

    image_count = count_pngs(images_dir)
    line_count = count_lines(train_txt)
    summary = {
        "output_images": str(images_dir),
        "output_annotation": str(train_txt),
        "image_count": image_count,
        "line_count": line_count,
        "manifest_pairs": total_pairs,
        "written_pairs": written_pairs,
        "skipped_missing_source": total_pairs - written_pairs,
    }

    if image_count != line_count or line_count != written_pairs:
        raise RuntimeError(
            f"Merged dataset mismatch: written={written_pairs}, images={image_count}, lines={line_count}"
        )

    if not keep_manifest:
        manifest_dir = sorted_manifest_path.parent
        shutil.rmtree(manifest_dir)

    print(f"[merge] done: {image_count} images, {line_count} annotations")
    return summary


def merge_datasets(
    merged_root: Path,
    datasets: Sequence[Tuple[str, Path, Path]],
    seed: int,
    keep_manifest: bool,
    workers: int,
    chunk_size: int,
    reuse_existing_manifest: bool,
    reset_output: bool,
    link_mode: str,
) -> dict:
    work_root = merged_root / "_merge_work"
    manifest_path = work_root / "pairs.tsv"
    sorted_manifest_path = work_root / "pairs.sorted.tsv"
    counts = None
    if reuse_existing_manifest and sorted_manifest_path.exists():
        total_pairs = count_lines(sorted_manifest_path)
        print(f"[merge] reuse existing sorted manifest: {sorted_manifest_path}")
    else:
        total_pairs, counts = build_merge_manifest(datasets, manifest_path, seed)
        print(f"[merge] sorting manifest with system sort: {manifest_path}")
        subprocess.run(
            ["sort", "-t", "\t", "-k1,1", str(manifest_path), "-o", str(sorted_manifest_path)],
            check=True,
        )
    summary = materialize_merged_dataset(
        sorted_manifest_path=sorted_manifest_path,
        merged_root=merged_root,
        total_pairs=total_pairs,
        keep_manifest=keep_manifest,
        workers=workers,
        chunk_size=chunk_size,
        reset_output=reset_output,
        link_mode=link_mode,
    )
    summary.update(
        {
            "seed": seed,
            "sources": {
                name: {
                    "images": str(images_dir),
                    "annotation": str(annotation_path),
                    "valid_pairs": counts[name] if counts is not None else None,
                }
                for name, images_dir, annotation_path in datasets
            },
        }
    )
    with (merged_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    args = parse_args()

    archive_path = args.archive.resolve()
    raw_root = args.raw_root.resolve()
    hme_output_root = args.hme_output_root.resolve()
    merged_root = args.merged_root.resolve()
    unimerm_root = args.unimerm_root.resolve()

    if not args.skip_extract:
        extract_archive(archive_path, raw_root)

    if args.skip_convert:
        hme_summary_path = hme_output_root / "summary.json"
        if hme_summary_path.exists():
            with hme_summary_path.open("r", encoding="utf-8") as f:
                hme_summary = json.load(f)
        else:
            hme_summary = None
        print("[convert] skipped by flag")
    else:
        hme_summary = convert_hme_train(raw_root, hme_output_root, args.seed, args.convert_workers)

    if args.skip_merge:
        print("[merge] skipped by flag")
        return

    hme_images = hme_output_root / "images"
    hme_train_txt = hme_output_root / "train.txt"
    if not hme_images.exists() or not hme_train_txt.exists():
        raise FileNotFoundError("HME100K converted train set is missing; cannot merge.")

    datasets = [
        ("UniMER1M_existing", unimerm_root / "images", unimerm_root / "train.txt"),
        ("HME100K_converted", hme_images, hme_train_txt),
    ]
    merged_summary = merge_datasets(
        merged_root=merged_root,
        datasets=datasets,
        seed=args.seed,
        keep_manifest=args.keep_merge_manifest,
        workers=args.materialize_workers,
        chunk_size=args.materialize_chunk_size,
        reuse_existing_manifest=args.reuse_existing_manifest,
        reset_output=args.reset_merged_output,
        link_mode=args.link_mode,
    )

    final_summary = {
        "archive": str(archive_path),
        "raw_root": str(raw_root),
        "hme_output_root": str(hme_output_root),
        "merged_root": str(merged_root),
        "seed": args.seed,
        "hme_summary": hme_summary,
        "merged_summary": merged_summary,
    }
    with (args.repo_root.resolve() / "HME100K_ISSUE14_PREP_SUMMARY.json").open("w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)
    print("[done] wrote HME100K_ISSUE14_PREP_SUMMARY.json")


if __name__ == "__main__":
    main()
