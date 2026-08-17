#!/usr/bin/env python3
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_ROOT = ROOT / 'data' / 'openocr'


def link(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f'Missing downloaded dataset: {source.relative_to(ROOT)}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(f'Refusing to replace existing path: {destination.relative_to(ROOT)}')
    destination.symlink_to(Path(os.path.relpath(source, destination.parent)), target_is_directory=True)


def link_children(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for child in sorted(source.iterdir()):
        if child.is_dir():
            link(child, destination / child.name)


def main() -> None:
    english_root = ROOT / 'data' / 'english'
    link(DOWNLOAD_ROOT / 'Union14M-L-LMDB-Filtered', english_root / 'train')
    link_children(DOWNLOAD_ROOT / 'evaluation', english_root / 'test')
    link(DOWNLOAD_ROOT / 'u14m', english_root / 'test' / 'u14m')
    link(DOWNLOAD_ROOT / 'OST', english_root / 'test' / 'ost')
    print('English datasets are ready under data/english.')


if __name__ == '__main__':
    main()
